#include "paged_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
float load_value(const T *ptr, size_t idx) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        return llaisys::utils::cast<float>(ptr[idx]);
    } else {
        return ptr[idx];
    }
}

template <typename T>
void store_value(T *ptr, size_t idx, float value) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        ptr[idx] = llaisys::utils::cast<T>(value);
    } else {
        ptr[idx] = value;
    }
}

inline size_t cache_index(size_t slot, size_t num_kv_heads, size_t head_dim, size_t kv_head, size_t d) {
    return (slot * num_kv_heads + kv_head) * head_dim + d;
}

inline size_t q_index(size_t token_idx, size_t num_heads, size_t head_dim, size_t head, size_t d) {
    return (token_idx * num_heads + head) * head_dim + d;
}

inline int32_t slot_for_position(const std::vector<int32_t> &block_table,
                                 size_t position, size_t block_size) {
    const size_t block_index = position / block_size;
    const size_t block_offset = position % block_size;
    return block_table[block_index] * static_cast<int32_t>(block_size) + static_cast<int32_t>(block_offset);
}

template <typename T>
void store_paged_kv_cache_(T *k_cache, T *v_cache, const T *k, const T *v,
                           size_t token_count, size_t num_kv_heads, size_t head_dim,
                           const std::vector<int32_t> &slot_mapping) {
    for (size_t token_idx = 0; token_idx < token_count; ++token_idx) {
        const size_t slot = static_cast<size_t>(slot_mapping[token_idx]);
        for (size_t kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
            for (size_t d = 0; d < head_dim; ++d) {
                const size_t src_idx = cache_index(token_idx, num_kv_heads, head_dim, kv_head, d);
                const size_t dst_idx = cache_index(slot, num_kv_heads, head_dim, kv_head, d);
                k_cache[dst_idx] = k[src_idx];
                v_cache[dst_idx] = v[src_idx];
            }
        }
    }
}

template <typename T>
void paged_attention_prefill_(T *attn_val, const T *q, const T *k_cache, const T *v_cache,
                              const llaisys::core::paged_kv::PrefillBatch &batch,
                              size_t num_heads, size_t num_kv_heads, size_t head_dim,
                              float scale, size_t block_size) {
    const size_t group_size = num_heads / num_kv_heads;
    const size_t batch_size = batch.cu_seqlens_q.size() - 1;

    for (size_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
        const size_t q_begin = static_cast<size_t>(batch.cu_seqlens_q[seq_idx]);
        const size_t q_end = static_cast<size_t>(batch.cu_seqlens_q[seq_idx + 1]);
        const size_t q_len = q_end - q_begin;
        const size_t kv_len = static_cast<size_t>(batch.cu_seqlens_k[seq_idx + 1] - batch.cu_seqlens_k[seq_idx]);
        const size_t num_cached_tokens = kv_len - q_len;

        for (size_t local_q = 0; local_q < q_len; ++local_q) {
            const size_t q_token_idx = q_begin + local_q;
            const size_t q_position = static_cast<size_t>(batch.positions[q_token_idx]);
            for (size_t head = 0; head < num_heads; ++head) {
                const size_t kv_head = head / group_size;
                std::vector<float> scores(q_position + 1, 0.0f);
                float max_score = -std::numeric_limits<float>::infinity();

                for (size_t kv_pos = 0; kv_pos <= q_position; ++kv_pos) {
                    size_t slot = 0;
                    if (kv_pos < num_cached_tokens) {
                        const auto &table = batch.block_tables[seq_idx];
                        slot = static_cast<size_t>(slot_for_position(table, kv_pos, block_size));
                    } else {
                        slot = static_cast<size_t>(batch.slot_mapping[q_begin + (kv_pos - num_cached_tokens)]);
                    }
                    float score = 0.0f;
                    for (size_t d = 0; d < head_dim; ++d) {
                        score += load_value(q, q_index(q_token_idx, num_heads, head_dim, head, d)) *
                                 load_value(k_cache, cache_index(slot, num_kv_heads, head_dim, kv_head, d));
                    }
                    score *= scale;
                    scores[kv_pos] = score;
                    max_score = std::max(max_score, score);
                }

                float sum = 0.0f;
                for (float &score : scores) {
                    score = std::exp(score - max_score);
                    sum += score;
                }

                for (size_t d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (size_t kv_pos = 0; kv_pos <= q_position; ++kv_pos) {
                        size_t slot = 0;
                        if (kv_pos < num_cached_tokens) {
                            const auto &table = batch.block_tables[seq_idx];
                            slot = static_cast<size_t>(slot_for_position(table, kv_pos, block_size));
                        } else {
                            slot = static_cast<size_t>(batch.slot_mapping[q_begin + (kv_pos - num_cached_tokens)]);
                        }
                        acc += (scores[kv_pos] / sum) *
                               load_value(v_cache, cache_index(slot, num_kv_heads, head_dim, kv_head, d));
                    }
                    store_value(attn_val, q_index(q_token_idx, num_heads, head_dim, head, d), acc);
                }
            }
        }
    }
}

template <typename T>
void paged_attention_decode_(T *attn_val, const T *q, const T *k_cache, const T *v_cache,
                             const llaisys::core::paged_kv::DecodeBatch &batch,
                             size_t num_heads, size_t num_kv_heads, size_t head_dim,
                             float scale, size_t block_size) {
    const size_t group_size = num_heads / num_kv_heads;
    const size_t batch_size = batch.input_ids.size();

    for (size_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
        const size_t kv_len = static_cast<size_t>(batch.context_lens[seq_idx]);
        const auto &table = batch.block_tables[seq_idx];

        for (size_t head = 0; head < num_heads; ++head) {
            const size_t kv_head = head / group_size;
            std::vector<float> scores(kv_len, 0.0f);
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
                const size_t slot = static_cast<size_t>(slot_for_position(table, kv_pos, block_size));
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    score += load_value(q, q_index(seq_idx, num_heads, head_dim, head, d)) *
                             load_value(k_cache, cache_index(slot, num_kv_heads, head_dim, kv_head, d));
                }
                score *= scale;
                scores[kv_pos] = score;
                max_score = std::max(max_score, score);
            }

            float sum = 0.0f;
            for (float &score : scores) {
                score = std::exp(score - max_score);
                sum += score;
            }

            for (size_t d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (size_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
                    const size_t slot = static_cast<size_t>(slot_for_position(table, kv_pos, block_size));
                    acc += (scores[kv_pos] / sum) *
                           load_value(v_cache, cache_index(slot, num_kv_heads, head_dim, kv_head, d));
                }
                store_value(attn_val, q_index(seq_idx, num_heads, head_dim, head, d), acc);
            }
        }
    }
}

} // namespace

namespace llaisys::ops::cpu {

void store_paged_kv_cache(std::byte *k_cache, std::byte *v_cache,
                          const std::byte *k, const std::byte *v,
                          llaisysDataType_t type,
                          size_t token_count, size_t num_kv_heads, size_t head_dim,
                          const std::vector<int32_t> &slot_mapping) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return store_paged_kv_cache_(reinterpret_cast<float *>(k_cache), reinterpret_cast<float *>(v_cache),
                                     reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                                     token_count, num_kv_heads, head_dim, slot_mapping);
    case LLAISYS_DTYPE_BF16:
        return store_paged_kv_cache_(reinterpret_cast<llaisys::bf16_t *>(k_cache), reinterpret_cast<llaisys::bf16_t *>(v_cache),
                                     reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                                     token_count, num_kv_heads, head_dim, slot_mapping);
    case LLAISYS_DTYPE_F16:
        return store_paged_kv_cache_(reinterpret_cast<llaisys::fp16_t *>(k_cache), reinterpret_cast<llaisys::fp16_t *>(v_cache),
                                     reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                                     token_count, num_kv_heads, head_dim, slot_mapping);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

void paged_attention_prefill(std::byte *attn_val, const std::byte *q,
                             const std::byte *k_cache, const std::byte *v_cache,
                             llaisysDataType_t type,
                             const core::paged_kv::PrefillBatch &batch,
                             size_t num_heads, size_t num_kv_heads, size_t head_dim,
                             float scale, size_t block_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return paged_attention_prefill_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                                        reinterpret_cast<const float *>(k_cache), reinterpret_cast<const float *>(v_cache),
                                        batch, num_heads, num_kv_heads, head_dim, scale, block_size);
    case LLAISYS_DTYPE_BF16:
        return paged_attention_prefill_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                                        reinterpret_cast<const llaisys::bf16_t *>(k_cache), reinterpret_cast<const llaisys::bf16_t *>(v_cache),
                                        batch, num_heads, num_kv_heads, head_dim, scale, block_size);
    case LLAISYS_DTYPE_F16:
        return paged_attention_prefill_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                                        reinterpret_cast<const llaisys::fp16_t *>(k_cache), reinterpret_cast<const llaisys::fp16_t *>(v_cache),
                                        batch, num_heads, num_kv_heads, head_dim, scale, block_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

void paged_attention_decode(std::byte *attn_val, const std::byte *q,
                            const std::byte *k_cache, const std::byte *v_cache,
                            llaisysDataType_t type,
                            const core::paged_kv::DecodeBatch &batch,
                            size_t num_heads, size_t num_kv_heads, size_t head_dim,
                            float scale, size_t block_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return paged_attention_decode_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                                       reinterpret_cast<const float *>(k_cache), reinterpret_cast<const float *>(v_cache),
                                       batch, num_heads, num_kv_heads, head_dim, scale, block_size);
    case LLAISYS_DTYPE_BF16:
        return paged_attention_decode_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                                       reinterpret_cast<const llaisys::bf16_t *>(k_cache), reinterpret_cast<const llaisys::bf16_t *>(v_cache),
                                       batch, num_heads, num_kv_heads, head_dim, scale, block_size);
    case LLAISYS_DTYPE_F16:
        return paged_attention_decode_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                                       reinterpret_cast<const llaisys::fp16_t *>(k_cache), reinterpret_cast<const llaisys::fp16_t *>(v_cache),
                                       batch, num_heads, num_kv_heads, head_dim, scale, block_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
