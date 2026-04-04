#include "paged_attention_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../utils/cuda_cast.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace {

template <typename T>
__global__ void store_paged_kv_cache_kernel(
    T *k_cache, T *v_cache, const T *k, const T *v,
    const int32_t *slot_mapping, size_t token_count,
    size_t num_kv_heads, size_t head_dim) {
    const size_t total = token_count * num_kv_heads * head_dim;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const size_t d = idx % head_dim;
    const size_t tmp = idx / head_dim;
    const size_t kv_head = tmp % num_kv_heads;
    const size_t token_idx = tmp / num_kv_heads;
    const size_t slot = static_cast<size_t>(slot_mapping[token_idx]);
    const size_t cache_idx = (slot * num_kv_heads + kv_head) * head_dim + d;
    k_cache[cache_idx] = k[idx];
    v_cache[cache_idx] = v[idx];
}

__device__ __forceinline__ int32_t slot_for_position_device(const int32_t *block_table,
                                                            size_t position, size_t block_size) {
    const size_t block_index = position / block_size;
    const size_t block_offset = position % block_size;
    return block_table[block_index] * static_cast<int32_t>(block_size) + static_cast<int32_t>(block_offset);
}

template <typename T>
__global__ void paged_attention_prefill_kernel(
    T *attn_val, const T *q, const T *k_cache, const T *v_cache,
    const int32_t *cu_seqlens_q, const int32_t *cu_seqlens_k,
    const int64_t *positions, const int32_t *slot_mapping,
    const int32_t *block_tables, size_t max_block_count,
    size_t num_seqs, size_t num_heads, size_t num_kv_heads, size_t head_dim,
    float scale, size_t block_size) {
    const size_t total = static_cast<size_t>(cu_seqlens_q[num_seqs]) * num_heads;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const size_t head = idx % num_heads;
    const size_t q_token_idx = idx / num_heads;

    size_t seq_idx = 0;
    while (seq_idx + 1 < num_seqs && static_cast<size_t>(cu_seqlens_q[seq_idx + 1]) <= q_token_idx) {
        ++seq_idx;
    }

    const size_t q_begin = static_cast<size_t>(cu_seqlens_q[seq_idx]);
    const size_t q_end = static_cast<size_t>(cu_seqlens_q[seq_idx + 1]);
    const size_t q_len = q_end - q_begin;
    const size_t kv_len = static_cast<size_t>(cu_seqlens_k[seq_idx + 1] - cu_seqlens_k[seq_idx]);
    const size_t num_cached_tokens = kv_len - q_len;
    // const size_t local_q = q_token_idx - q_begin;
    const size_t q_position = static_cast<size_t>(positions[q_token_idx]);
    const size_t group_size = num_heads / num_kv_heads;
    const size_t kv_head = head / group_size;

    float max_score = -1.0e30f;
    for (size_t kv_pos = 0; kv_pos <= q_position; ++kv_pos) {
        size_t slot = 0;
        if (kv_pos < num_cached_tokens) {
            const int32_t *table = block_tables + seq_idx * max_block_count;
            slot = static_cast<size_t>(slot_for_position_device(table, kv_pos, block_size));
        } else {
            slot = static_cast<size_t>(slot_mapping[q_begin + (kv_pos - num_cached_tokens)]);
        }
        float score = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            const size_t q_idx = (q_token_idx * num_heads + head) * head_dim + d;
            const size_t k_idx = (slot * num_kv_heads + kv_head) * head_dim + d;
            score += llaisys::utils::cuda::to_float(q[q_idx]) * llaisys::utils::cuda::to_float(k_cache[k_idx]);
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
    }

    float sum = 0.0f;
    for (size_t kv_pos = 0; kv_pos <= q_position; ++kv_pos) {
        size_t slot = 0;
        if (kv_pos < num_cached_tokens) {
            const int32_t *table = block_tables + seq_idx * max_block_count;
            slot = static_cast<size_t>(slot_for_position_device(table, kv_pos, block_size));
        } else {
            slot = static_cast<size_t>(slot_mapping[q_begin + (kv_pos - num_cached_tokens)]);
        }
        float score = 0.0f;
        for (size_t d = 0; d < head_dim; ++d) {
            const size_t q_idx = (q_token_idx * num_heads + head) * head_dim + d;
            const size_t k_idx = (slot * num_kv_heads + kv_head) * head_dim + d;
            score += llaisys::utils::cuda::to_float(q[q_idx]) * llaisys::utils::cuda::to_float(k_cache[k_idx]);
        }
        sum += expf(score * scale - max_score);
    }

    for (size_t d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (size_t kv_pos = 0; kv_pos <= q_position; ++kv_pos) {
            size_t slot = 0;
            if (kv_pos < num_cached_tokens) {
                const int32_t *table = block_tables + seq_idx * max_block_count;
                slot = static_cast<size_t>(slot_for_position_device(table, kv_pos, block_size));
            } else {
                slot = static_cast<size_t>(slot_mapping[q_begin + (kv_pos - num_cached_tokens)]);
            }
            float score = 0.0f;
            for (size_t inner = 0; inner < head_dim; ++inner) {
                const size_t q_idx = (q_token_idx * num_heads + head) * head_dim + inner;
                const size_t k_idx = (slot * num_kv_heads + kv_head) * head_dim + inner;
                score += llaisys::utils::cuda::to_float(q[q_idx]) * llaisys::utils::cuda::to_float(k_cache[k_idx]);
            }
            const float prob = expf(score * scale - max_score) / sum;
            const size_t v_idx = (slot * num_kv_heads + kv_head) * head_dim + d;
            acc += prob * llaisys::utils::cuda::to_float(v_cache[v_idx]);
        }
        const size_t out_idx = (q_token_idx * num_heads + head) * head_dim + d;
        attn_val[out_idx] = llaisys::utils::cuda::from_float<T>(acc);
    }
}

template <typename T>
__global__ void paged_attention_decode_kernel(
    T *attn_val, const T *q, const T *k_cache, const T *v_cache,
    const int32_t *context_lens, const int32_t *block_tables, size_t max_block_count,
    size_t batch_size, size_t num_heads, size_t num_kv_heads, size_t head_dim,
    float scale, size_t block_size) {
    const size_t head = static_cast<size_t>(blockIdx.x);
    const size_t seq_idx = static_cast<size_t>(blockIdx.y);
    const size_t tid = static_cast<size_t>(threadIdx.x);

    if (head >= num_heads || seq_idx >= batch_size) {
        return;
    }

    const size_t kv_len = static_cast<size_t>(context_lens[seq_idx]);
    if (kv_len == 0) {
        for (size_t d = tid; d < head_dim; d += blockDim.x) {
            const size_t out_idx = (seq_idx * num_heads + head) * head_dim + d;
            attn_val[out_idx] = llaisys::utils::cuda::from_float<T>(0.0f);
        }
        return;
    }

    const size_t group_size = num_heads / num_kv_heads;
    const size_t kv_head = head / group_size;
    const int32_t *table = block_tables + seq_idx * max_block_count;

    constexpr int kMaxVecPerThread = 8;
    float q_local[kMaxVecPerThread];
    float acc_local[kMaxVecPerThread];
    int valid_count = 0;

    const size_t q_base = (seq_idx * num_heads + head) * head_dim;
    for (int i = 0; i < kMaxVecPerThread; ++i) {
        const size_t d = tid + static_cast<size_t>(i) * blockDim.x;
        if (d < head_dim) {
            q_local[i] = llaisys::utils::cuda::to_float(q[q_base + d]);
            acc_local[i] = 0.0f;
            ++valid_count;
        } else {
            q_local[i] = 0.0f;
            acc_local[i] = 0.0f;
        }
    }

    __shared__ float s_reduce[256];
    __shared__ float s_m;
    __shared__ float s_l;
    __shared__ float s_alpha;
    __shared__ float s_beta;

    if (tid == 0) {
        s_m = -1.0e30f;
        s_l = 0.0f;
        s_alpha = 0.0f;
        s_beta = 0.0f;
    }
    __syncthreads();

    for (size_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
        const size_t slot = static_cast<size_t>(slot_for_position_device(table, kv_pos, block_size));
        const size_t k_base = (slot * num_kv_heads + kv_head) * head_dim;
        const size_t v_base = (slot * num_kv_heads + kv_head) * head_dim;

        float dot_partial = 0.0f;
        for (int i = 0; i < valid_count; ++i) {
            const size_t d = tid + static_cast<size_t>(i) * blockDim.x;
            dot_partial += q_local[i] * llaisys::utils::cuda::to_float(k_cache[k_base + d]);
        }

        s_reduce[tid] = dot_partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < static_cast<size_t>(stride)) {
                s_reduce[tid] += s_reduce[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float score = s_reduce[0] * scale;
            const float m_new = fmaxf(s_m, score);
            const float alpha = expf(s_m - m_new);
            const float beta = expf(score - m_new);
            s_l = s_l * alpha + beta;
            s_m = m_new;
            s_alpha = alpha;
            s_beta = beta;
        }
        __syncthreads();

        for (int i = 0; i < valid_count; ++i) {
            const size_t d = tid + static_cast<size_t>(i) * blockDim.x;
            const float vv = llaisys::utils::cuda::to_float(v_cache[v_base + d]);
            acc_local[i] = acc_local[i] * s_alpha + s_beta * vv;
        }
        __syncthreads();
    }

    const float inv_l = 1.0f / fmaxf(s_l, 1e-12f);
    const size_t out_base = (seq_idx * num_heads + head) * head_dim;
    for (int i = 0; i < valid_count; ++i) {
        const size_t d = tid + static_cast<size_t>(i) * blockDim.x;
        attn_val[out_base + d] = llaisys::utils::cuda::from_float<T>(acc_local[i] * inv_l);
    }
}

template <typename T>
void store_paged_kv_cache_impl(
    std::byte *k_cache, std::byte *v_cache, const std::byte *k, const std::byte *v,
    size_t token_count, size_t num_kv_heads, size_t head_dim,
    const std::vector<int32_t> &slot_mapping, cudaStream_t stream) {
    int32_t *d_slot_mapping = nullptr;
    const size_t slots_bytes = slot_mapping.size() * sizeof(int32_t);
    LLAISYS_CUDA_CHECK(cudaMalloc(&d_slot_mapping, slots_bytes));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(d_slot_mapping, slot_mapping.data(), slots_bytes,
                                       cudaMemcpyHostToDevice, stream));

    const size_t total = token_count * num_kv_heads * head_dim;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    store_paged_kv_cache_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(k_cache), reinterpret_cast<T *>(v_cache),
        reinterpret_cast<const T *>(k), reinterpret_cast<const T *>(v),
        d_slot_mapping, token_count, num_kv_heads, head_dim);
    LLAISYS_CUDA_CHECK(cudaGetLastError());
    LLAISYS_CUDA_CHECK(cudaFree(d_slot_mapping));
}

template <typename T>
void paged_attention_prefill_impl(
    std::byte *attn_val, const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const llaisys::core::paged_kv::PrefillBatch &batch,
    size_t num_heads, size_t num_kv_heads, size_t head_dim,
    float scale, size_t block_size, cudaStream_t stream) {
    const size_t num_seqs = batch.cu_seqlens_q.size() - 1;
    std::vector<int32_t> host_block_tables;
    size_t max_block_count = 0;
    if (!batch.block_tables.empty()) {
        max_block_count = batch.block_tables[0].size();
        host_block_tables.reserve(batch.block_tables.size() * max_block_count);
        for (const auto &table : batch.block_tables) {
            host_block_tables.insert(host_block_tables.end(), table.begin(), table.end());
        }
    } else {
        max_block_count = 1;
        host_block_tables.assign(num_seqs, -1);
    }

    int32_t *d_cu_q = nullptr;
    int32_t *d_cu_k = nullptr;
    int64_t *d_positions = nullptr;
    int32_t *d_slots = nullptr;
    int32_t *d_tables = nullptr;

    LLAISYS_CUDA_CHECK(cudaMalloc(&d_cu_q, batch.cu_seqlens_q.size() * sizeof(int32_t)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&d_cu_k, batch.cu_seqlens_k.size() * sizeof(int32_t)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&d_positions, batch.positions.size() * sizeof(int64_t)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&d_slots, batch.slot_mapping.size() * sizeof(int32_t)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&d_tables, host_block_tables.size() * sizeof(int32_t)));

    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(d_cu_q, batch.cu_seqlens_q.data(), batch.cu_seqlens_q.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(d_cu_k, batch.cu_seqlens_k.data(), batch.cu_seqlens_k.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(d_positions, batch.positions.data(), batch.positions.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(d_slots, batch.slot_mapping.data(), batch.slot_mapping.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(d_tables, host_block_tables.data(), host_block_tables.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    const size_t total = static_cast<size_t>(batch.cu_seqlens_q.back()) * num_heads;
    const int threads = 128;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    paged_attention_prefill_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(attn_val), reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k_cache), reinterpret_cast<const T *>(v_cache),
        d_cu_q, d_cu_k, d_positions, d_slots, d_tables, max_block_count,
        num_seqs, num_heads, num_kv_heads, head_dim, scale, block_size);
    LLAISYS_CUDA_CHECK(cudaGetLastError());

    LLAISYS_CUDA_CHECK(cudaFree(d_cu_q));
    LLAISYS_CUDA_CHECK(cudaFree(d_cu_k));
    LLAISYS_CUDA_CHECK(cudaFree(d_positions));
    LLAISYS_CUDA_CHECK(cudaFree(d_slots));
    LLAISYS_CUDA_CHECK(cudaFree(d_tables));
}

template <typename T>
void paged_attention_decode_impl(
    std::byte *attn_val, const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const llaisys::core::paged_kv::DecodeBatch &batch,
    size_t num_heads, size_t num_kv_heads, size_t head_dim,
    float scale, size_t block_size, cudaStream_t stream) {
    const size_t batch_size = batch.input_ids.size();
    const size_t max_block_count = batch.block_tables[0].size();
    std::vector<int32_t> host_block_tables;
    host_block_tables.reserve(batch.block_tables.size() * max_block_count);
    for (const auto &table : batch.block_tables) {
        host_block_tables.insert(host_block_tables.end(), table.begin(), table.end());
    }

    int32_t *d_context_lens = nullptr;
    int32_t *d_tables = nullptr;
    LLAISYS_CUDA_CHECK(cudaMalloc(&d_context_lens, batch.context_lens.size() * sizeof(int32_t)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&d_tables, host_block_tables.size() * sizeof(int32_t)));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(d_context_lens, batch.context_lens.data(), batch.context_lens.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    LLAISYS_CUDA_CHECK(cudaMemcpyAsync(d_tables, host_block_tables.data(), host_block_tables.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    const int threads = static_cast<int>(std::min<size_t>(256, std::max<size_t>(32, head_dim)));
    dim3 grid(static_cast<unsigned int>(num_heads), static_cast<unsigned int>(batch_size));
    paged_attention_decode_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<T *>(attn_val), reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k_cache), reinterpret_cast<const T *>(v_cache),
        d_context_lens, d_tables, max_block_count, batch_size,
        num_heads, num_kv_heads, head_dim, scale, block_size);
    LLAISYS_CUDA_CHECK(cudaGetLastError());

    LLAISYS_CUDA_CHECK(cudaFree(d_context_lens));
    LLAISYS_CUDA_CHECK(cudaFree(d_tables));
}

} // namespace

namespace llaisys::ops::nvidia {

void store_paged_kv_cache(std::byte *k_cache, std::byte *v_cache,
                          const std::byte *k, const std::byte *v,
                          llaisysDataType_t type,
                          size_t token_count, size_t num_kv_heads, size_t head_dim,
                          const std::vector<int32_t> &slot_mapping) {
    auto stream = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return store_paged_kv_cache_impl<float>(k_cache, v_cache, k, v, token_count, num_kv_heads, head_dim, slot_mapping, stream);
    case LLAISYS_DTYPE_BF16:
        return store_paged_kv_cache_impl<nv_bfloat16>(k_cache, v_cache, k, v, token_count, num_kv_heads, head_dim, slot_mapping, stream);
    case LLAISYS_DTYPE_F16:
        return store_paged_kv_cache_impl<half>(k_cache, v_cache, k, v, token_count, num_kv_heads, head_dim, slot_mapping, stream);
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
    auto stream = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return paged_attention_prefill_impl<float>(attn_val, q, k_cache, v_cache, batch, num_heads, num_kv_heads, head_dim, scale, block_size, stream);
    case LLAISYS_DTYPE_BF16:
        return paged_attention_prefill_impl<nv_bfloat16>(attn_val, q, k_cache, v_cache, batch, num_heads, num_kv_heads, head_dim, scale, block_size, stream);
    case LLAISYS_DTYPE_F16:
        return paged_attention_prefill_impl<half>(attn_val, q, k_cache, v_cache, batch, num_heads, num_kv_heads, head_dim, scale, block_size, stream);
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
    auto stream = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return paged_attention_decode_impl<float>(attn_val, q, k_cache, v_cache, batch, num_heads, num_kv_heads, head_dim, scale, block_size, stream);
    case LLAISYS_DTYPE_BF16:
        return paged_attention_decode_impl<nv_bfloat16>(attn_val, q, k_cache, v_cache, batch, num_heads, num_kv_heads, head_dim, scale, block_size, stream);
    case LLAISYS_DTYPE_F16:
        return paged_attention_decode_impl<half>(attn_val, q, k_cache, v_cache, batch, num_heads, num_kv_heads, head_dim, scale, block_size, stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
