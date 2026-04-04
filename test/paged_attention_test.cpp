#include "../src/core/paged_kv/paged_kv.hpp"
#include "../src/ops/paged_attention/op.hpp"
#include "../src/tensor/tensor.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using llaisys::Tensor;
using llaisys::tensor_t;
using llaisys::core::paged_kv::BlockManager;
using llaisys::core::paged_kv::DecodeBatch;
using llaisys::core::paged_kv::PrefillBatch;
using llaisys::core::paged_kv::SequenceState;

namespace {

void expect(bool cond, const std::string &message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

tensor_t make_tensor(const std::vector<size_t> &shape, const std::vector<float> &values) {
    auto tensor = Tensor::create(shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    expect(tensor->numel() == values.size(), "make_tensor: shape/value size mismatch");
    tensor->load(values.data());
    return tensor;
}

std::vector<float> to_vector(tensor_t tensor) {
    const float *ptr = reinterpret_cast<const float *>(tensor->data());
    return std::vector<float>(ptr, ptr + tensor->numel());
}

std::vector<float> reference_attention(const std::vector<float> &q_vec,
                                       const std::vector<float> &k_vec,
                                       const std::vector<float> &v_vec,
                                       size_t q_len, size_t kv_len,
                                       size_t num_heads, size_t num_kv_heads, size_t head_dim,
                                       const std::vector<size_t> &query_positions,
                                       float scale) {
    const size_t group_size = num_heads / num_kv_heads;
    std::vector<float> out(q_len * num_heads * head_dim, 0.0f);

    auto q_at = [&](size_t token, size_t head, size_t d) -> float {
        return q_vec[(token * num_heads + head) * head_dim + d];
    };
    auto k_at = [&](size_t token, size_t kv_head, size_t d) -> float {
        return k_vec[(token * num_kv_heads + kv_head) * head_dim + d];
    };
    auto v_at = [&](size_t token, size_t kv_head, size_t d) -> float {
        return v_vec[(token * num_kv_heads + kv_head) * head_dim + d];
    };

    for (size_t qt = 0; qt < q_len; ++qt) {
        const size_t q_pos = query_positions[qt];
        for (size_t head = 0; head < num_heads; ++head) {
            const size_t kv_head = head / group_size;
            std::vector<float> scores(q_pos + 1, 0.0f);
            float max_score = -1e30f;
            for (size_t kv_pos = 0; kv_pos <= q_pos; ++kv_pos) {
                float score = 0.0f;
                for (size_t d = 0; d < head_dim; ++d) {
                    score += q_at(qt, head, d) * k_at(kv_pos, kv_head, d);
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
                for (size_t kv_pos = 0; kv_pos <= q_pos; ++kv_pos) {
                    acc += (scores[kv_pos] / sum) * v_at(kv_pos, kv_head, d);
                }
                out[(qt * num_heads + head) * head_dim + d] = acc;
            }
        }
    }
    return out;
}

void expect_close(const std::vector<float> &got, const std::vector<float> &want, float tol, const std::string &message) {
    expect(got.size() == want.size(), message + ": size mismatch");
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - want[i]) > tol) {
            throw std::runtime_error(message + ": mismatch at index " + std::to_string(i) +
                                     ", got=" + std::to_string(got[i]) +
                                     ", want=" + std::to_string(want[i]));
        }
    }
}

void test_paged_attention_prefill_without_prefix() {
    constexpr size_t block_size = 2;
    constexpr size_t num_heads = 2;
    constexpr size_t num_kv_heads = 1;
    constexpr size_t head_dim = 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    BlockManager manager(8, block_size);
    SequenceState seq(0, {1, 2, 3}, block_size);
    manager.allocate(seq);
    PrefillBatch batch = BlockManager::preparePrefill({&seq}, block_size);

    auto q = make_tensor({3, num_heads, head_dim}, {
        1, 0, 0, 1,
        1, 1, 1, 0,
        0, 1, 1, 1,
    });
    auto k = make_tensor({3, num_kv_heads, head_dim}, {
        1, 0,
        0, 1,
        1, 1,
    });
    auto v = make_tensor({3, num_kv_heads, head_dim}, {
        1, 2,
        3, 4,
        5, 6,
    });
    auto k_cache = Tensor::create({8, block_size, num_kv_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    auto v_cache = Tensor::create({8, block_size, num_kv_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    auto out = Tensor::create({3, num_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);

    llaisys::ops::store_paged_kv_cache(k_cache, v_cache, k, v, batch.slot_mapping);
    llaisys::ops::paged_attention_prefill(out, q, k_cache, v_cache, batch, num_kv_heads, scale);

    auto got = to_vector(out);
    auto want = reference_attention(
        to_vector(q), to_vector(k), to_vector(v),
        3, 3, num_heads, num_kv_heads, head_dim, {0, 1, 2}, scale);
    expect_close(got, want, 1e-5f, "paged_attention_prefill without prefix");
}

void test_paged_attention_prefill_with_prefix() {
    constexpr size_t block_size = 2;
    constexpr size_t num_heads = 2;
    constexpr size_t num_kv_heads = 1;
    constexpr size_t head_dim = 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    BlockManager manager(8, block_size);
    auto k_cache = Tensor::create({8, block_size, num_kv_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    auto v_cache = Tensor::create({8, block_size, num_kv_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);

    SequenceState warmup(0, {1, 2}, block_size);
    manager.allocate(warmup);
    PrefillBatch warmup_batch = BlockManager::preparePrefill({&warmup}, block_size);
    auto warmup_k = make_tensor({2, num_kv_heads, head_dim}, {1, 0, 0, 1});
    auto warmup_v = make_tensor({2, num_kv_heads, head_dim}, {1, 2, 3, 4});
    llaisys::ops::store_paged_kv_cache(k_cache, v_cache, warmup_k, warmup_v, warmup_batch.slot_mapping);
    manager.deallocate(warmup);

    SequenceState seq(1, {1, 2, 3, 4}, block_size);
    manager.allocate(seq);
    expect(seq.numCachedTokens() == 2, "expected one cached prefix block");
    PrefillBatch batch = BlockManager::preparePrefill({&seq}, block_size);

    auto q = make_tensor({2, num_heads, head_dim}, {
        1, 1, 0, 1,
        0, 1, 1, 1,
    });
    auto k = make_tensor({2, num_kv_heads, head_dim}, {
        1, 1,
        2, 0,
    });
    auto v = make_tensor({2, num_kv_heads, head_dim}, {
        5, 6,
        7, 8,
    });
    auto out = Tensor::create({2, num_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);

    llaisys::ops::store_paged_kv_cache(k_cache, v_cache, k, v, batch.slot_mapping);
    llaisys::ops::paged_attention_prefill(out, q, k_cache, v_cache, batch, num_kv_heads, scale);

    std::vector<float> full_k = {1, 0, 0, 1, 1, 1, 2, 0};
    std::vector<float> full_v = {1, 2, 3, 4, 5, 6, 7, 8};
    auto want = reference_attention(
        to_vector(q), full_k, full_v,
        2, 4, num_heads, num_kv_heads, head_dim, {2, 3}, scale);
    expect_close(to_vector(out), want, 1e-5f, "paged_attention_prefill with prefix");
}

void test_paged_attention_prefill_multi_sequence() {
    constexpr size_t block_size = 16;
    constexpr size_t num_heads = 2;
    constexpr size_t num_kv_heads = 1;
    constexpr size_t head_dim = 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    BlockManager manager(8, block_size);
    SequenceState seq0(0, {1, 2, 3}, block_size);
    SequenceState seq1(1, {4, 5}, block_size);
    manager.allocate(seq0);
    manager.allocate(seq1);
    PrefillBatch batch = BlockManager::preparePrefill({&seq0, &seq1}, block_size);

    auto q = make_tensor({5, num_heads, head_dim}, {
        1, 0, 0, 1,
        1, 1, 1, 0,
        0, 1, 1, 1,
        1, 0, 1, 1,
        0, 1, 1, 0,
    });
    auto k = make_tensor({5, num_kv_heads, head_dim}, {
        1, 0,
        0, 1,
        1, 1,
        2, 0,
        0, 2,
    });
    auto v = make_tensor({5, num_kv_heads, head_dim}, {
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
    });
    auto k_cache = Tensor::create({8, block_size, num_kv_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    auto v_cache = Tensor::create({8, block_size, num_kv_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    auto out = Tensor::create({5, num_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);

    llaisys::ops::store_paged_kv_cache(k_cache, v_cache, k, v, batch.slot_mapping);
    llaisys::ops::paged_attention_prefill(out, q, k_cache, v_cache, batch, num_kv_heads, scale);

    auto q_all = to_vector(q);
    auto k_all = to_vector(k);
    auto v_all = to_vector(v);
    auto want0 = reference_attention(
        std::vector<float>(q_all.begin(), q_all.begin() + 3 * num_heads * head_dim),
        std::vector<float>(k_all.begin(), k_all.begin() + 3 * num_kv_heads * head_dim),
        std::vector<float>(v_all.begin(), v_all.begin() + 3 * num_kv_heads * head_dim),
        3, 3, num_heads, num_kv_heads, head_dim, {0, 1, 2}, scale);
    auto want1 = reference_attention(
        std::vector<float>(q_all.begin() + 3 * num_heads * head_dim, q_all.end()),
        std::vector<float>(k_all.begin() + 3 * num_kv_heads * head_dim, k_all.end()),
        std::vector<float>(v_all.begin() + 3 * num_kv_heads * head_dim, v_all.end()),
        2, 2, num_heads, num_kv_heads, head_dim, {0, 1}, scale);
    auto want = want0;
    want.insert(want.end(), want1.begin(), want1.end());
    expect_close(to_vector(out), want, 1e-5f, "paged_attention_prefill multi sequence");
}

void test_paged_attention_decode() {
    constexpr size_t block_size = 2;
    constexpr size_t num_heads = 2;
    constexpr size_t num_kv_heads = 1;
    constexpr size_t head_dim = 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    BlockManager manager(8, block_size);
    SequenceState seq0(0, {1, 2, 3}, block_size);
    SequenceState seq1(1, {4, 5}, block_size);
    manager.allocate(seq0);
    manager.allocate(seq1);

    auto k_cache = Tensor::create({8, block_size, num_kv_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    auto v_cache = Tensor::create({8, block_size, num_kv_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);

    PrefillBatch prefill_batch = BlockManager::preparePrefill({&seq0, &seq1}, block_size);
    auto k_prefill = make_tensor({5, num_kv_heads, head_dim}, {
        1, 0,
        0, 1,
        1, 1,
        2, 0,
        0, 2,
    });
    auto v_prefill = make_tensor({5, num_kv_heads, head_dim}, {
        1, 2,
        3, 4,
        5, 6,
        7, 8,
        9, 10,
    });
    llaisys::ops::store_paged_kv_cache(k_cache, v_cache, k_prefill, v_prefill, prefill_batch.slot_mapping);

    DecodeBatch batch = BlockManager::prepareDecode({&seq0, &seq1}, block_size);
    auto q = make_tensor({2, num_heads, head_dim}, {
        0, 1, 1, 1,
        1, 0, 0, 1,
    });
    auto out = Tensor::create({2, num_heads, head_dim}, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    llaisys::ops::paged_attention_decode(out, q, k_cache, v_cache, batch, num_kv_heads, scale);

    std::vector<float> want0 = reference_attention(
        {0, 1, 1, 1},
        {1, 0, 0, 1, 1, 1},
        {1, 2, 3, 4, 5, 6},
        1, 3, num_heads, num_kv_heads, head_dim, {2}, scale);
    std::vector<float> want1 = reference_attention(
        {1, 0, 0, 1},
        {2, 0, 0, 2},
        {7, 8, 9, 10},
        1, 2, num_heads, num_kv_heads, head_dim, {1}, scale);
    std::vector<float> want = want0;
    want.insert(want.end(), want1.begin(), want1.end());
    expect_close(to_vector(out), want, 1e-5f, "paged_attention_decode");
}

} // namespace

int main() {
    test_paged_attention_prefill_without_prefix();
    test_paged_attention_prefill_with_prefix();
    test_paged_attention_prefill_multi_sequence();
    test_paged_attention_decode();
    std::cout << "paged_attention_test passed" << std::endl;
    return 0;
}
