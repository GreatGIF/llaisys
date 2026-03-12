#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void rope_(T *out, const T *in, size_t seq_len, size_t head_num, size_t head_dim,
           const std::int64_t *pos_ids, float theta) {
    size_t half_head_dim = head_dim / 2;
    std::vector<float> freqs_sin(seq_len * half_head_dim, 0);
    std::vector<float> freqs_cos(seq_len * half_head_dim, 0);
#ifdef ENABLE_OPENMP
    #pragma omp parallel for schedule(static) if (seq_len >= 16)
#endif
    for (size_t i = 0; i < seq_len; i++) {
        float curr_pos_id = static_cast<float>(pos_ids[i]);
        for (size_t j = 0; j < half_head_dim; j++) {
            float freqs = curr_pos_id / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(head_dim));
            freqs_sin[i * half_head_dim + j] = sin(freqs);
            freqs_cos[i * half_head_dim + j] = cos(freqs);
        }
    }
#ifdef ENABLE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if (seq_len * head_num >= 16)
#endif
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < head_num; j++) {
            for (size_t k1 = 0; k1 < half_head_dim; k1++) {
                size_t idx1 = (i * head_num + j) * head_dim + k1;
                size_t idx2 = idx1 + half_head_dim;
                float in1 = 0, in2 = 0;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    in1 = llaisys::utils::cast<float>(in[idx1]);
                    in2 = llaisys::utils::cast<float>(in[idx2]);
                } else {
                    in1 = in[idx1];
                    in2 = in[idx2];
                }
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[idx1] = llaisys::utils::cast<T>(in1 * freqs_cos[i * half_head_dim + k1] - in2 * freqs_sin[i * half_head_dim + k1]);
                    out[idx2] = llaisys::utils::cast<T>(in1 * freqs_sin[i * half_head_dim + k1] + in2 * freqs_cos[i * half_head_dim + k1]);
                } else {
                    out[idx1] = in1 * freqs_cos[i * half_head_dim + k1] - in2 * freqs_sin[i * half_head_dim + k1];
                    out[idx2] = in1 * freqs_sin[i * half_head_dim + k1] + in2 * freqs_cos[i * half_head_dim + k1];
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::int64_t *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t head_num,
          size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in), seq_len, head_num,
                     head_dim, pos_ids, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                     reinterpret_cast<const llaisys::bf16_t *>(in), seq_len,
                     head_num, head_dim, pos_ids, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                     reinterpret_cast<const llaisys::fp16_t *>(in), seq_len,
                     head_num, head_dim, pos_ids, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
