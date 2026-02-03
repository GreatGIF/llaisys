#include "op.hpp"
#include <cmath>

template <typename T1, typename T2>
void rope_(T1* out, const T1* in, std::vector<size_t> in_shape, const T2* pos_ids, float theta) {
    size_t seq_len = in_shape[0];
    size_t head_num = in_shape[1];
    size_t head_dim = in_shape[2];
    size_t half_head_dim = head_dim / 2;
    std::vector<float> freqs_sin(seq_len * half_head_dim, 0);
    std::vector<float> freqs_cos(seq_len * half_head_dim, 0);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < half_head_dim; j++) {
            float curr_pos_id;
            if constexpr (std::is_same_v<T1, llaisys::bf16_t> || std::is_same_v<T1, llaisys::fp16_t>) {
                curr_pos_id = llaisys::utils::cast<float>(pos_ids[i]);
            } else {
                curr_pos_id = static_cast<float>(pos_ids[i]);
            }
            float freqs = curr_pos_id / std::pow(theta, static_cast<float>(2 * j) / static_cast<float>(head_dim));
            freqs_sin[i * half_head_dim + j] = sin(freqs);
            freqs_cos[i * half_head_dim + j] = cos(freqs);
        }
    }
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < head_num; j++) {
            for (size_t k1 = 0; k1 < half_head_dim; k1++) {
                size_t idx1 = (i * head_num + j) * head_dim + k1;
                size_t idx2 = idx1 + half_head_dim;
                float in1 = 0, in2 = 0;
                if constexpr (std::is_same_v<T1, llaisys::bf16_t> || std::is_same_v<T1, llaisys::fp16_t>) {
                    in1 = llaisys::utils::cast<float>(in[idx1]);
                    in2 = llaisys::utils::cast<float>(in[idx2]);
                } else {
                    in1 = in[idx1];
                    in2 = in[idx2];
                }
                if constexpr (std::is_same_v<T1, llaisys::bf16_t> || std::is_same_v<T1, llaisys::fp16_t>) {
                    out[idx1] = llaisys::utils::cast<T1>(in1* freqs_cos[i * half_head_dim + k1] 
                        - in2 * freqs_sin[i * half_head_dim + k1]);
                    out[idx2] = llaisys::utils::cast<T1>(in1 * freqs_sin[i * half_head_dim + k1] 
                        + in2 * freqs_cos[i * half_head_dim + k1]);
                } else {
                    out[idx1] = in1 * freqs_cos[i * half_head_dim + k1] - in2 * freqs_sin[i * half_head_dim + k1];
                    out[idx2] = in1 * freqs_sin[i * half_head_dim + k1] + in2 * freqs_cos[i * half_head_dim + k1];
                }
            }
        }
    }
}

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // TO_BE_IMPLEMENTED();
    if (out->ndim() != 3) {
        throw std::invalid_argument("rope: out must be a 3D tensor.");
    }
    if (out->shape()[2] % 2 != 0) {
        throw std::invalid_argument("rope: Head dimension must be even for RoPE.");
    }
    if (out->dtype() != in->dtype()) {
        throw std::invalid_argument("rope: in and out must have the same dtype.");
    }
    if (pos_ids->dtype() != LLAISYS_DTYPE_I64) {
        throw std::invalid_argument("rope: pos_ids must be int64.");
    }
    if (out->shape()[0] != pos_ids->shape()[0]) {
        throw std::invalid_argument("rope: out and pos_ids must have the same seq len.");
    }
    
    llaisysDataType_t type = in->dtype();
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), in->shape(), 
        reinterpret_cast<const int64_t *>(pos_ids->data()), theta);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), in->shape(), 
        reinterpret_cast<const int64_t *>(pos_ids->data()), theta);
        break;
    case LLAISYS_DTYPE_F16:
        rope_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), in->shape(), 
        reinterpret_cast<const int64_t *>(pos_ids->data()), theta);
        break;
    default:
        throw std::invalid_argument("rope: unsupported dtype.");
    }

}
} // namespace llaisys::ops
