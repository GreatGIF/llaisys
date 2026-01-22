#include "op.hpp"
#include <cmath>

template <typename T>
void rms_norm_(T* out, const T* in, const T* weight, const std::vector<size_t> &shape, float eps) {
    size_t m = shape[0];
    size_t n = shape[1];
    std::vector<float> sum(m, 0);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float x = llaisys::utils::cast<float>(in[i * n + j]);
            sum[i] += x * x;
        }
        sum[i] = std::sqrt(sum[i] / static_cast<float>(n) + eps);
    }

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[i * n + j] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in[i * n + j]) * llaisys::utils::cast<float>(weight[j]) / sum[i]);
            } else {
                out[i * n + j] = in[i * n + j] * weight[j] / sum[i];
            }
        }
    }
}

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // TO_BE_IMPLEMENTED();
    if (in->dtype() != out->dtype()) {
        throw std::invalid_argument("linear: in and out must have the same dtype");
    }
    if (weight->dtype() != out->dtype()) {
        throw std::invalid_argument("linear: weight and out must have the same dtype");
    }
    if (eps == 0.0f) {
        throw std::invalid_argument("rms_norm: eps must not be 0");
    }
    if (in->ndim() != 2) {
        throw std::invalid_argument("rms_norm: in must be a 2D tensor");
    }

    llaisysDataType_t type = in->dtype();
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), 
        reinterpret_cast<const float *>(weight->data()), in->shape(), eps);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), 
        reinterpret_cast<const bf16_t *>(weight->data()), in->shape(), eps);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), 
        reinterpret_cast<const fp16_t *>(weight->data()), in->shape(), eps);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

}
} // namespace llaisys::ops
