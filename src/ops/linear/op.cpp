#include "op.hpp"

template <typename T>
void linear_(T* out, const T* in, const T* weight, const T* bias, const size_t M, const size_t N, const size_t K) {
#ifdef ENABLE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if (M * N >= 256)
#endif
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            // 使用fp32累加
            // y = x * w^T + b
            float sum = 0;
            for (size_t k = 0; k < K; k++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum = sum + llaisys::utils::cast<float>(in[m * K + k]) * llaisys::utils::cast<float>(weight[n * K + k]);
                } else {
                    sum += in[m * K + k] * weight[n * K + k];
                }
            }
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[m * N + n] = llaisys::utils::cast<T>(sum + (bias == nullptr ? 0.0f : llaisys::utils::cast<float>(bias[n])));
            } else {
                out[m * N + n] = sum + (bias == nullptr ? 0.0f : bias[n]);
            }
        }
    }
}

// template <typename T>
// void linear_(T* out, const T* in, const T* weight, const T* bias, size_t M, size_t N, size_t K) {
//     if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
//         // 使用fp32累加
//         for (size_t m = 0; m < M; ++m) {
//             for (size_t n = 0; n < N; ++n) {
//                 float sum = 0.0f;
//                 for (size_t k = 0; k < K; ++k) {
//                     float x_val = llaisys::utils::cast<float>(in[m * K + k]);
//                     float w_val = llaisys::utils::cast<float>(weight[n * K + k]);
//                     sum += x_val * w_val;
//                 }
//                 float b_val = (bias == nullptr) ? 0.0f : llaisys::utils::cast<float>(bias[n]);
//                 out[m * N + n] = llaisys::utils::cast<T>(sum + b_val);
//             }
//         }
//     } else {
//         for (size_t m = 0; m < M; ++m) {
//             for (size_t n = 0; n < N; ++n) {
//                 T sum = T(0);
//                 for (size_t k = 0; k < K; ++k) {
//                     sum += in[m * K + k] * weight[n * K + k];
//                 }
//                 out[m * N + n] = sum + (bias == nullptr ? T(0) : bias[n]);
//             }
//         }
//     }
// }


namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // TO_BE_IMPLEMENTED();
    // y = x * w^T + b
    if (in->dtype() != out->dtype()) {
        throw std::invalid_argument("linear: in and out must have the same dtype");
    }

    llaisysDataType_t type = in->dtype();
    bool is_bias = bias != nullptr;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        linear_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), 
        reinterpret_cast<const float *>(weight->data()), (is_bias ? reinterpret_cast<const float *>(bias->data()) : nullptr),
        in->shape()[0], weight->shape()[0], weight->shape()[1]);
        break;
    case LLAISYS_DTYPE_BF16:
        linear_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), 
        reinterpret_cast<const bf16_t *>(weight->data()), (is_bias ? reinterpret_cast<const bf16_t *>(bias->data()) : nullptr), 
        in->shape()[0], weight->shape()[0], weight->shape()[1]);
        break;
    case LLAISYS_DTYPE_F16:
        linear_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), 
        reinterpret_cast<const fp16_t *>(weight->data()), (is_bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr), 
        in->shape()[0], weight->shape()[0], weight->shape()[1]);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops
