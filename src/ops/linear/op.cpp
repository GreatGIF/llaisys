// #include "op.hpp"

// template <typename T>
// void linear_(T* out, const T* in, const T* weight, const T* bias, const size_t M, const size_t N, const size_t K) {
// #ifdef ENABLE_OPENMP
//     #pragma omp parallel for collapse(2) schedule(static) if (M * N >= 256)
// #endif
//     for (size_t m = 0; m < M; m++) {
//         for (size_t n = 0; n < N; n++) {
//             // 使用fp32累加
//             // y = x * w^T + b
//             float sum = 0;
//             for (size_t k = 0; k < K; k++) {
//                 if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
//                     sum = sum + llaisys::utils::cast<float>(in[m * K + k]) * llaisys::utils::cast<float>(weight[n * K + k]);
//                 } else {
//                     sum += in[m * K + k] * weight[n * K + k];
//                 }
//             }
//             if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
//                 out[m * N + n] = llaisys::utils::cast<T>(sum + (bias == nullptr ? 0.0f : llaisys::utils::cast<float>(bias[n])));
//             } else {
//                 out[m * N + n] = sum + (bias == nullptr ? 0.0f : bias[n]);
//             }
//         }
//     }
// }

// namespace llaisys::ops {
// void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
//     // TO_BE_IMPLEMENTED();
//     // y = x * w^T + b
//     if (in->dtype() != out->dtype()) {
//         throw std::invalid_argument("linear: in and out must have the same dtype");
//     }

//     llaisysDataType_t type = in->dtype();
//     bool is_bias = bias != nullptr;
//     switch (type) {
//     case LLAISYS_DTYPE_F32:
//         linear_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), 
//         reinterpret_cast<const float *>(weight->data()), (is_bias ? reinterpret_cast<const float *>(bias->data()) : nullptr),
//         in->shape()[0], weight->shape()[0], weight->shape()[1]);
//         break;
//     case LLAISYS_DTYPE_BF16:
//         linear_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), 
//         reinterpret_cast<const bf16_t *>(weight->data()), (is_bias ? reinterpret_cast<const bf16_t *>(bias->data()) : nullptr), 
//         in->shape()[0], weight->shape()[0], weight->shape()[1]);
//         break;
//     case LLAISYS_DTYPE_F16:
//         linear_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), 
//         reinterpret_cast<const fp16_t *>(weight->data()), (is_bias ? reinterpret_cast<const fp16_t *>(bias->data()) : nullptr), 
//         in->shape()[0], weight->shape()[0], weight->shape()[1]);
//         break;
//     default:
//         EXCEPTION_UNSUPPORTED_DATATYPE(type);
//     }
// }
// } // namespace llaisys::ops

#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"


namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }

    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }

    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
           "Linear: out/in/weight must be 2D tensors.");
    if (bias) {
        ASSERT(bias->ndim() == 1, "Linear: bias must be a 1D tensor.");
    }

    const size_t M = in->shape()[0];
    const size_t K = in->shape()[1];
    const size_t N = weight->shape()[0];

    ASSERT(weight->shape()[1] == K,
           "Linear: weight shape mismatch, expected weight.shape[1] == in.shape[1].");
    ASSERT(out->shape()[0] == M && out->shape()[1] == N,
           "Linear: out shape mismatch, expected [M, N] where M=in.shape[0], N=weight.shape[0].");
    if (bias) {
        ASSERT(bias->shape()[0] == N,
               "Linear: bias shape mismatch, expected bias.shape[0] == weight.shape[0].");
    }

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && (!bias || bias->isContiguous()),
           "Linear: all tensors must be contiguous.");

    // if (out->deviceType() == LLAISYS_DEVICE_CPU) {
    //     return cpu::linear(out->data(), in->data(), weight->data(), (bias ? bias->data() : nullptr), out->dtype(), M, N, K);
    // }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), (bias ? bias->data() : nullptr), out->dtype(), M, N, K);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
