#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif
#ifdef ENABLE_MX_API
#include "mx/linear_mx.hpp"
#endif


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

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), (bias ? bias->data() : nullptr), out->dtype(), M, N, K);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), (bias ? bias->data() : nullptr), out->dtype(), M, N, K);
#endif
#ifdef ENABLE_MX_API
    case LLAISYS_DEVICE_MX:
        return mx::linear(out->data(), in->data(), weight->data(), (bias ? bias->data() : nullptr), out->dtype(), M, N, K);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
