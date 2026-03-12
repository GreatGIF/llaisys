#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    ASSERT(eps != 0.0f, "rms_norm: eps must not be 0");
    ASSERT(in->ndim() == 2, "rms_norm: in must be a 2D tensor");
    ASSERT(weight->ndim() == 1, "rms_norm: weight must be a 1D tensor");
    ASSERT(weight->shape()[0] == in->shape()[1], "rms_norm: weight shape mismatch");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous");

    const size_t m = in->shape()[0];    
    const size_t n = in->shape()[1];

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), m, n, eps);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
