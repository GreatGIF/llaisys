#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif
#ifdef ENABLE_MX_API
#include "mx/argmax_mx.hpp"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    
    ASSERT(vals->ndim() == 1, "argmax: vals must be a 1D tensor");
    ASSERT(vals->numel() > 0, "argmax: vals is empty");
    ASSERT(max_val->shape() == std::vector<size_t>{1}, "argmax: max_val must have shape [1]");
    ASSERT(max_idx->shape() == std::vector<size_t>{1}, "argmax: max_idx must have shape [1]");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must have dtype int64");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "argmax: all tensors must be contiguous");

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(reinterpret_cast<int64_t *>(max_idx->data()), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(reinterpret_cast<int64_t *>(max_idx->data()), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#endif
#ifdef ENABLE_MX_API
    case LLAISYS_DEVICE_MX:
        return mx::argmax(reinterpret_cast<int64_t *>(max_idx->data()), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
