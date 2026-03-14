#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif
#ifdef ENABLE_MX_API
#include "mx/rope_mx.hpp"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    ASSERT(out->ndim() == 3 && in->ndim() == 3, "rope: in/out must be 3D tensors.");
    ASSERT(out->shape()[2] % 2 == 0, "rope: head dimension must be even for RoPE.");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64.");
    ASSERT(pos_ids->ndim() == 1, "rope: pos_ids must be a 1D tensor.");
    ASSERT(out->shape()[0] == pos_ids->shape()[0],
           "rope: out and pos_ids must have the same seq len.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous.");

    const size_t seq_len = in->shape()[0];
    const size_t head_num = in->shape()[1];
    const size_t head_dim = in->shape()[2];

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), reinterpret_cast<const std::int64_t *>(pos_ids->data()),
                         out->dtype(), seq_len, head_num, head_dim, theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), reinterpret_cast<const std::int64_t *>(pos_ids->data()),
                            out->dtype(), seq_len, head_num, head_dim, theta);
#endif
#ifdef ENABLE_MX_API
    case LLAISYS_DEVICE_MX:
        return mx::rope(out->data(), in->data(), reinterpret_cast<const std::int64_t *>(pos_ids->data()),
                            out->dtype(), seq_len, head_num, head_dim, theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
