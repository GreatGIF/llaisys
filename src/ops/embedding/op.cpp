#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif
#ifdef ENABLE_MX_API
#include "mx/embedding_mx.hpp"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_SAME_DEVICE(out, index, weight);

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), reinterpret_cast<const int64_t *>(index->data()), 
                              weight->data(), weight->dtype(), index->numel(), weight->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), reinterpret_cast<const int64_t *>(index->data()), 
                                 weight->data(), weight->dtype(), index->numel(), weight->shape()[1]);
#endif
#ifdef ENABLE_MX_API
    case LLAISYS_DEVICE_MX:
        return mx::embedding(out->data(), reinterpret_cast<const int64_t *>(index->data()), 
                                 weight->data(), weight->dtype(), index->numel(), weight->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

}
} // namespace llaisys::ops
