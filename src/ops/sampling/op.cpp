#include "op.hpp"
#include <string>
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/sampling_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/sampling_nvidia.cuh"
#endif
#ifdef ENABLE_MX_API
#include "mx/sampling_mx.hpp"
#endif

namespace llaisys::ops {

void sampling(tensor_t out, tensor_t logits, float temperature, int32_t top_k, 
              float top_p, uint64_t seed) {

    // Validate inputs
    // CHECK_SAME_DTYPE(out, logits);
    CHECK_SAME_DEVICE(out, logits);
    
    // Validate output shape: should be [batch_size]
    ASSERT(out->ndim() == 1, "Output tensor should be 1D (batch_size)");
    
    // Validate logits shape: should be [batch_size, vocab_size]
    ASSERT(logits->ndim() == 2, "Logits tensor should be 2D (batch_size, vocab_size)");
    
    size_t batch_size = logits->shape()[0];
    size_t vocab_size = logits->shape()[1];
    
    ASSERT(out->shape()[0] == batch_size, "Output batch size mismatch: expected " + 
           std::to_string(batch_size) + ", got " + std::to_string(out->shape()[0]));
    
    // Validate output dtype: should be int64
    ASSERT(out->dtype() == LLAISYS_DTYPE_I64, "Output tensor should be int64, got " + 
           std::to_string(out->dtype()));
    
    llaisys::core::context().setDevice(logits->deviceType(), logits->deviceId());
    
    switch (logits->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::sampling(out->data(), logits->data(), logits->dtype(),
                            batch_size, vocab_size, temperature, top_k, top_p, seed);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::sampling(out->data(), logits->data(), logits->dtype(),
                               batch_size, vocab_size, temperature, top_k, top_p, seed);
#endif
#ifdef ENABLE_MX_API
    case LLAISYS_DEVICE_MX:
        return mx::sampling(out->data(), logits->data(), logits->dtype(),
                               batch_size, vocab_size, temperature, top_k, top_p, seed);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
