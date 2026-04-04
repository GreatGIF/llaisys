#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/paged_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/paged_attention_nvidia.cuh"
#endif

namespace llaisys::ops {

void store_paged_kv_cache(tensor_t k_cache, tensor_t v_cache, tensor_t k, tensor_t v,
                          const std::vector<int32_t> &slot_mapping) {
    CHECK_SAME_DEVICE(k_cache, v_cache, k, v);
    CHECK_SAME_DTYPE(k_cache->dtype(), v_cache->dtype(), k->dtype(), v->dtype());
    ASSERT(k_cache->ndim() == 4 && v_cache->ndim() == 4, "store_paged_kv_cache: cache tensors must be 4D");
    ASSERT(k->ndim() == 3 && v->ndim() == 3, "store_paged_kv_cache: k/v tensors must be 3D");
    ASSERT(k_cache->shape() == v_cache->shape(), "store_paged_kv_cache: k/v cache shape mismatch");
    ASSERT(k->shape() == v->shape(), "store_paged_kv_cache: input k/v shape mismatch");
    ASSERT(k_cache->shape()[2] == k->shape()[1] && k_cache->shape()[3] == k->shape()[2],
           "store_paged_kv_cache: head shape mismatch");
    ASSERT(slot_mapping.size() == k->shape()[0], "store_paged_kv_cache: slot_mapping size mismatch");
    ASSERT(k_cache->isContiguous() && v_cache->isContiguous() && k->isContiguous() && v->isContiguous(),
           "store_paged_kv_cache: all tensors must be contiguous");

    llaisys::core::context().setDevice(k_cache->deviceType(), k_cache->deviceId());
    switch (k_cache->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::store_paged_kv_cache(k_cache->data(), v_cache->data(), k->data(), v->data(),
                                         k_cache->dtype(), k->shape()[0], k->shape()[1], k->shape()[2],
                                         slot_mapping);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::store_paged_kv_cache(k_cache->data(), v_cache->data(), k->data(), v->data(),
                                            k_cache->dtype(), k->shape()[0], k->shape()[1], k->shape()[2],
                                            slot_mapping);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void paged_attention_prefill(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
                             const core::paged_kv::PrefillBatch &batch,
                             size_t num_kv_heads, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k_cache, v_cache);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k_cache->dtype(), v_cache->dtype());
    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3, "paged_attention_prefill: attn_val/q must be 3D");
    ASSERT(k_cache->ndim() == 4 && v_cache->ndim() == 4, "paged_attention_prefill: k/v cache must be 4D");
    ASSERT(q->shape() == attn_val->shape(), "paged_attention_prefill: output shape mismatch");
    ASSERT(q->shape()[1] % num_kv_heads == 0, "paged_attention_prefill: q heads must be divisible by kv heads");
    ASSERT(k_cache->shape()[2] == num_kv_heads && k_cache->shape()[3] == q->shape()[2],
           "paged_attention_prefill: cache head shape mismatch");
    ASSERT(k_cache->shape() == v_cache->shape(), "paged_attention_prefill: k/v cache shape mismatch");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k_cache->isContiguous() && v_cache->isContiguous(),
           "paged_attention_prefill: all tensors must be contiguous");
    ASSERT(batch.input_ids.size() == q->shape()[0] && batch.positions.size() == q->shape()[0],
           "paged_attention_prefill: batch token metadata size mismatch");

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::paged_attention_prefill(attn_val->data(), q->data(), k_cache->data(), v_cache->data(),
                                            attn_val->dtype(), batch, q->shape()[1], num_kv_heads, q->shape()[2],
                                            scale, k_cache->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::paged_attention_prefill(attn_val->data(), q->data(), k_cache->data(), v_cache->data(),
                                               attn_val->dtype(), batch, q->shape()[1], num_kv_heads, q->shape()[2],
                                               scale, k_cache->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void paged_attention_decode(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
                            const core::paged_kv::DecodeBatch &batch,
                            size_t num_kv_heads, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k_cache, v_cache);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k_cache->dtype(), v_cache->dtype());
    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3, "paged_attention_decode: attn_val/q must be 3D");
    ASSERT(k_cache->ndim() == 4 && v_cache->ndim() == 4, "paged_attention_decode: k/v cache must be 4D");
    ASSERT(q->shape() == attn_val->shape(), "paged_attention_decode: output shape mismatch");
    ASSERT(q->shape()[1] % num_kv_heads == 0, "paged_attention_decode: q heads must be divisible by kv heads");
    ASSERT(k_cache->shape()[2] == num_kv_heads && k_cache->shape()[3] == q->shape()[2],
           "paged_attention_decode: cache head shape mismatch");
    ASSERT(k_cache->shape() == v_cache->shape(), "paged_attention_decode: k/v cache shape mismatch");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k_cache->isContiguous() && v_cache->isContiguous(),
           "paged_attention_decode: all tensors must be contiguous");
    ASSERT(batch.input_ids.size() == q->shape()[0] && batch.positions.size() == q->shape()[0],
           "paged_attention_decode: batch token metadata size mismatch");

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::paged_attention_decode(attn_val->data(), q->data(), k_cache->data(), v_cache->data(),
                                           attn_val->dtype(), batch, q->shape()[1], num_kv_heads, q->shape()[2],
                                           scale, k_cache->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::paged_attention_decode(attn_val->data(), q->data(), k_cache->data(), v_cache->data(),
                                              attn_val->dtype(), batch, q->shape()[1], num_kv_heads, q->shape()[2],
                                              scale, k_cache->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
