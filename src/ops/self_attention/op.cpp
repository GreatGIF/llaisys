#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());

    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "self_attention: all tensors must be 3D");
    ASSERT(q->shape()[2] == k->shape()[2] && q->shape()[2] == v->shape()[2],
           "self_attention: head_dim mismatch");
    ASSERT(k->shape()[1] == v->shape()[1], "self_attention: k/v head number mismatch");
    ASSERT(k->shape()[0] == v->shape()[0], "self_attention: k/v seq len mismatch");
    ASSERT(q->shape()[1] % k->shape()[1] == 0,
           "self_attention: q heads must be divisible by kv heads");

    const size_t qlen = q->shape()[0];
    const size_t kvlen = k->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t nkvh = k->shape()[1];
    const size_t hd = q->shape()[2];

    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == nh && attn_val->shape()[2] == hd,
           "self_attention: output shape mismatch");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                   attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
#ifdef ENABLE_NVIDIA_API
       case LLAISYS_DEVICE_NVIDIA:
              return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                                                 attn_val->dtype(), qlen, kvlen, nh, nkvh, hd, scale);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
