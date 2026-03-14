#pragma once

#ifdef ENABLE_NVIDIA_API
#define _LLAISYS_CUDA_CAST_
#endif

#ifdef ENABLE_MX_API
#define _LLAISYS_CUDA_CAST_
#endif

#ifdef _LLAISYS_CUDA_CAST_
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace llaisys::utils::cuda {

template <typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<half>(half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ half from_float<half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

} // namespace llaisys::utils::cuda

#endif // _LLAISYS_CUDA_CAST_
