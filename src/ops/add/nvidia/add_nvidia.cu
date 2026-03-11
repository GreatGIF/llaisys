#include "add_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = a[idx] + b[idx];
    }
}

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    int threads_per_block = 256;
    int blocks_per_grid = (numel + threads_per_block - 1) / threads_per_block;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<<<blocks_per_grid, threads_per_block>>>(reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
                                                           reinterpret_cast<const float *>(b), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<<<blocks_per_grid, threads_per_block>>>(reinterpret_cast<nv_bfloat16 *>(c),
                                                           reinterpret_cast<const nv_bfloat16 *>(a),
                                                           reinterpret_cast<const nv_bfloat16 *>(b), numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<<<blocks_per_grid, threads_per_block>>>(reinterpret_cast<half *>(c), reinterpret_cast<const half *>(a),
                                                           reinterpret_cast<const half *>(b), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
