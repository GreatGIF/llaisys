#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename T>
__global__ void embedding_kernel(T *out, const std::int64_t *index, const T *weight, size_t index_size, size_t row_size) {
    size_t total_elements = index_size * row_size;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < total_elements; i += stride) {
        size_t row_idx = i / row_size;
        size_t col_idx = i % row_size;
        std::int64_t token_id = index[row_idx];
        size_t src_offset = token_id * row_size + col_idx;
        out[i] = weight[src_offset];
    }
}

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::int64_t *index, const std::byte *weight, llaisysDataType_t type, size_t index_size, size_t row_size) {
    int threads_per_block = 256;
    int blocks_per_grid = (index_size * row_size + threads_per_block - 1) / threads_per_block;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<<<blocks_per_grid, threads_per_block>>>(reinterpret_cast<float *>(out), index, 
                                                                 reinterpret_cast<const float *>(weight), index_size, row_size);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<<<blocks_per_grid, threads_per_block>>>(reinterpret_cast<nv_bfloat16 *>(out), index, 
                                                                 reinterpret_cast<const nv_bfloat16 *>(weight), index_size, row_size);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<<<blocks_per_grid, threads_per_block>>>(reinterpret_cast<half *>(out), index, 
                                                                 reinterpret_cast<const half *>(weight), index_size, row_size);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}