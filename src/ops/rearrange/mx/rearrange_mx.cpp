#include "rearrange_mx.hpp"

#include "../../../utils.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T>
__global__ void rearrange_kernel(T *out, const T *in, const size_t *shape_d,
                                  const ptrdiff_t *out_strides_d,
                                  const ptrdiff_t *in_strides_d, size_t ndim,
                                  size_t total_elems) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) {
        // Convert linear index to multi-dimensional index
        size_t out_offset = 0, in_offset = 0;
        size_t tmp_idx = idx;

        for (int i = ndim - 1; i >= 0; i--) {
            size_t coord = tmp_idx % shape_d[i];
            tmp_idx /= shape_d[i];
            out_offset += coord * out_strides_d[i];
            in_offset += coord * in_strides_d[i];
        }

        out[out_offset] = in[in_offset];
    }
}

namespace llaisys::ops::mx {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides, size_t ndim) {
    // Calculate total elements
    size_t total_elems = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elems *= shape[i];
    }

    // Copy shape and strides to device memory
    size_t *shape_d;
    ptrdiff_t *out_strides_d, *in_strides_d;
    
    cudaMalloc(&shape_d, ndim * sizeof(size_t));
    cudaMalloc(&out_strides_d, ndim * sizeof(ptrdiff_t));
    cudaMalloc(&in_strides_d, ndim * sizeof(ptrdiff_t));

    cudaMemcpy(shape_d, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(out_strides_d, out_strides.data(), ndim * sizeof(ptrdiff_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(in_strides_d, in_strides.data(), ndim * sizeof(ptrdiff_t),
               cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (total_elems + threads_per_block - 1) / threads_per_block;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rearrange_kernel<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
            shape_d, out_strides_d, in_strides_d, ndim, total_elems);
        break;
    case LLAISYS_DTYPE_BF16:
        rearrange_kernel<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<nv_bfloat16 *>(out),
            reinterpret_cast<const nv_bfloat16 *>(in), shape_d, out_strides_d,
            in_strides_d, ndim, total_elems);
        break;
    case LLAISYS_DTYPE_F16:
        rearrange_kernel<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
            shape_d, out_strides_d, in_strides_d, ndim, total_elems);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    // Free device memory
    cudaFree(shape_d);
    cudaFree(out_strides_d);
    cudaFree(in_strides_d);
}
} // namespace llaisys::ops::mx
