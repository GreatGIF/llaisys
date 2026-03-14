#include "swiglu_mx.hpp"

#include "../../../utils.hpp"
#include "../../../utils/cuda_cast.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace {

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    float gate_val = llaisys::utils::cuda::to_float(gate[idx]);
    float up_val = llaisys::utils::cuda::to_float(up[idx]);
    float y = up_val * gate_val / (1.0f + expf(-gate_val));
    out[idx] = llaisys::utils::cuda::from_float<T>(y);
}

} // namespace

namespace llaisys::ops::mx {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t seq_len, size_t hid_dim) {
    constexpr int threads_per_block = 256;
    const size_t total = seq_len * hid_dim;
    const int blocks_per_grid = static_cast<int>((total + threads_per_block - 1) / threads_per_block);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up),
            total);
        LLAISYS_CUDA_CHECK(cudaGetLastError());
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<nv_bfloat16 *>(out),
            reinterpret_cast<const nv_bfloat16 *>(gate),
            reinterpret_cast<const nv_bfloat16 *>(up),
            total);
        LLAISYS_CUDA_CHECK(cudaGetLastError());
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<half *>(out),
            reinterpret_cast<const half *>(gate),
            reinterpret_cast<const half *>(up),
            total);
        LLAISYS_CUDA_CHECK(cudaGetLastError());
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::mx
