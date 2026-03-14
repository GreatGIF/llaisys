#include "rms_norm_mx.hpp"

#include "../../../utils.hpp"
#include "../../../utils/cuda_cast.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

template <typename T>
__global__ void rms_norm_inv_rms_kernel(float *inv_rms, const T *in, size_t m,
                                        size_t n, float eps) {
    size_t row = blockIdx.x;
    if (row >= m) {
        return;
    }

    float local_sum = 0.0f;
    for (size_t col = threadIdx.x; col < n; col += blockDim.x) {
        float x = llaisys::utils::cuda::to_float(in[row * n + col]);
        local_sum += x * x;
    }

    __shared__ float sdata[256];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float mean = sdata[0] / static_cast<float>(n);
        inv_rms[row] = rsqrtf(mean + eps);
    }
}

template <typename T>
__global__ void rms_norm_apply_kernel(T *out, const T *in, const T *weight,
                                      const float *inv_rms, size_t m,
                                      size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = m * n;
    if (idx >= total) {
        return;
    }

    size_t row = idx / n;
    size_t col = idx % n;

    float x = llaisys::utils::cuda::to_float(in[idx]);
    float w = llaisys::utils::cuda::to_float(weight[col]);
    float y = x * w * inv_rms[row];
    out[idx] = llaisys::utils::cuda::from_float<T>(y);
}

namespace llaisys::ops::mx {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t m, size_t n, float eps) {
    constexpr int threads_per_block = 256;
    const int row_blocks = static_cast<int>(m);
    const int elem_blocks = static_cast<int>((m * n + threads_per_block - 1) / threads_per_block);

    float *inv_rms = nullptr;
    LLAISYS_CUDA_CHECK(cudaMalloc(&inv_rms, m * sizeof(float)));

    try {
        switch (type) {
        case LLAISYS_DTYPE_F32:
            rms_norm_inv_rms_kernel<<<row_blocks, threads_per_block>>>(
                inv_rms, reinterpret_cast<const float *>(in), m, n, eps);
            LLAISYS_CUDA_CHECK(cudaGetLastError());
            rms_norm_apply_kernel<<<elem_blocks, threads_per_block>>>(
                reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                reinterpret_cast<const float *>(weight), inv_rms, m, n);
            LLAISYS_CUDA_CHECK(cudaGetLastError());
            break;
        case LLAISYS_DTYPE_BF16:
            rms_norm_inv_rms_kernel<<<row_blocks, threads_per_block>>>(
                inv_rms, reinterpret_cast<const nv_bfloat16 *>(in), m, n, eps);
            LLAISYS_CUDA_CHECK(cudaGetLastError());
            rms_norm_apply_kernel<<<elem_blocks, threads_per_block>>>(
                reinterpret_cast<nv_bfloat16 *>(out),
                reinterpret_cast<const nv_bfloat16 *>(in),
                reinterpret_cast<const nv_bfloat16 *>(weight), inv_rms, m, n);
            LLAISYS_CUDA_CHECK(cudaGetLastError());
            break;
        case LLAISYS_DTYPE_F16:
            rms_norm_inv_rms_kernel<<<row_blocks, threads_per_block>>>(
                inv_rms, reinterpret_cast<const half *>(in), m, n, eps);
            LLAISYS_CUDA_CHECK(cudaGetLastError());
            rms_norm_apply_kernel<<<elem_blocks, threads_per_block>>>(
                reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
                reinterpret_cast<const half *>(weight), inv_rms, m, n);
            LLAISYS_CUDA_CHECK(cudaGetLastError());
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    } catch (...) {
        cudaFree(inv_rms);
        throw;
    }

    LLAISYS_CUDA_CHECK(cudaFree(inv_rms));
}
} // namespace llaisys::ops::mx
