#include "rope_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/nvidia_cast.cuh"
#include "../../../utils/nvidia_check.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>

template <typename T>
__global__ void rope_kernel(T *out, const T *in, const std::int64_t *pos_ids,
                            size_t seq_len, size_t head_num, size_t head_dim,
                            float theta) {
    const size_t half_head_dim = head_dim >> 1;
    const size_t total_pairs = seq_len * head_num * half_head_dim;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) {
        return;
    }

    const size_t pair = idx % half_head_dim;
    const size_t tmp = idx / half_head_dim;
    const size_t head = tmp % head_num;
    const size_t seq = tmp / head_num;

    const size_t base = (seq * head_num + head) * head_dim;
    const size_t idx1 = base + pair;
    const size_t idx2 = idx1 + half_head_dim;

    const float pos = static_cast<float>(pos_ids[seq]);
    const float inv_freq = powf(theta, (2.0f * static_cast<float>(pair)) / static_cast<float>(head_dim));
    const float freq = pos / inv_freq;
    const float sin = sinf(freq);
    const float cos = cosf(freq);

    const float x1 = llaisys::utils::nvidia::to_float(in[idx1]);
    const float x2 = llaisys::utils::nvidia::to_float(in[idx2]);

    out[idx1] = llaisys::utils::nvidia::from_float<T>(x1 * cos - x2 * sin);
    out[idx2] = llaisys::utils::nvidia::from_float<T>(x1 * sin + x2 * cos);
}

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::int64_t *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t head_num,
          size_t head_dim, float theta) {
    constexpr int threads_per_block = 256;
    const size_t half_head_dim = head_dim >> 1;
    const size_t total_pairs = seq_len * head_num * half_head_dim;
    const int blocks_per_grid = static_cast<int>((total_pairs + threads_per_block - 1) / threads_per_block);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
            pos_ids, seq_len, head_num, head_dim, theta);
        LLAISYS_CUDA_CHECK(cudaGetLastError());
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<nv_bfloat16 *>(out), reinterpret_cast<const nv_bfloat16 *>(in),
            pos_ids, seq_len, head_num, head_dim, theta);
        LLAISYS_CUDA_CHECK(cudaGetLastError());
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<<<blocks_per_grid, threads_per_block>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
            pos_ids, seq_len, head_num, head_dim, theta);
        LLAISYS_CUDA_CHECK(cudaGetLastError());
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
