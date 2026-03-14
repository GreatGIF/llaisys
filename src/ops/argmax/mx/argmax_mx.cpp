#include "argmax_mx.hpp"

#include "../../../utils.hpp"
#include "../../../utils/cuda_cast.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>

template <typename T>
__global__ void argmax_kernel(std::int64_t *max_idx, T *max_val, const T *vals, size_t size) {
    constexpr int kBlockSize = 256;
    __shared__ float s_scores[kBlockSize];
    __shared__ std::int64_t s_indices[kBlockSize];
    __shared__ T s_values[kBlockSize];

    const int tid = threadIdx.x;

    float local_best_score = -FLT_MAX;
    std::int64_t local_best_idx = -1;
    T local_best_val = vals[0];

    for (size_t i = static_cast<size_t>(tid); i < size; i += blockDim.x) {
        const T v_raw = vals[i];
        const float v = llaisys::utils::cuda::to_float(v_raw);
        if (v > local_best_score || (v == local_best_score && static_cast<std::int64_t>(i) < local_best_idx)) {
            local_best_score = v;
            local_best_idx = static_cast<std::int64_t>(i);
            local_best_val = v_raw;
        }
    }

    s_scores[tid] = local_best_score;
    s_indices[tid] = local_best_idx;
    s_values[tid] = local_best_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_score = s_scores[tid + stride];
            const std::int64_t other_idx = s_indices[tid + stride];

            if (other_score > s_scores[tid] || (other_score == s_scores[tid] && other_idx < s_indices[tid])) {
                s_scores[tid] = other_score;
                s_indices[tid] = other_idx;
                s_values[tid] = s_values[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = s_indices[0];
        *max_val = s_values[0];
    }
}

namespace llaisys::ops::mx {
void argmax(std::int64_t *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size) {
    constexpr int threads_per_block = 256;
    constexpr int blocks_per_grid = 1;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<<<blocks_per_grid, threads_per_block>>>(max_idx,
                                                              reinterpret_cast<float *>(max_val),
                                                              reinterpret_cast<const float *>(vals), size);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<<<blocks_per_grid, threads_per_block>>>(max_idx,
                                                              reinterpret_cast<nv_bfloat16 *>(max_val),
                                                              reinterpret_cast<const nv_bfloat16 *>(vals), size);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<<<blocks_per_grid, threads_per_block>>>(max_idx,
                                                              reinterpret_cast<half *>(max_val),
                                                              reinterpret_cast<const half *>(vals), size);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::mx


