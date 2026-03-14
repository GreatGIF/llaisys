#include "self_attention_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../utils/cuda_cast.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>

// ========================
// 1. CUDA Kernels
// ========================
__global__ void scale_causal_mask_kernel(float *attn_score, size_t qlen,
                                               size_t kvlen, size_t nh,
                                               float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nh * qlen * kvlen;
    if (idx >= total) return;

    size_t rem = idx % (qlen * kvlen);
    size_t ql = rem / kvlen;
    size_t kvl = rem % kvlen;

    long long causal_offset = static_cast<long long>(kvlen) - static_cast<long long>(qlen);
    bool masked = static_cast<long long>(kvl) > (static_cast<long long>(ql) + causal_offset);
    attn_score[idx] = masked ? -FLT_MAX : (attn_score[idx] * scale);
}

__global__ void softmax_rows_kernel(float *attn_score, size_t rows, size_t kvlen) {
    size_t row = blockIdx.x;
    if (row >= rows) return;

    float *row_ptr = attn_score + row * kvlen;
    constexpr int BLOCK_SIZE = 256;
    __shared__ float sdata[BLOCK_SIZE];

    // Find max
    float local_max = -FLT_MAX;
    for (size_t i = threadIdx.x; i < kvlen; i += blockDim.x)
        local_max = fmaxf(local_max, row_ptr[i]);

    sdata[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        __syncthreads();
    }
    float max_val = sdata[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (size_t i = threadIdx.x; i < kvlen; i += blockDim.x) {
        float e = expf(row_ptr[i] - max_val);
        row_ptr[i] = e;
        local_sum += e;
    }

    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    float sum_val = sdata[0];

    // Normalize
    for (size_t i = threadIdx.x; i < kvlen; i += blockDim.x)
        row_ptr[i] /= sum_val;
}

template <typename T>
__global__ void cast_float_to_lowp_kernel(T *dst, const float *src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = llaisys::utils::cuda::from_float<T>(src[idx]);
}

template <typename T>
__global__ void cast_lowp_to_float_kernel(float *dst, const T *src, size_t count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = llaisys::utils::cuda::to_float<T>(src[idx]);
}

// ========================
// 2. Helper Utilities
// ========================
template <typename T>
inline void copy_head_matrix_async(T *dst, const T *src,
                                   size_t dst_pitch_elems, size_t src_pitch_elems,
                                   size_t width_elems, size_t height_rows,
                                   cudaStream_t stream) {
    LLAISYS_CUDA_CHECK(cudaMemcpy2DAsync(
        dst, dst_pitch_elems * sizeof(T),
        src, src_pitch_elems * sizeof(T),
        width_elems * sizeof(T), height_rows,
        cudaMemcpyDeviceToDevice, stream));
}

// ========================
// 3. Core Implementation
// ========================
template <typename T>
void compute_query_key(cublasHandle_t handle,
                   float *attn_score,
                   const T *q_ptr, const T *k_ptr,
                   T *q_head_buf, T *k_head_buf,
                   size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, size_t ng,
                   cudaStream_t stream) {
    for (size_t head = 0; head < nh; ++head) {
        size_t kv_head = head / ng;
        const T *q_head_src = q_ptr + head * hd;
        const T *k_head_src = k_ptr + kv_head * hd;
        float *score_head = attn_score + head * qlen * kvlen;

        copy_head_matrix_async(q_head_buf, q_head_src, hd, nh * hd, hd, qlen, stream);
        copy_head_matrix_async(k_head_buf, k_head_src, hd, nkvh * hd, hd, kvlen, stream);

        const int m = static_cast<int>(kvlen);
        const int n = static_cast<int>(qlen);
        const int k = static_cast<int>(hd);

        const float alpha = 1.0f, beta = 0.0f;
        if constexpr (std::is_same_v<T, float>) {
            LLAISYS_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                            &alpha, k_head_buf, k, q_head_buf, k, &beta, score_head, m));
        } else if constexpr (std::is_same_v<T, half>) {
            LLAISYS_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                            &alpha, k_head_buf, CUDA_R_16F, k,
                                            q_head_buf, CUDA_R_16F, k,
                                            &beta, score_head, CUDA_R_32F, m,
                                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
            LLAISYS_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                            &alpha, k_head_buf, CUDA_R_16BF, k,
                                            q_head_buf, CUDA_R_16BF, k,
                                            &beta, score_head, CUDA_R_32F, m,
                                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }
}

template <typename T>
void compute_atteion_value(cublasHandle_t handle,
                   T *out_ptr,
                   const float *attn_score,
                   const T *v_ptr,
                   T *v_head_buf, float *v_head_buf_f32, float *score_head_buf, float *out_head_buf_f32,
                   T *out_head_buf,
                   size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, size_t ng,
                   cudaStream_t stream) {
    for (size_t head = 0; head < nh; ++head) {
        size_t kv_head = head / ng;
        const T *v_head_src = v_ptr + kv_head * hd;
        const float *score_head = attn_score + head * qlen * kvlen;
        T *out_head_dst = out_ptr + head * hd;

        copy_head_matrix_async(v_head_buf, v_head_src, hd, nkvh * hd, hd, kvlen, stream);

        // AV GEMM with optional cast
        const int m = static_cast<int>(hd);
        const int n = static_cast<int>(qlen);
        const int k = static_cast<int>(kvlen);
        const float alpha = 1.0f, beta = 0.0f;

        if constexpr (std::is_same_v<T, float>) {
            LLAISYS_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                             &alpha, v_head_buf, m, score_head, k, &beta, out_head_buf_f32, m));
        } else {
            constexpr int THREADS = 256;

            // Cast V from lowp to fp32 for broad cuBLAS compatibility
            size_t v_count = kvlen * hd;
            int v_blocks = static_cast<int>((v_count + THREADS - 1) / THREADS);
            cast_lowp_to_float_kernel<<<v_blocks, THREADS, 0, stream>>>(v_head_buf_f32, v_head_buf, v_count);
            LLAISYS_CUDA_CHECK(cudaGetLastError());

            // Keep score in fp32 to avoid precision loss on BF16/FP16 models
            LLAISYS_CUDA_CHECK(cudaMemcpyAsync(
                score_head_buf, score_head, qlen * kvlen * sizeof(float),
                cudaMemcpyDeviceToDevice, stream));

            // AV GEMM in fp32
            LLAISYS_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                                             &alpha, v_head_buf_f32, m, score_head_buf, k,
                                             &beta, out_head_buf_f32, m));

            size_t out_count = qlen * hd;
            int out_blocks = static_cast<int>((out_count + THREADS - 1) / THREADS);
            cast_float_to_lowp_kernel<<<out_blocks, THREADS, 0, stream>>>(out_head_buf, out_head_buf_f32, out_count);
            LLAISYS_CUDA_CHECK(cudaGetLastError());
        }

        if constexpr (std::is_same_v<T, float>) {
            copy_head_matrix_async(out_head_dst, reinterpret_cast<T *>(out_head_buf_f32), nh * hd, hd, hd, qlen, stream);
        } else {
            copy_head_matrix_async(out_head_dst, out_head_buf, nh * hd, hd, hd, qlen, stream);
        }
    }
}

template <typename T>
void self_attention_impl(std::byte *attn_val, const std::byte *q,
                         const std::byte *k, const std::byte *v,
                         size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd,
                         float scale, cudaStream_t stream) {
    constexpr int THREADS = 256;

    CHECK_ARGUMENT(nh <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                   qlen <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                   kvlen <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                   hd <= static_cast<size_t>(std::numeric_limits<int>::max()),
                   "self_attention(nvidia): dims exceed int range required by cuBLAS");

    size_t score_count = nh * qlen * kvlen;
    float *attn_score = nullptr;
    T *q_head_buf = nullptr, *k_head_buf = nullptr, *v_head_buf = nullptr;
    float *v_head_buf_f32 = nullptr;
    float *score_head_buf = nullptr;
    float *out_head_buf_f32 = nullptr;
    T *out_head_buf = nullptr;

    LLAISYS_CUDA_CHECK(cudaMalloc(&attn_score, score_count * sizeof(float)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&q_head_buf, qlen * hd * sizeof(T)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&k_head_buf, kvlen * hd * sizeof(T)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&v_head_buf, kvlen * hd * sizeof(T)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&v_head_buf_f32, kvlen * hd * sizeof(float)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&score_head_buf, qlen * kvlen * sizeof(float)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&out_head_buf_f32, qlen * hd * sizeof(float)));
    LLAISYS_CUDA_CHECK(cudaMalloc(&out_head_buf, qlen * hd * sizeof(T)));

    cublasHandle_t handle = nullptr;
    LLAISYS_CUBLAS_CHECK(cublasCreate(&handle));
    LLAISYS_CUBLAS_CHECK(cublasSetStream(handle, stream));

    try {
        const T *q_ptr = reinterpret_cast<const T *>(q);
        const T *k_ptr = reinterpret_cast<const T *>(k);
        const T *v_ptr = reinterpret_cast<const T *>(v);
        T *out_ptr = reinterpret_cast<T *>(attn_val);
        size_t ng = nh / nkvh;

        // QK^T
        compute_query_key(handle, attn_score, q_ptr, k_ptr,
                      q_head_buf, k_head_buf,
                      qlen, kvlen, nh, nkvh, hd, ng, stream);

        // Scale + Causal Mask 
        size_t total = nh * qlen * kvlen;
        int blocks = (total + THREADS - 1) / THREADS;
        scale_causal_mask_kernel<<<blocks, THREADS, 0, stream>>>(
            attn_score, qlen, kvlen, nh, scale);
        LLAISYS_CUDA_CHECK(cudaGetLastError());

        // Softmax
        size_t rows = nh * qlen;
        softmax_rows_kernel<<<static_cast<int>(rows), THREADS, 0, stream>>>(
            attn_score, rows, kvlen);
        LLAISYS_CUDA_CHECK(cudaGetLastError());

        // AV
        compute_atteion_value(handle, out_ptr, attn_score, v_ptr,
                      v_head_buf, v_head_buf_f32, score_head_buf, out_head_buf_f32, out_head_buf,
                      qlen, kvlen, nh, nkvh, hd, ng, stream);

    } catch (...) {
        cublasDestroy(handle);
        cudaFree(out_head_buf); cudaFree(out_head_buf_f32); cudaFree(score_head_buf); cudaFree(v_head_buf_f32); cudaFree(v_head_buf);
        cudaFree(k_head_buf); cudaFree(q_head_buf); cudaFree(attn_score);
        throw;
    }

    LLAISYS_CUBLAS_CHECK(cublasDestroy(handle));
    LLAISYS_CUDA_CHECK(cudaFree(out_head_buf));
    LLAISYS_CUDA_CHECK(cudaFree(out_head_buf_f32));
    LLAISYS_CUDA_CHECK(cudaFree(score_head_buf));
    LLAISYS_CUDA_CHECK(cudaFree(v_head_buf_f32));
    LLAISYS_CUDA_CHECK(cudaFree(v_head_buf));
    LLAISYS_CUDA_CHECK(cudaFree(k_head_buf));
    LLAISYS_CUDA_CHECK(cudaFree(q_head_buf));
    LLAISYS_CUDA_CHECK(cudaFree(attn_score));
}

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd,
                    float scale) {
    auto stream = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    switch (type) {
        case LLAISYS_DTYPE_F32:
            self_attention_impl<float>(attn_val, q, k, v, qlen, kvlen, nh, nkvh, hd, scale, stream);
            break;
        case LLAISYS_DTYPE_BF16:
            self_attention_impl<nv_bfloat16>(attn_val, q, k, v, qlen, kvlen, nh, nkvh, hd, scale, stream);
            break;
        case LLAISYS_DTYPE_F16:
            self_attention_impl<half>(attn_val, q, k, v, qlen, kvlen, nh, nkvh, hd, scale, stream);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia