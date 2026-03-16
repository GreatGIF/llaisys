#include "self_attention_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../utils/cuda_cast.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <limits>
#include <type_traits>
#include <unordered_map>

// ========================
// 1. CUDA Kernels
// ========================
__global__ void scale_causal_mask_kernel(float *attn_score, size_t qlen,
                                               size_t kvlen, size_t nh,
                                               float scale) {
    const size_t total = nh * qlen * kvlen;
    const size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const size_t rem = idx % (qlen * kvlen);
        const size_t ql = rem / kvlen;
        const size_t kvl = rem % kvlen;

        const long long causal_offset = static_cast<long long>(kvlen) - static_cast<long long>(qlen);
        const bool masked = static_cast<long long>(kvl) > (static_cast<long long>(ql) + causal_offset);
        attn_score[idx] = masked ? -FLT_MAX : (attn_score[idx] * scale);
    }
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

template <typename T>
__global__ void cast_strided_lowp_to_float_kernel(float *dst, const T *src,
                                                   size_t rows, size_t cols,
                                                   size_t src_row_stride) {
    const size_t total = rows * cols;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t r = idx / cols;
    const size_t c = idx % cols;
    dst[idx] = llaisys::utils::cuda::to_float<T>(src[r * src_row_stride + c]);
}

template <typename T>
__global__ void cast_float_to_strided_lowp_kernel(T *dst, const float *src,
                                                   size_t rows, size_t cols,
                                                   size_t dst_row_stride) {
    const size_t total = rows * cols;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t r = idx / cols;
    const size_t c = idx % cols;
    dst[r * dst_row_stride + c] = llaisys::utils::cuda::from_float<T>(src[idx]);
}

template <typename T>
__global__ void cast_float_group_to_strided_lowp_kernel(T *dst, const float *src,
                                                         size_t heads, size_t rows, size_t cols,
                                                         size_t dst_row_stride, size_t dst_head_stride) {
    const size_t total = heads * rows * cols;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t rc = rows * cols;
    const size_t h = idx / rc;
    const size_t rem = idx % rc;
    const size_t r = rem / cols;
    const size_t c = rem % cols;
    dst[h * dst_head_stride + r * dst_row_stride + c] = llaisys::utils::cuda::from_float<T>(src[idx]);
}

struct SelfAttentionWorkspace {
    int device_id{-1};
    float *attn_score{nullptr};
    size_t attn_score_capacity{0};
    float *v_head_buf_f32{nullptr};
    size_t v_head_buf_capacity{0};
    float *out_head_buf_f32{nullptr};
    size_t out_head_buf_capacity{0};
};

inline SelfAttentionWorkspace &workspace_for_device(int device_id) {
    thread_local std::unordered_map<int, SelfAttentionWorkspace> workspaces;
    auto &ws = workspaces[device_id];
    ws.device_id = device_id;
    return ws;
}

inline void ensure_workspace_buffer(float *&ptr, size_t &capacity, size_t required_count) {
    if (capacity >= required_count && ptr != nullptr) {
        return;
    }
    if (ptr != nullptr) {
        LLAISYS_CUDA_CHECK(cudaFree(ptr));
    }
    LLAISYS_CUDA_CHECK(cudaMalloc(&ptr, required_count * sizeof(float)));
    capacity = required_count;
}

inline cublasHandle_t cublas_handle_for_device(int device_id) {
    thread_local std::unordered_map<int, cublasHandle_t> handles;
    auto it = handles.find(device_id);
    if (it != handles.end()) {
        return it->second;
    }
    cublasHandle_t handle = nullptr;
    LLAISYS_CUBLAS_CHECK(cublasCreate(&handle));
    handles.emplace(device_id, handle);
    return handle;
}

// ========================
// 3. Core Implementation
// ========================
template <typename T>
void compute_query_key(cublasHandle_t handle,
                   float *attn_score,
                   const T *q_ptr, const T *k_ptr,
                   size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, size_t ng,
                   cudaStream_t stream) {
    (void)stream;
    const int m = static_cast<int>(kvlen);
    const int n = static_cast<int>(qlen);
    const int k = static_cast<int>(hd);
    const int lda_q = static_cast<int>(nh * hd);
    const int lda_k = static_cast<int>(nkvh * hd);

    if constexpr (std::is_same_v<T, float>) {
        const float alpha = 1.0f, beta = 0.0f;
        const long long stride_b = static_cast<long long>(hd);
        const long long stride_c = static_cast<long long>(qlen) * static_cast<long long>(kvlen);
        for (size_t kv_head = 0; kv_head < nkvh; ++kv_head) {
            const size_t base_head = kv_head * ng;
            const float *k_head_src = reinterpret_cast<const float *>(k_ptr + kv_head * hd);
            const float *q_head_src = reinterpret_cast<const float *>(q_ptr + base_head * hd);
            float *score_head = attn_score + base_head * qlen * kvlen;
            LLAISYS_CUBLAS_CHECK(cublasSgemmStridedBatched(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                k_head_src, lda_k, 0,
                q_head_src, lda_q, stride_b,
                &beta,
                score_head, m, stride_c,
                static_cast<int>(ng)));
        }
        return;
    }

    if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16>) {
        const float alpha = 1.0f, beta = 0.0f;
        const long long stride_b = static_cast<long long>(hd);
        const long long stride_c = static_cast<long long>(qlen) * static_cast<long long>(kvlen);
        const cudaDataType_t ab_type = std::is_same_v<T, half> ? CUDA_R_16F : CUDA_R_16BF;
        for (size_t kv_head = 0; kv_head < nkvh; ++kv_head) {
            const size_t base_head = kv_head * ng;
            const T *k_head_src = k_ptr + kv_head * hd;
            const T *q_head_src = q_ptr + base_head * hd;
            float *score_head = attn_score + base_head * qlen * kvlen;
            LLAISYS_CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                k_head_src, ab_type, lda_k, 0,
                q_head_src, ab_type, lda_q, stride_b,
                &beta,
                score_head, CUDA_R_32F, m, stride_c,
                static_cast<int>(ng),
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        return;
    }

    for (size_t head = 0; head < nh; ++head) {
        const size_t kv_head = head / ng;
        const T *q_head_src = q_ptr + head * hd;
        const T *k_head_src = k_ptr + kv_head * hd;
        float *score_head = attn_score + head * qlen * kvlen;

        const float alpha = 1.0f, beta = 0.0f;
        if constexpr (std::is_same_v<T, float>) {
            LLAISYS_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                            &alpha, k_head_src, lda_k, q_head_src, lda_q, &beta, score_head, m));
        } else if constexpr (std::is_same_v<T, half>) {
            LLAISYS_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                            &alpha, k_head_src, CUDA_R_16F, lda_k,
                                            q_head_src, CUDA_R_16F, lda_q,
                                            &beta, score_head, CUDA_R_32F, m,
                                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        } else if constexpr (std::is_same_v<T, nv_bfloat16>) {
            LLAISYS_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
                                            &alpha, k_head_src, CUDA_R_16BF, lda_k,
                                            q_head_src, CUDA_R_16BF, lda_q,
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
                   float *v_head_buf_f32, float *out_head_buf_f32,
                   size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd, size_t ng,
                   cudaStream_t stream) {
    constexpr int THREADS = 256;
    const int m = static_cast<int>(hd);
    const int n = static_cast<int>(qlen);
    const int k = static_cast<int>(kvlen);
    const int lda_v = static_cast<int>(nkvh * hd);
    const int ldc_out = static_cast<int>(nh * hd);

    if constexpr (std::is_same_v<T, float>) {
        const float alpha = 1.0f, beta = 0.0f;
        const long long stride_b = static_cast<long long>(qlen) * static_cast<long long>(kvlen);
        const long long stride_c = static_cast<long long>(hd);
        for (size_t kv_head = 0; kv_head < nkvh; ++kv_head) {
            const size_t base_head = kv_head * ng;
            const float *v_head_src = reinterpret_cast<const float *>(v_ptr + kv_head * hd);
            const float *score_head = attn_score + base_head * qlen * kvlen;
            float *out_head_dst = reinterpret_cast<float *>(out_ptr + base_head * hd);
            LLAISYS_CUBLAS_CHECK(cublasSgemmStridedBatched(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                v_head_src, lda_v, 0,
                score_head, k, stride_b,
                &beta,
                out_head_dst, ldc_out, stride_c,
                static_cast<int>(ng)));
        }
        return;
    }

    const float alpha = 1.0f, beta = 0.0f;
    const long long stride_b = static_cast<long long>(qlen) * static_cast<long long>(kvlen);
    const long long stride_c = static_cast<long long>(qlen) * static_cast<long long>(hd);
    for (size_t kv_head = 0; kv_head < nkvh; ++kv_head) {
        const size_t base_head = kv_head * ng;
        const T *v_head_src = v_ptr + kv_head * hd;
        const float *score_head = attn_score + base_head * qlen * kvlen;
        T *out_head_dst = out_ptr + base_head * hd;

        // Cast shared V(head) only once, then reuse across ng query heads.
        const size_t v_count = kvlen * hd;
        const int v_blocks = static_cast<int>((v_count + THREADS - 1) / THREADS);
        cast_strided_lowp_to_float_kernel<<<v_blocks, THREADS, 0, stream>>>(
            v_head_buf_f32, v_head_src, kvlen, hd, nkvh * hd);
        LLAISYS_CUDA_CHECK(cudaGetLastError());

        LLAISYS_CUBLAS_CHECK(cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, n, k,
            &alpha,
            v_head_buf_f32, m, 0,
            score_head, k, stride_b,
            &beta,
            out_head_buf_f32, m, stride_c,
            static_cast<int>(ng)));

        const size_t out_count = ng * qlen * hd;
        const int out_blocks = static_cast<int>((out_count + THREADS - 1) / THREADS);
        cast_float_group_to_strided_lowp_kernel<<<out_blocks, THREADS, 0, stream>>>(
            out_head_dst, out_head_buf_f32, ng, qlen, hd, nh * hd, hd);
        LLAISYS_CUDA_CHECK(cudaGetLastError());
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
                   hd <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                   nh * hd <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                   nkvh * hd <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
                   nh * qlen <= static_cast<size_t>(std::numeric_limits<int>::max()),
                   "self_attention(nvidia): dims exceed int range required by cuBLAS");
    CHECK_ARGUMENT(nkvh > 0 && nh > 0 && (nh % nkvh == 0),
                   "self_attention(nvidia): require nh % nkvh == 0 and nh/nkvh > 0");

    const size_t ng = nh / nkvh;
    auto &runtime = llaisys::core::context().runtime();
    auto &workspace = workspace_for_device(runtime.deviceId());
    const size_t score_count = nh * qlen * kvlen;
    ensure_workspace_buffer(workspace.attn_score, workspace.attn_score_capacity, score_count);

    float *v_head_buf_f32 = nullptr;
    float *out_head_buf_f32 = nullptr;
    if constexpr (!std::is_same_v<T, float>) {
        ensure_workspace_buffer(workspace.v_head_buf_f32, workspace.v_head_buf_capacity, kvlen * hd);
        ensure_workspace_buffer(workspace.out_head_buf_f32, workspace.out_head_buf_capacity, ng * qlen * hd);
        v_head_buf_f32 = workspace.v_head_buf_f32;
        out_head_buf_f32 = workspace.out_head_buf_f32;
    }

    cublasHandle_t handle = cublas_handle_for_device(runtime.deviceId());
    LLAISYS_CUBLAS_CHECK(cublasSetStream(handle, stream));

    try {
        const T *q_ptr = reinterpret_cast<const T *>(q);
        const T *k_ptr = reinterpret_cast<const T *>(k);
        const T *v_ptr = reinterpret_cast<const T *>(v);
        T *out_ptr = reinterpret_cast<T *>(attn_val);

        // QK^T
        compute_query_key(handle, workspace.attn_score, q_ptr, k_ptr,
                      qlen, kvlen, nh, nkvh, hd, ng, stream);

        // Scale + Causal Mask 
        const size_t total = nh * qlen * kvlen;
        const int blocks = static_cast<int>((total + THREADS - 1) / THREADS);
        scale_causal_mask_kernel<<<blocks, THREADS, 0, stream>>>(
            workspace.attn_score, qlen, kvlen, nh, scale);
        LLAISYS_CUDA_CHECK(cudaGetLastError());

        // Softmax
        const size_t rows = nh * qlen;
        softmax_rows_kernel<<<static_cast<int>(rows), THREADS, 0, stream>>>(
            workspace.attn_score, rows, kvlen);
        LLAISYS_CUDA_CHECK(cudaGetLastError());

        // AV
        compute_atteion_value(handle, out_ptr, workspace.attn_score, v_ptr,
                      v_head_buf_f32, out_head_buf_f32,
                      qlen, kvlen, nh, nkvh, hd, ng, stream);

    } catch (...) {
        throw;
    }
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