#include "linear_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../utils/cuda_cast.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <limits>
#include <unordered_map>

namespace {

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
	const size_t total = M * N;
	const size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
	for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total; idx += stride) {
		const size_t n = idx % N;
		const float v = llaisys::utils::cuda::to_float(out[idx]) + llaisys::utils::cuda::to_float(bias[n]);
		out[idx] = llaisys::utils::cuda::from_float<T>(v);
	}
}

inline cublasHandle_t get_cublas_handle_for_current_thread_device(int device_id) {
	// Keep handles alive for thread lifetime to avoid per-op create/destroy overhead.
	// Key by device id to safely support device switches in one thread.
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

template <typename T>
inline void maybe_copy_bias_for_single_row(T *out, const T *bias, size_t M, size_t N, cudaStream_t stream) {
	if (bias != nullptr && M == 1) {
		LLAISYS_CUDA_CHECK(cudaMemcpyAsync(out, bias, N * sizeof(T), cudaMemcpyDeviceToDevice, stream));
	}
}

} // namespace

// cuBLAS中gemm采用列主序, 行主序的 Y = X * W^T 直接运算需要 Y^T = x^T * W
// 行主序 Y = X * W^T => Y^T = W * X^T
namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t M, size_t N, size_t K) {
	CHECK_ARGUMENT(M <= static_cast<size_t>(std::numeric_limits<int>::max())
					   && N <= static_cast<size_t>(std::numeric_limits<int>::max())
					   && K <= static_cast<size_t>(std::numeric_limits<int>::max()),
				   "linear(nvidia): M/N/K exceed int range required by cuBLAS");

	const int m = static_cast<int>(N);
	const int n = static_cast<int>(M);
	const int k = static_cast<int>(K);

	auto &runtime = llaisys::core::context().runtime();
	auto stream = static_cast<cudaStream_t>(runtime.stream());
	cublasHandle_t handle = get_cublas_handle_for_current_thread_device(runtime.deviceId());
	LLAISYS_CUBLAS_CHECK(cublasSetStream(handle, stream));

	constexpr int threads_per_block = 256;
	const int blocks_per_grid = (bias == nullptr)
		? 0
		: static_cast<int>((M * N + threads_per_block - 1) / threads_per_block);

	switch (type) {
	case LLAISYS_DTYPE_F32: {
		const float alpha = 1.0f;
		const float beta = (bias != nullptr && M == 1) ? 1.0f : 0.0f;
		maybe_copy_bias_for_single_row(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(bias), M, N, stream);
		LLAISYS_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
									 &alpha,
								     reinterpret_cast<const float *>(weight), k,
									 reinterpret_cast<const float *>(in), k,
									 &beta,
									 reinterpret_cast<float *>(out), m));
		if (bias != nullptr && M != 1) {
			add_bias_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
				reinterpret_cast<float *>(out), reinterpret_cast<const float *>(bias), M, N);
			LLAISYS_CUDA_CHECK(cudaGetLastError());
		}
		break;
	}
	case LLAISYS_DTYPE_F16: {
		const float alpha = 1.0f;
		const float beta = (bias != nullptr && M == 1) ? 1.0f : 0.0f;
		maybe_copy_bias_for_single_row(reinterpret_cast<half *>(out), reinterpret_cast<const half *>(bias), M, N, stream);
		LLAISYS_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
									  &alpha,
							          reinterpret_cast<const half *>(weight), CUDA_R_16F, k,
									  reinterpret_cast<const half *>(in), CUDA_R_16F, k,
									  &beta,
									  reinterpret_cast<half *>(out), CUDA_R_16F, m,
									  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		if (bias != nullptr && M != 1) {
			add_bias_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
				reinterpret_cast<half *>(out), reinterpret_cast<const half *>(bias), M, N);
			LLAISYS_CUDA_CHECK(cudaGetLastError());
		}
		break;
	}
	case LLAISYS_DTYPE_BF16: {
		const float alpha = 1.0f;
		const float beta = (bias != nullptr && M == 1) ? 1.0f : 0.0f;
		maybe_copy_bias_for_single_row(reinterpret_cast<nv_bfloat16 *>(out), reinterpret_cast<const nv_bfloat16 *>(bias), M, N, stream);
		LLAISYS_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
									  &alpha,
						  reinterpret_cast<const nv_bfloat16 *>(weight), CUDA_R_16BF, k,
									  reinterpret_cast<const nv_bfloat16 *>(in), CUDA_R_16BF, k,
									  &beta,
									  reinterpret_cast<nv_bfloat16 *>(out), CUDA_R_16BF, m,
									  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		if (bias != nullptr && M != 1) {
			add_bias_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
				reinterpret_cast<nv_bfloat16 *>(out), reinterpret_cast<const nv_bfloat16 *>(bias), M, N);
			LLAISYS_CUDA_CHECK(cudaGetLastError());
		}
		break;
	}
	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}

}
} // namespace llaisys::ops::nvidia