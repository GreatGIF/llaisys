#include "linear_nvidia.cuh"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../utils/nvidia_cast.cuh"
#include "../../../utils/nvidia_check.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <limits>

namespace {

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t M, size_t N) {
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t total = M * N;
	if (idx < total) {
		size_t n = idx % N;
		float v = llaisys::utils::nvidia::to_float(out[idx]) + llaisys::utils::nvidia::to_float(bias[n]);
		out[idx] = llaisys::utils::nvidia::from_float<T>(v);
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

	cublasHandle_t handle = nullptr;
	LLAISYS_CUBLAS_CHECK(cublasCreate(&handle));

	try {
		auto stream = static_cast<cudaStream_t>(llaisys::core::context().runtime().stream());
		LLAISYS_CUBLAS_CHECK(cublasSetStream(handle, stream));

		constexpr int threads_per_block = 256;
		const int blocks_per_grid = static_cast<int>((M * N + threads_per_block - 1) / threads_per_block);

		switch (type) {
		case LLAISYS_DTYPE_F32: {
			const float alpha = 1.0f;
			const float beta = 0.0f;
			LLAISYS_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
											 &alpha,
										     reinterpret_cast<const float *>(weight), k,
											 reinterpret_cast<const float *>(in), k,
											 &beta,
											 reinterpret_cast<float *>(out), m));
			if (bias != nullptr) {
				add_bias_kernel<<<blocks_per_grid, threads_per_block>>>(
					reinterpret_cast<float *>(out), reinterpret_cast<const float *>(bias), M, N);
				LLAISYS_CUDA_CHECK(cudaGetLastError());
			}
			break;
		}
		case LLAISYS_DTYPE_F16: {
			const float alpha = 1.0f;
			const float beta = 0.0f;
			LLAISYS_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
											  &alpha,
									          reinterpret_cast<const half *>(weight), CUDA_R_16F, k,
											  reinterpret_cast<const half *>(in), CUDA_R_16F, k,
											  &beta,
											  reinterpret_cast<half *>(out), CUDA_R_16F, m,
											  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			if (bias != nullptr) {
				add_bias_kernel<<<blocks_per_grid, threads_per_block>>>(
					reinterpret_cast<half *>(out), reinterpret_cast<const half *>(bias), M, N);
				LLAISYS_CUDA_CHECK(cudaGetLastError());
			}
			break;
		}
		case LLAISYS_DTYPE_BF16: {
			const float alpha = 1.0f;
			const float beta = 0.0f;
			LLAISYS_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
											  &alpha,
									  reinterpret_cast<const nv_bfloat16 *>(weight), CUDA_R_16BF, k,
											  reinterpret_cast<const nv_bfloat16 *>(in), CUDA_R_16BF, k,
											  &beta,
											  reinterpret_cast<nv_bfloat16 *>(out), CUDA_R_16BF, m,
											  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			if (bias != nullptr) {
				add_bias_kernel<<<blocks_per_grid, threads_per_block>>>(
					reinterpret_cast<nv_bfloat16 *>(out), reinterpret_cast<const nv_bfloat16 *>(bias), M, N);
				LLAISYS_CUDA_CHECK(cudaGetLastError());
			}
			break;
		}
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
		}
	} catch (...) {
		cublasDestroy(handle);
		throw;
	}

	LLAISYS_CUBLAS_CHECK(cublasDestroy(handle));

}
} // namespace llaisys::ops::nvidia