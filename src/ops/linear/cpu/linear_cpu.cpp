#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

LinearCPUStrategy choose_linear_strategy(size_t M, size_t N, size_t K) {
#ifdef ENABLE_OPENMP
    constexpr size_t kSmallWorkload = 256;
	const size_t workload = M * N;
	if (workload >= kSmallWorkload) {
		return LinearCPUStrategy::OMP_PARALLEL;
	}
#endif
	return LinearCPUStrategy::NAIVE_SMALL;
}

template <typename T, bool kParallel>
void linear_kernel(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K) {
#ifdef ENABLE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if (kParallel)
#endif
	for (size_t m = 0; m < M; m++) {
		for (size_t n = 0; n < N; n++) {
            // 使用fp32累加
            // y = x * w^T + b
			float sum = 0.0f;
			for (size_t k = 0; k < K; k++) {
				if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
					sum += llaisys::utils::cast<float>(in[m * K + k]) * llaisys::utils::cast<float>(weight[n * K + k]);
				} else {
					sum += in[m * K + k] * weight[n * K + k];
				}
			}
			if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
				const float b = (bias == nullptr) ? 0.0f : llaisys::utils::cast<float>(bias[n]);
				out[m * N + n] = llaisys::utils::cast<T>(sum + b);
			} else {
				out[m * N + n] = sum + (bias == nullptr ? 0.0f : bias[n]);
			}
		}
	}
}

template <typename T>
void linear_dispatch_by_strategy(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K, LinearCPUStrategy strategy) {
	switch (strategy) {
	case LinearCPUStrategy::OMP_PARALLEL:
		linear_kernel<T, true>(out, in, weight, bias, M, N, K);
		break;
	case LinearCPUStrategy::NAIVE_SMALL:
	default:
		linear_kernel<T, false>(out, in, weight, bias, M, N, K);
		break;
	}
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t M, size_t N, size_t K) {
	const auto strategy = choose_linear_strategy(M, N, K);

	switch (type) {
	case LLAISYS_DTYPE_F32:
		return linear_dispatch_by_strategy(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), M, N, K, strategy);
	case LLAISYS_DTYPE_BF16:
		return linear_dispatch_by_strategy(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), M, N, K, strategy);
	case LLAISYS_DTYPE_F16:
		return linear_dispatch_by_strategy(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), M, N, K, strategy);
	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}

} // namespace llaisys::ops::cpu
