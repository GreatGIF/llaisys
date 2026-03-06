#include "linear_cpu.hpp"

#include "../../../utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>

#if defined(LLAISYS_ENABLE_CPU_BLAS)
#if __has_include(<cblas.h>)
#include <cblas.h>
#define LLAISYS_HAS_CBLAS 1
#else
#define LLAISYS_HAS_CBLAS 0
#endif
#else
#define LLAISYS_HAS_CBLAS 0
#endif

namespace llaisys::ops::cpu {

namespace {

// size_t read_block_size_from_env(const char *name, size_t default_value) {
// 	const char *value = std::getenv(name);
// 	if (value == nullptr || value[0] == '\0') {
// 		return default_value;
// 	}

// 	char *end = nullptr;
// 	unsigned long parsed = std::strtoul(value, &end, 10);
// 	if (end == value || *end != '\0' || parsed == 0UL) {
// 		return default_value;
// 	}
// 	return static_cast<size_t>(parsed);
// }

// struct LinearTileConfig {
// 	size_t BM;
// 	size_t BN;
// 	size_t BK;
// };

// LinearTileConfig get_tiled_config() {
// 	static const size_t BM = read_block_size_from_env("LLAISYS_LINEAR_BM", 32);
// 	static const size_t BN = read_block_size_from_env("LLAISYS_LINEAR_BN", 64);
// 	static const size_t BK = read_block_size_from_env("LLAISYS_LINEAR_BK", 64);
// 	return {BM, BN, BK};
// }

// template <typename T>
// inline float load_as_f32(const T *ptr, size_t idx) {
// 	if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
// 		return llaisys::utils::cast<float>(ptr[idx]);
// 	} else {
// 		return static_cast<float>(ptr[idx]);
// 	}
// }

// template <typename T>
// inline T cast_from_f32(float v) {
// 	if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
// 		return llaisys::utils::cast<T>(v);
// 	} else {
// 		return static_cast<T>(v);
// 	}
// }

inline bool can_use_blas_for_type(llaisysDataType_t type) {
	return LLAISYS_HAS_CBLAS && type == LLAISYS_DTYPE_F32;
}

} // namespace

LinearCPUStrategy choose_linear_strategy(size_t M, size_t N, size_t K, bool can_use_blas) {
#ifdef ENABLE_OPENMP
	constexpr size_t kMediumWorkload = 64 * 1024;
	constexpr size_t kLargeWorkload = 4 * 1024 * 1024;
	const size_t workload = M * N * K;
	if (can_use_blas && workload >= kLargeWorkload) {
		return LinearCPUStrategy::BLAS_GEMM;
	}
	if (workload >= kLargeWorkload) {
		return LinearCPUStrategy::OMP_TILED;
	}
	if (workload >= kMediumWorkload) {
		return LinearCPUStrategy::OMP_PARALLEL;
	}
#else
	const size_t workload = M * N * K;
	constexpr size_t kLargeWorkload = 4 * 1024 * 1024;
	if (can_use_blas && workload >= kLargeWorkload) {
		return LinearCPUStrategy::BLAS_GEMM;
	}
#endif
	return LinearCPUStrategy::NAIVE_SMALL;
}

template <typename T, bool kParallel>
void linear_kernel_naive(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K) {
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
void linear_kernel_tiled(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K) {
	// const auto cfg = get_tiled_config();
	// const size_t BM = cfg.BM;
	// const size_t BN = cfg.BN;
	// const size_t BK = cfg.BK;
	const size_t BM = 32;
	const size_t BN = 16;
	const size_t BK = 16;

#ifdef ENABLE_OPENMP
	#pragma omp parallel for collapse(2) schedule(static)
#endif
	for (size_t bm = 0; bm < M; bm += BM) {
		for (size_t bn = 0; bn < N; bn += BN) {
			const size_t m_end = std::min(bm + BM, M);
			const size_t n_end = std::min(bn + BN, N);
			for (size_t m = bm; m < m_end; ++m) {
				for (size_t n = bn; n < n_end; ++n) {
					float sum = 0.0f;
					for (size_t bk = 0; bk < K; bk += BK) {
						const size_t k_end = std::min(bk + BK, K);
						for (size_t k = bk; k < k_end; ++k) {
							if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
								sum += llaisys::utils::cast<float>(in[m * K + k]) * llaisys::utils::cast<float>(weight[n * K + k]);
							} else {
								sum += in[m * K + k] * weight[n * K + k];
							}
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
	}
}

void linear_kernel_blas_f32(float *out, const float *in, const float *weight, const float *bias, size_t M, size_t N, size_t K) {
#if LLAISYS_HAS_CBLAS
	float beta = 0.0f;
	if (bias != nullptr) {
#ifdef ENABLE_OPENMP
		#pragma omp parallel for schedule(static) if (M * N >= 4096)
#endif
		for (size_t m = 0; m < M; ++m) {
			float *row = out + m * N;
			for (size_t n = 0; n < N; ++n) {
				row[n] = bias[n];
			}
		}
		beta = 1.0f;
	}

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), 
				1.0f, in, static_cast<int>(K), weight, static_cast<int>(K), beta, out, static_cast<int>(N));
#else
	linear_kernel_tiled<float>(out, in, weight, bias, M, N, K);
#endif
}

template <typename T>
void linear_dispatch_by_strategy(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K, 
								 LinearCPUStrategy strategy) {
	switch (strategy) {
	case LinearCPUStrategy::BLAS_GEMM:
		if constexpr (std::is_same_v<T, float>) {
			linear_kernel_blas_f32(out, in, weight, bias, M, N, K);
			break;
		}
		[[fallthrough]];
	case LinearCPUStrategy::OMP_TILED:
		linear_kernel_tiled<T>(out, in, weight, bias, M, N, K);
		break;
	case LinearCPUStrategy::OMP_PARALLEL:
		linear_kernel_naive<T, true>(out, in, weight, bias, M, N, K);
		break;
	case LinearCPUStrategy::NAIVE_SMALL:
	default:
		linear_kernel_naive<T, false>(out, in, weight, bias, M, N, K);
		break;
	}
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t M, size_t N, size_t K) {
	const bool can_use_blas = can_use_blas_for_type(type) &&
					 M <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
					 N <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
					 K <= static_cast<size_t>(std::numeric_limits<int>::max());
	const auto strategy = choose_linear_strategy(M, N, K, can_use_blas);

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
