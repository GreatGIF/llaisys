#include "linear_cpu.hpp"

#include "../../../utils.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#if defined(ENABLE_CPU_MKL) && __has_include(<mkl.h>)
#define LLAISYS_SKIP_SYSTEM_CBLAS 1
#else
#define LLAISYS_SKIP_SYSTEM_CBLAS 0
#endif

#if defined(ENABLE_CPU_BLAS)
#if !LLAISYS_SKIP_SYSTEM_CBLAS && __has_include(<cblas.h>)
#include <cblas.h>
#define LLAISYS_HAS_CBLAS 1
#else
#define LLAISYS_HAS_CBLAS 0
#endif
#else
#define LLAISYS_HAS_CBLAS 0
#endif

#if defined(ENABLE_CPU_MKL)
#if __has_include(<mkl.h>)
#include <mkl.h>
#define LLAISYS_HAS_MKL 1
#else
#define LLAISYS_HAS_MKL 0
#endif
#else
#define LLAISYS_HAS_MKL 0
#endif

#if defined(ENABLE_CPU_ONEDNN)
#if __has_include(<oneapi/dnnl/dnnl.hpp>)
#include <oneapi/dnnl/dnnl.hpp>
#define LLAISYS_HAS_ONEDNN 1
#elif __has_include(<dnnl.hpp>)
#include <dnnl.hpp>
#define LLAISYS_HAS_ONEDNN 1
#else
#define LLAISYS_HAS_ONEDNN 0
#endif
#else
#define LLAISYS_HAS_ONEDNN 0
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

// inline bool env_flag_enabled(const char *name) {
// 	const char *value = std::getenv(name);
// 	if (value == nullptr) {
// 		return false;
// 	}
// 	if (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 ||
// 		std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0) {
// 		return true;
// 	}
// 	return false;
// }

inline const char *strategy_to_string(LinearCPUStrategy s) {
	switch (s) {
	case LinearCPUStrategy::NAIVE_SMALL:
		return "naive";
	case LinearCPUStrategy::OMP_PARALLEL:
		return "omp_parallel";
	case LinearCPUStrategy::OMP_TILED:
		return "omp_tiled";
	case LinearCPUStrategy::BLAS_GEMM:
		return "blas";
	case LinearCPUStrategy::ONEDNN_GEMM:
		return "onednn";
	case LinearCPUStrategy::MKL_GEMM:
		return "mkl";
	default:
		return "unknown";
	}
}

inline bool can_use_blas_for_type(llaisysDataType_t type) {
	return LLAISYS_HAS_CBLAS && type == LLAISYS_DTYPE_F32;
}

inline bool can_use_mkl_for_type(llaisysDataType_t type) {
#if LLAISYS_HAS_MKL
	return type == LLAISYS_DTYPE_F32 || type == LLAISYS_DTYPE_BF16 || type == LLAISYS_DTYPE_F16;
#else
	(void)type;
	return false;
#endif
}

inline bool can_use_onednn_for_type(llaisysDataType_t type) {
#if LLAISYS_HAS_ONEDNN
	return type == LLAISYS_DTYPE_F32 || type == LLAISYS_DTYPE_BF16 || type == LLAISYS_DTYPE_F16;
#else
	(void)type;
	return false;
#endif
}

inline LinearCPUStrategy choose_openmp_sub_strategy(size_t M, size_t N, size_t K) {
#ifdef ENABLE_OPENMP
	constexpr size_t kMediumWorkload = 64 * 1024;
	constexpr size_t kLargeWorkload = 4 * 1024 * 1024;
	const size_t workload = M * N * K;
	if (workload >= kLargeWorkload) {
		return LinearCPUStrategy::OMP_TILED;
	} else if (workload >= kMediumWorkload) {
		return LinearCPUStrategy::OMP_PARALLEL;
	}
#endif
	return LinearCPUStrategy::NAIVE_SMALL;
}

} // namespace

LinearCPUStrategy choose_linear_strategy(size_t M, size_t N, size_t K, llaisysDataType_t type, bool can_use_blas,
								 		bool can_use_onednn, bool can_use_mkl) {
	(void)type;
	if (can_use_onednn) {
		return LinearCPUStrategy::ONEDNN_GEMM;
	}
	if (can_use_mkl) {
		return LinearCPUStrategy::MKL_GEMM;
	}
	if (can_use_blas) {
		return LinearCPUStrategy::BLAS_GEMM;
	}
	return choose_openmp_sub_strategy(M, N, K);
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

bool linear_kernel_blas_f32(float *out, const float *in, const float *weight, const float *bias, size_t M, size_t N, size_t K) {
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
	return true;
#else
	(void)out;
	(void)in;
	(void)weight;
	(void)bias;
	(void)M;
	(void)N;
	(void)K;
	return false;
#endif
}

#if LLAISYS_HAS_ONEDNN
template <typename T>
inline dnnl::memory::data_type to_onednn_dtype();

template <>
inline dnnl::memory::data_type to_onednn_dtype<float>() {
	return dnnl::memory::data_type::f32;
}

template <>
inline dnnl::memory::data_type to_onednn_dtype<llaisys::bf16_t>() {
	return dnnl::memory::data_type::bf16;
}

template <>
inline dnnl::memory::data_type to_onednn_dtype<llaisys::fp16_t>() {
	return dnnl::memory::data_type::f16;
}
#endif

template <typename T>
bool linear_kernel_onednn(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K) {
#if LLAISYS_HAS_ONEDNN
	try {
		auto eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
		auto stream = dnnl::stream(eng);

		const auto dt = to_onednn_dtype<T>();
		const auto src_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(M), static_cast<dnnl::memory::dim>(K)}, dt,
								   {static_cast<dnnl::memory::dim>(K), 1});
		// weight is stored as [N, K], here we create a transposed view [K, N] with stride [1, K]
		const auto weight_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(K), static_cast<dnnl::memory::dim>(N)}, dt,
									 {1, static_cast<dnnl::memory::dim>(K)});
		const auto dst_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(M), static_cast<dnnl::memory::dim>(N)}, dt,
								   {static_cast<dnnl::memory::dim>(N), 1});

		auto src_mem = dnnl::memory(src_md, eng, const_cast<T *>(in));
		auto weight_mem = dnnl::memory(weight_md, eng, const_cast<T *>(weight));
		auto dst_mem = dnnl::memory(dst_md, eng, out);

		if (bias != nullptr) {
			const auto bias_md = dnnl::memory::desc({static_cast<dnnl::memory::dim>(N)}, dt, {1});
			auto bias_mem = dnnl::memory(bias_md, eng, const_cast<T *>(bias));
			auto desc = dnnl::matmul::desc(src_md, weight_md, bias_md, dst_md);
			auto pd = dnnl::matmul::primitive_desc(desc, eng);
			dnnl::matmul(pd).execute(stream,
								 {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weight_mem},
								  {DNNL_ARG_BIAS, bias_mem}, {DNNL_ARG_DST, dst_mem}});
		} else {
			auto desc = dnnl::matmul::desc(src_md, weight_md, dst_md);
			auto pd = dnnl::matmul::primitive_desc(desc, eng);
			dnnl::matmul(pd).execute(stream,
								 {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weight_mem}, {DNNL_ARG_DST, dst_mem}});
		}
		stream.wait();
		return true;
	} catch (...) {
		return false;
	}
#else
	(void)out;
	(void)in;
	(void)weight;
	(void)bias;
	(void)M;
	(void)N;
	(void)K;
	return false;
#endif
}

bool linear_kernel_mkl_f32(float *out, const float *in, const float *weight, const float *bias, size_t M, size_t N, size_t K) {
#if LLAISYS_HAS_MKL
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
	return true;
#else
	(void)out;
	(void)in;
	(void)weight;
	(void)bias;
	(void)M;
	(void)N;
	(void)K;
	return false;
#endif
}

bool linear_kernel_mkl_bf16(llaisys::bf16_t *out, const llaisys::bf16_t *in, const llaisys::bf16_t *weight,
							const llaisys::bf16_t *bias, size_t M, size_t N, size_t K) {
#if LLAISYS_HAS_MKL && defined(MKL_BF16)
	if (M > static_cast<size_t>(std::numeric_limits<MKL_INT>::max()) ||
		N > static_cast<size_t>(std::numeric_limits<MKL_INT>::max()) ||
		K > static_cast<size_t>(std::numeric_limits<MKL_INT>::max())) {
		return false;
	}

	static_assert(sizeof(llaisys::bf16_t) == sizeof(MKL_BF16), "bf16_t and MKL_BF16 must have same size");

	std::vector<float> acc(M * N, 0.0f);
	float beta = 0.0f;
	if (bias != nullptr) {
#ifdef ENABLE_OPENMP
		#pragma omp parallel for schedule(static) if (M * N >= 4096)
#endif
		for (size_t m = 0; m < M; ++m) {
			float *row = acc.data() + m * N;
			for (size_t n = 0; n < N; ++n) {
				row[n] = llaisys::utils::cast<float>(bias[n]);
			}
		}
		beta = 1.0f;
	}

	cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(M), static_cast<MKL_INT>(N),
						   static_cast<MKL_INT>(K), 1.0f,
						   reinterpret_cast<const MKL_BF16 *>(in), static_cast<MKL_INT>(K),
						   reinterpret_cast<const MKL_BF16 *>(weight), static_cast<MKL_INT>(K), beta,
						   acc.data(), static_cast<MKL_INT>(N));

#ifdef ENABLE_OPENMP
	#pragma omp parallel for schedule(static) if (M * N >= 4096)
#endif
	for (size_t i = 0; i < M * N; ++i) {
		out[i] = llaisys::utils::cast<llaisys::bf16_t>(acc[i]);
	}
	return true;
#else
	(void)out;
	(void)in;
	(void)weight;
	(void)bias;
	(void)M;
	(void)N;
	(void)K;
	return false;
#endif
}

bool linear_kernel_mkl_f16(llaisys::fp16_t *out, const llaisys::fp16_t *in, const llaisys::fp16_t *weight,
						  const llaisys::fp16_t *bias, size_t M, size_t N, size_t K) {
#if LLAISYS_HAS_MKL && defined(MKL_HAS_HGEMM)
	if (M > static_cast<size_t>(std::numeric_limits<MKL_INT>::max()) ||
		N > static_cast<size_t>(std::numeric_limits<MKL_INT>::max()) ||
		K > static_cast<size_t>(std::numeric_limits<MKL_INT>::max())) {
		return false;
	}

	static_assert(sizeof(llaisys::fp16_t) == sizeof(MKL_F16), "fp16_t and MKL_F16 must have same size");

	MKL_F16 alpha{};
	const llaisys::fp16_t one = llaisys::utils::cast<llaisys::fp16_t>(1.0f);
	std::memcpy(&alpha, &one, sizeof(alpha));

	MKL_F16 beta{};
	if (bias != nullptr) {
#ifdef ENABLE_OPENMP
		#pragma omp parallel for schedule(static) if (M * N >= 4096)
#endif
		for (size_t m = 0; m < M; ++m) {
			llaisys::fp16_t *row = out + m * N;
			for (size_t n = 0; n < N; ++n) {
				row[n] = bias[n];
			}
		}
		std::memcpy(&beta, &one, sizeof(beta));
	}

	cblas_hgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<MKL_INT>(M), static_cast<MKL_INT>(N),
				static_cast<MKL_INT>(K), alpha, reinterpret_cast<const MKL_F16 *>(in), static_cast<MKL_INT>(K),
				reinterpret_cast<const MKL_F16 *>(weight), static_cast<MKL_INT>(K), beta,
				reinterpret_cast<MKL_F16 *>(out), static_cast<MKL_INT>(N));
	return true;
#else
	(void)out;
	(void)in;
	(void)weight;
	(void)bias;
	(void)M;
	(void)N;
	(void)K;
	return false;
#endif
}

template <typename T>
void linear_execute_by_strategy(T *out, const T *in, const T *weight, const T *bias, size_t M, size_t N, size_t K,
								 LinearCPUStrategy strategy) {
	switch (strategy) {
	case LinearCPUStrategy::ONEDNN_GEMM:
		if (!linear_kernel_onednn<T>(out, in, weight, bias, M, N, K)) {
			linear_execute_by_strategy<T>(out, in, weight, bias, M, N, K, choose_openmp_sub_strategy(M, N, K));
		}
		break;
	case LinearCPUStrategy::MKL_GEMM:
		if constexpr (std::is_same_v<T, float>) {
			if (!linear_kernel_mkl_f32(out, in, weight, bias, M, N, K)) {
				linear_execute_by_strategy<T>(out, in, weight, bias, M, N, K, choose_openmp_sub_strategy(M, N, K));
			}
		} else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
			if (!linear_kernel_mkl_bf16(out, in, weight, bias, M, N, K)) {
				linear_execute_by_strategy<T>(out, in, weight, bias, M, N, K, choose_openmp_sub_strategy(M, N, K));
			}
		} else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
			if (!linear_kernel_mkl_f16(out, in, weight, bias, M, N, K)) {
				linear_execute_by_strategy<T>(out, in, weight, bias, M, N, K, choose_openmp_sub_strategy(M, N, K));
			}
		} else {
			if (!linear_kernel_onednn<T>(out, in, weight, bias, M, N, K)) {
				linear_execute_by_strategy<T>(out, in, weight, bias, M, N, K, choose_openmp_sub_strategy(M, N, K));
			}
		}
		break;
	case LinearCPUStrategy::BLAS_GEMM:
		if constexpr (std::is_same_v<T, float>) {
			if (!linear_kernel_blas_f32(out, in, weight, bias, M, N, K)) {
				linear_execute_by_strategy<T>(out, in, weight, bias, M, N, K, choose_openmp_sub_strategy(M, N, K));
			}
		}
		break;
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
	const bool can_use_onednn = can_use_onednn_for_type(type);
	const bool can_use_mkl = can_use_mkl_for_type(type) &&
					M <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
					N <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
					K <= static_cast<size_t>(std::numeric_limits<int>::max());
	const auto selected = choose_linear_strategy(M, N, K, type, can_use_blas, can_use_onednn, can_use_mkl);
	// const bool log_backend = env_flag_enabled("LLAISYS_LINEAR_LOG_BACKEND");
	// if (log_backend) {
	// 	const char *dtype_str = "unknown";
	// 	switch (type) {
	// 	case LLAISYS_DTYPE_F32:
	// 		dtype_str = "f32";
	// 		break;
	// 	case LLAISYS_DTYPE_F16:
	// 		dtype_str = "f16";
	// 		break;
	// 	case LLAISYS_DTYPE_BF16:
	// 		dtype_str = "bf16";
	// 		break;
	// 	default:
	// 		break;
	// 	}
	// 	std::fprintf(stderr,
	// 		"[llaisys.linear.cpu] dtype=%s M=%zu N=%zu K=%zu backend=%s caps{blas=%d,onednn=%d,mkl=%d}\n",
	// 		dtype_str, M, N, K, strategy_to_string(selected), can_use_blas ? 1 : 0, can_use_onednn ? 1 : 0,
	// 		can_use_mkl ? 1 : 0);
	// }

	switch (type) {
	case LLAISYS_DTYPE_F32:
		return linear_execute_by_strategy(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), M, N, K, selected);
	case LLAISYS_DTYPE_BF16:
		return linear_execute_by_strategy(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), M, N, K, selected);
	case LLAISYS_DTYPE_F16:
		return linear_execute_by_strategy(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), M, N, K, selected);
	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}
}

} // namespace llaisys::ops::cpu
