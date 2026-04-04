#include "sampling_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/cuda_cast.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <mutex>

namespace {

constexpr int kBlockSize = 256;

std::mutex g_sampling_workspace_mutex;
float *g_workspace_logits = nullptr;
float *g_workspace_selected_logits = nullptr;
int32_t *g_workspace_selected_indices = nullptr;
size_t g_workspace_capacity = 0;

void ensure_sampling_workspace(size_t n_elem) {
	if (n_elem <= g_workspace_capacity) {
		return;
	}

	if (g_workspace_logits != nullptr) {
		LLAISYS_CUDA_CHECK(cudaFree(g_workspace_logits));
		g_workspace_logits = nullptr;
	}
	if (g_workspace_selected_logits != nullptr) {
		LLAISYS_CUDA_CHECK(cudaFree(g_workspace_selected_logits));
		g_workspace_selected_logits = nullptr;
	}
	if (g_workspace_selected_indices != nullptr) {
		LLAISYS_CUDA_CHECK(cudaFree(g_workspace_selected_indices));
		g_workspace_selected_indices = nullptr;
	}

	LLAISYS_CUDA_CHECK(cudaMalloc(&g_workspace_logits, n_elem * sizeof(float)));
	LLAISYS_CUDA_CHECK(cudaMalloc(&g_workspace_selected_logits, n_elem * sizeof(float)));
	LLAISYS_CUDA_CHECK(cudaMalloc(&g_workspace_selected_indices, n_elem * sizeof(int32_t)));
	g_workspace_capacity = n_elem;
}

__device__ __forceinline__ uint64_t splitmix64_next(uint64_t &x) {
	x += 0x9E3779B97F4A7C15ull;
	uint64_t z = x;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

__device__ __forceinline__ float uniform01(uint64_t &state) {
	// Use top 24 bits to build a float in [0, 1).
	constexpr float inv_2pow24 = 1.0f / 16777216.0f;
	return static_cast<float>(splitmix64_next(state) >> 40) * inv_2pow24;
}


template <typename T>
__global__ void sampling_top1_kernel(std::int64_t *out, const T *logits, size_t vocab_size) {
	const size_t b = static_cast<size_t>(blockIdx.x);
	const int tid = static_cast<int>(threadIdx.x);
	const T *batch_logits = logits + b * vocab_size;

	__shared__ float s_values[kBlockSize];
	__shared__ int32_t s_indices[kBlockSize];

	float local_best = -FLT_MAX;
	int32_t local_idx = -1;
	for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += blockDim.x) {
		const float v = llaisys::utils::cuda::to_float(batch_logits[i]);
		if (v > local_best || (v == local_best && static_cast<int32_t>(i) < local_idx)) {
			local_best = v;
			local_idx = static_cast<int32_t>(i);
		}
	}

	s_values[tid] = local_best;
	s_indices[tid] = local_idx;
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			const float v0 = s_values[tid];
			const float v1 = s_values[tid + stride];
			const int32_t i0 = s_indices[tid];
			const int32_t i1 = s_indices[tid + stride];
			if (v1 > v0 || (v1 == v0 && i1 < i0)) {
				s_values[tid] = v1;
				s_indices[tid] = i1;
			}
		}
		__syncthreads();
	}

	if (tid == 0) {
		out[b] = static_cast<std::int64_t>(s_indices[0]);
	}
}

template <typename T>
__global__ void sampling_kernel(std::int64_t *out, const T *logits, float *workspace_logits,
								float *workspace_selected_logits, int32_t *workspace_selected_indices,
								size_t vocab_size, float temperature,
								int32_t top_k, float top_p, uint64_t seed) {
	const size_t b = static_cast<size_t>(blockIdx.x);
	const int tid = static_cast<int>(threadIdx.x);

	float *row_logits = workspace_logits + b * vocab_size;
	float *row_selected_logits = workspace_selected_logits + b * vocab_size;
	int32_t *row_selected_indices = workspace_selected_indices + b * vocab_size;
	const T *batch_logits = logits + b * vocab_size;

	__shared__ float s_values[kBlockSize];
	__shared__ int32_t s_indices[kBlockSize];
	__shared__ float s_max_logit;
	__shared__ float s_total_exp_all;
	__shared__ int32_t s_selected_count;
	__shared__ float s_total_ref;

	// 1) temperature scaling to workspace
	for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += blockDim.x) {
		row_logits[i] = llaisys::utils::cuda::to_float(batch_logits[i]) / temperature;
	}
	__syncthreads();

	// reduce max logit
	float local_max = -FLT_MAX;
	for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += blockDim.x) {
		local_max = fmaxf(local_max, row_logits[i]);
	}
	s_values[tid] = local_max;
	__syncthreads();
	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		if (tid < stride) {
			s_values[tid] = fmaxf(s_values[tid], s_values[tid + stride]);
		}
		__syncthreads();
	}
	if (tid == 0) {
		s_max_logit = s_values[0];
		s_selected_count = 0;
		s_total_ref = 0.0f;
	}
	__syncthreads();

	const bool use_top_p = (top_p > 0.0f && top_p < 1.0f);
	const bool use_top_k = (top_k > 0);

	// Fast path: no top-k and no top-p => full-vocab multinomial
	if (!use_top_k && !use_top_p) {
		float local_sum = 0.0f;
		for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += blockDim.x) {
			local_sum += expf(row_logits[i] - s_max_logit);
		}
		s_values[tid] = local_sum;
		__syncthreads();
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride) {
				s_values[tid] += s_values[tid + stride];
			}
			__syncthreads();
		}

		if (tid == 0) {
			const float total = fmaxf(s_values[0], 1e-12f);
			uint64_t rng_state = seed != 0
				? (seed ^ (0xD1B54A32D192ED03ull + static_cast<uint64_t>(b) * 0x9E3779B97F4A7C15ull))
				: (static_cast<uint64_t>(clock64()) ^
				(0xA24BAED4963EE407ull + static_cast<uint64_t>(b) * 0x9E3779B97F4A7C15ull));

			const float u = uniform01(rng_state) * total;
			float cumulative = 0.0f;
			int32_t sampled = 0;
			for (size_t i = 0; i < vocab_size; ++i) {
				cumulative += expf(row_logits[i] - s_max_logit);
				if (u <= cumulative || i == vocab_size - 1) {
					sampled = static_cast<int32_t>(i);
					break;
				}
			}
			out[b] = static_cast<std::int64_t>(sampled);
		}
		return;
	}

	// If top-p is enabled without top-k, we need all-vocab total mass as reference.
	if (!use_top_k && use_top_p) {
		float local_sum = 0.0f;
		for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += blockDim.x) {
			local_sum += expf(row_logits[i] - s_max_logit);
		}
		s_values[tid] = local_sum;
		__syncthreads();
		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride) {
				s_values[tid] += s_values[tid + stride];
			}
			__syncthreads();
		}
		if (tid == 0) {
			s_total_exp_all = fmaxf(s_values[0], 1e-12f);
		}
		__syncthreads();
	}

	// Iterative parallel top-1 extraction (used for top-k, and top-p without top-k)
	int32_t k_limit = static_cast<int32_t>(vocab_size);
	if (use_top_k) {
		k_limit = min(static_cast<int32_t>(vocab_size), max(top_k, 1));
	}

	for (int32_t it = 0; it < k_limit; ++it) {
		float local_best = -FLT_MAX;
		int32_t local_idx = -1;
		for (size_t i = static_cast<size_t>(tid); i < vocab_size; i += blockDim.x) {
			const float v = row_logits[i];
			if (v > local_best || (v == local_best && static_cast<int32_t>(i) < local_idx)) {
				local_best = v;
				local_idx = static_cast<int32_t>(i);
			}
		}

		s_values[tid] = local_best;
		s_indices[tid] = local_idx;
		__syncthreads();

		for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
			if (tid < stride) {
				const float v0 = s_values[tid];
				const float v1 = s_values[tid + stride];
				const int32_t i0 = s_indices[tid];
				const int32_t i1 = s_indices[tid + stride];
				if (v1 > v0 || (v1 == v0 && i1 < i0)) {
					s_values[tid] = v1;
					s_indices[tid] = i1;
				}
			}
			__syncthreads();
		}

		if (tid == 0) {
			const int32_t best_idx = s_indices[0];
			const float best_val = s_values[0];
			if (best_idx < 0 || best_val <= -FLT_MAX / 2.0f) {
				break;
			}

			row_selected_indices[it] = best_idx;
			row_selected_logits[it] = best_val;
			row_logits[best_idx] = -FLT_MAX;
			s_selected_count = it + 1;

			if (!use_top_k && use_top_p) {
				// Early stop for pure top-p: cumulative over sorted values against all-vocab mass.
				s_total_ref += expf(best_val - s_max_logit);
				if (s_total_ref >= top_p * s_total_exp_all) {
					break;
				}
			}
		}
		__syncthreads();
	}

	if (tid == 0) {
		const int32_t n_sel = max(s_selected_count, 1);

		// Reference normalization mass.
		float ref_mass = 0.0f;
		if (use_top_k) {
			for (int32_t i = 0; i < n_sel; ++i) {
				ref_mass += expf(row_selected_logits[i] - s_max_logit);
			}
		} else {
			// pure top-p path
			ref_mass = s_total_exp_all;
		}
		ref_mass = fmaxf(ref_mass, 1e-12f);

		int32_t p_size = n_sel;
		if (use_top_p) {
			float cumulative = 0.0f;
			p_size = 0;
			for (int32_t i = 0; i < n_sel; ++i) {
				cumulative += expf(row_selected_logits[i] - s_max_logit) / ref_mass;
				p_size = i + 1;
				if (cumulative >= top_p) {
					break;
				}
			}
			p_size = max(p_size, 1);
		}

		float sample_mass = 0.0f;
		for (int32_t i = 0; i < p_size; ++i) {
			sample_mass += expf(row_selected_logits[i] - s_max_logit);
		}
		sample_mass = fmaxf(sample_mass, 1e-12f);

		uint64_t rng_state = seed != 0
			? (seed ^ (0xD1B54A32D192ED03ull + static_cast<uint64_t>(b) * 0x9E3779B97F4A7C15ull))
			: (static_cast<uint64_t>(clock64()) ^
			   (0xA24BAED4963EE407ull + static_cast<uint64_t>(b) * 0x9E3779B97F4A7C15ull));

		const float u = uniform01(rng_state) * sample_mass;
		float cumulative = 0.0f;
		int32_t sampled_slot = 0;
		for (int32_t i = 0; i < p_size; ++i) {
			cumulative += expf(row_selected_logits[i] - s_max_logit);
			if (u <= cumulative || i == p_size - 1) {
				sampled_slot = i;
				break;
			}
		}

		out[b] = static_cast<std::int64_t>(row_selected_indices[sampled_slot]);
	}
}

} // namespace

namespace llaisys::ops::nvidia {

void sampling(std::byte *out, const std::byte *logits, llaisysDataType_t type,
			  size_t batch_size, size_t vocab_size, float temperature,
			  int32_t top_k, float top_p, uint64_t seed) {
	if (batch_size == 0 || vocab_size == 0) {
		return;
	}

	temperature = std::max(temperature, 1e-6f);
	top_p = std::max(0.0f, std::min(top_p, 1.0f));

	const size_t n_elem = batch_size * vocab_size;

	dim3 grid(static_cast<unsigned int>(batch_size));
	dim3 block(kBlockSize);

	if (top_k == 1) {
		switch (type) {
		case LLAISYS_DTYPE_F32:
			sampling_top1_kernel<<<grid, block>>>(reinterpret_cast<std::int64_t *>(out),
				reinterpret_cast<const float *>(logits), vocab_size);
			break;
		case LLAISYS_DTYPE_BF16:
			sampling_top1_kernel<<<grid, block>>>(reinterpret_cast<std::int64_t *>(out),
				reinterpret_cast<const nv_bfloat16 *>(logits), vocab_size);
			break;
		case LLAISYS_DTYPE_F16:
			sampling_top1_kernel<<<grid, block>>>(reinterpret_cast<std::int64_t *>(out),
				reinterpret_cast<const half *>(logits), vocab_size);
			break;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
		}

		LLAISYS_CUDA_CHECK(cudaGetLastError());
		return;
	}

	// NOTE:
	// The single-thread heap path can become a severe bottleneck on large vocab
	// (e.g. >100k). Keep using the parallel sampling kernel for top_k > 1.

	if (n_elem > g_workspace_capacity) {
		std::lock_guard<std::mutex> lock(g_sampling_workspace_mutex);
		ensure_sampling_workspace(n_elem);
	}

	switch (type) {
	case LLAISYS_DTYPE_F32:
		sampling_kernel<<<grid, block>>>(reinterpret_cast<std::int64_t *>(out),
										 reinterpret_cast<const float *>(logits),
										 g_workspace_logits,
										 g_workspace_selected_logits,
										 g_workspace_selected_indices,
										 vocab_size, temperature, top_k, top_p, seed);
		break;
	case LLAISYS_DTYPE_BF16:
		sampling_kernel<<<grid, block>>>(reinterpret_cast<std::int64_t *>(out),
										 reinterpret_cast<const nv_bfloat16 *>(logits),
										 g_workspace_logits,
										 g_workspace_selected_logits,
										 g_workspace_selected_indices,
										 vocab_size, temperature, top_k, top_p, seed);
		break;
	case LLAISYS_DTYPE_F16:
		sampling_kernel<<<grid, block>>>(reinterpret_cast<std::int64_t *>(out),
										 reinterpret_cast<const half *>(logits),
										 g_workspace_logits,
										 g_workspace_selected_logits,
										 g_workspace_selected_indices,
										 vocab_size, temperature, top_k, top_p, seed);
		break;
	default:
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}

	LLAISYS_CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia

