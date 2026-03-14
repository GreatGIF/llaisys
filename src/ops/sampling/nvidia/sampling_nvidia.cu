#include "sampling_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/cuda_cast.hpp"
#include "../../../utils/cuda_check.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cmath>
#include <cstdint>

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t x) {
	x += 0x9E3779B97F4A7C15ULL;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
	return x ^ (x >> 31);
}

__device__ __forceinline__ float uniform01(std::uint64_t seed, std::uint64_t batch_id) {
	std::uint64_t x = seed;
	if (x == 0ULL) {
		x = static_cast<std::uint64_t>(clock64()) ^ (batch_id * 0x9E3779B97F4A7C15ULL);
	}
	const std::uint64_t r = splitmix64(x ^ (batch_id * 0xD1342543DE82EF95ULL));
	const double u = static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
	return static_cast<float>(u);
}

template <typename T>
__global__ void sampling_softmax_no_filter_kernel(std::int64_t *out, const T *logits,
										 size_t batch_size, size_t vocab_size,
										 float inv_temperature,
										 std::uint64_t seed) {
	const size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (b >= batch_size) {
		return;
	}

	const T *batch_logits = logits + b * vocab_size;

	if (vocab_size == 0) {
		out[b] = 0;
		return;
	}

	float max_logit = llaisys::utils::cuda::to_float(batch_logits[0]) * inv_temperature;
	std::int32_t argmax_idx = 0;
	for (size_t i = 1; i < vocab_size; ++i) {
		const float s = llaisys::utils::cuda::to_float(batch_logits[i]) * inv_temperature;
		if (s > max_logit) {
			max_logit = s;
			argmax_idx = static_cast<std::int32_t>(i);
		}
	}

	float total_prob = 0.0f;
	for (size_t i = 0; i < vocab_size; ++i) {
		const float s = llaisys::utils::cuda::to_float(batch_logits[i]) * inv_temperature;
		total_prob += expf(s - max_logit);
	}

	if (total_prob <= 0.0f) {
		out[b] = static_cast<std::int64_t>(argmax_idx);
		return;
	}

	const float rand_val = uniform01(seed, static_cast<std::uint64_t>(b));
	float cumulative = 0.0f;
	std::int32_t sampled_idx = static_cast<std::int32_t>(vocab_size - 1);

	for (size_t i = 0; i < vocab_size; ++i) {
		const float s = llaisys::utils::cuda::to_float(batch_logits[i]) * inv_temperature;
		const float p = expf(s - max_logit) / total_prob;
		cumulative += p;
		if (rand_val <= cumulative) {
			sampled_idx = static_cast<std::int32_t>(i);
			break;
		}
	}

	out[b] = static_cast<std::int64_t>(sampled_idx);
}

template <typename T>
__global__ void sampling_full_gpu_kernel(std::int64_t *out, const T *logits,
									 float *scores_workspace,
									 std::int32_t *indices_workspace,
									 float *probs_workspace,
									 size_t batch_size, size_t vocab_size,
									 float inv_temperature,
									 std::int32_t k_size, float top_p,
									 std::uint64_t seed) {
	const size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (b >= batch_size) {
		return;
	}

	float *scores = scores_workspace + b * vocab_size;
	std::int32_t *indices = indices_workspace + b * vocab_size;
	float *probs = probs_workspace + b * static_cast<size_t>(k_size);
	const T *batch_logits = logits + b * vocab_size;

	for (size_t i = 0; i < vocab_size; ++i) {
		scores[i] = llaisys::utils::cuda::to_float(batch_logits[i]) * inv_temperature;
		indices[i] = static_cast<std::int32_t>(i);
	}

	// Partial selection sort: keep the first k_size entries as sorted top-k.
	for (std::int32_t i = 0; i < k_size; ++i) {
		std::int32_t best = i;
		for (std::int32_t j = i + 1; j < static_cast<std::int32_t>(vocab_size); ++j) {
			if (scores[j] > scores[best]) {
				best = j;
			}
		}
		if (best != i) {
			const float tmp_score = scores[i];
			scores[i] = scores[best];
			scores[best] = tmp_score;

			const std::int32_t tmp_idx = indices[i];
			indices[i] = indices[best];
			indices[best] = tmp_idx;
		}
	}

	const float max_logit = scores[0];
	float total_prob = 0.0f;

	for (std::int32_t i = 0; i < k_size; ++i) {
		const float e = expf(scores[i] - max_logit);
		probs[i] = e;
		total_prob += e;
	}

	if (total_prob <= 0.0f) {
		out[b] = static_cast<std::int64_t>(indices[0]);
		return;
	}

	for (std::int32_t i = 0; i < k_size; ++i) {
		probs[i] /= total_prob;
	}

	std::int32_t p_size = k_size;
	if (top_p > 0.0f && top_p < 1.0f) {
		float cumulative_prob = 0.0f;
		p_size = 0;
		for (std::int32_t i = 0; i < k_size; ++i) {
			cumulative_prob += probs[i];
			p_size = i + 1;
			if (cumulative_prob >= top_p) {
				break;
			}
		}

		total_prob = 0.0f;
		for (std::int32_t i = 0; i < p_size; ++i) {
			total_prob += probs[i];
		}

		if (total_prob > 0.0f) {
			for (std::int32_t i = 0; i < p_size; ++i) {
				probs[i] /= total_prob;
			}
		}
	}

	const float rand_val = uniform01(seed, static_cast<std::uint64_t>(b));
	float cumulative = 0.0f;
	std::int32_t sampled_idx = p_size - 1;

	for (std::int32_t i = 0; i < p_size; ++i) {
		cumulative += probs[i];
		if (rand_val <= cumulative) {
			sampled_idx = i;
			break;
		}
	}

	out[b] = static_cast<std::int64_t>(indices[sampled_idx]);
}

namespace llaisys::ops::nvidia {

void sampling(std::byte *out, const std::byte *logits, llaisysDataType_t type,
			  size_t batch_size, size_t vocab_size, float temperature,
			  int32_t top_k, float top_p, uint64_t seed) {
	// Keep parameter behavior consistent with CPU implementation.
	temperature = std::max(temperature, 1e-6f);
	top_p = std::max(0.0f, std::min(top_p, 1.0f));
	const float inv_temperature = 1.0f / temperature;

	constexpr int threads_per_block = 128;
	const int sample_blocks = static_cast<int>((batch_size + threads_per_block - 1) / threads_per_block);

	// Fast path: no top-k / top-p filtering required.
	// This avoids O(V^2) selection sort and large temporary allocations.
	const bool no_top_k = (top_k <= 0 || top_k >= static_cast<std::int32_t>(vocab_size));
	const bool no_top_p = !(top_p > 0.0f && top_p < 1.0f);
	if (no_top_k && no_top_p) {
		switch (type) {
		case LLAISYS_DTYPE_F32:
			sampling_softmax_no_filter_kernel<<<sample_blocks, threads_per_block>>>(
				reinterpret_cast<std::int64_t *>(out), reinterpret_cast<const float *>(logits),
				batch_size, vocab_size, inv_temperature, seed);
			LLAISYS_CUDA_CHECK(cudaGetLastError());
			return;
		case LLAISYS_DTYPE_BF16:
			sampling_softmax_no_filter_kernel<<<sample_blocks, threads_per_block>>>(
				reinterpret_cast<std::int64_t *>(out), reinterpret_cast<const nv_bfloat16 *>(logits),
				batch_size, vocab_size, inv_temperature, seed);
			LLAISYS_CUDA_CHECK(cudaGetLastError());
			return;
		case LLAISYS_DTYPE_F16:
			sampling_softmax_no_filter_kernel<<<sample_blocks, threads_per_block>>>(
				reinterpret_cast<std::int64_t *>(out), reinterpret_cast<const half *>(logits),
				batch_size, vocab_size, inv_temperature, seed);
			LLAISYS_CUDA_CHECK(cudaGetLastError());
			return;
		default:
			EXCEPTION_UNSUPPORTED_DATATYPE(type);
		}
	}

	const size_t total_size = batch_size * vocab_size;

	std::int32_t k_size = static_cast<std::int32_t>(vocab_size);
	if (top_k > 0 && top_k < static_cast<std::int32_t>(vocab_size)) {
		k_size = top_k;
	}

	float *d_scores = nullptr;
	std::int32_t *d_indices = nullptr;
	float *d_probs_workspace = nullptr;

	LLAISYS_CUDA_CHECK(cudaMalloc(&d_scores, total_size * sizeof(float)));
	LLAISYS_CUDA_CHECK(cudaMalloc(&d_indices, total_size * sizeof(std::int32_t)));
	LLAISYS_CUDA_CHECK(cudaMalloc(&d_probs_workspace, batch_size * static_cast<size_t>(k_size) * sizeof(float)));

	auto cleanup = [&]() {
		if (d_probs_workspace != nullptr) {
			LLAISYS_CUDA_CHECK(cudaFree(d_probs_workspace));
			d_probs_workspace = nullptr;
		}
		if (d_indices != nullptr) {
			LLAISYS_CUDA_CHECK(cudaFree(d_indices));
			d_indices = nullptr;
		}
		if (d_scores != nullptr) {
			LLAISYS_CUDA_CHECK(cudaFree(d_scores));
			d_scores = nullptr;
		}
	};

	switch (type) {
	case LLAISYS_DTYPE_F32:
		sampling_full_gpu_kernel<<<sample_blocks, threads_per_block>>>(
			reinterpret_cast<std::int64_t *>(out), reinterpret_cast<const float *>(logits),
			d_scores, d_indices, d_probs_workspace,
			batch_size, vocab_size, inv_temperature, k_size, top_p, seed);
		LLAISYS_CUDA_CHECK(cudaGetLastError());
		break;
	case LLAISYS_DTYPE_BF16:
		sampling_full_gpu_kernel<<<sample_blocks, threads_per_block>>>(
			reinterpret_cast<std::int64_t *>(out), reinterpret_cast<const nv_bfloat16 *>(logits),
			d_scores, d_indices, d_probs_workspace,
			batch_size, vocab_size, inv_temperature, k_size, top_p, seed);
		LLAISYS_CUDA_CHECK(cudaGetLastError());
		break;
	case LLAISYS_DTYPE_F16:
		sampling_full_gpu_kernel<<<sample_blocks, threads_per_block>>>(
			reinterpret_cast<std::int64_t *>(out), reinterpret_cast<const half *>(logits),
			d_scores, d_indices, d_probs_workspace,
			batch_size, vocab_size, inv_temperature, k_size, top_p, seed);
		LLAISYS_CUDA_CHECK(cudaGetLastError());
		break;
	default:
		cleanup();
		EXCEPTION_UNSUPPORTED_DATATYPE(type);
	}

	cleanup();
}

} // namespace llaisys::ops::nvidia
