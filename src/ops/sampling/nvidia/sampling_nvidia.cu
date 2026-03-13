#include "sampling_nvidia.cuh"

#include "../../../utils.hpp"
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


namespace llaisys::ops::nvidia {

// Helper function for atomic max with floats
__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*) address;
    int old = *address_as_int, assumed;
    while (assumed = old, (old = atomicCAS(address_as_int, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))))) != assumed);
    return __int_as_float(old);
}

// Kernel for temperature-scaled softmax and sampling
template <typename T>
__global__ void sampling_kernel(
    int64_t *out,
    const T *logits,
    size_t batch_size,
    size_t vocab_size,
    float temperature,
    int32_t top_k,
    float top_p,
    unsigned long long seed) {
    
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    
    // Initialize random state (only one thread per batch)
    __shared__ curandState state;
    if (threadIdx.x == 0) {
        curand_init(seed + batch_idx, 0, 0, &state);
    }
    __syncthreads();
    
    const T *batch_logits = logits + batch_idx * vocab_size;
    
    // Allocate space for indices and logits in shared memory
    extern __shared__ char shared_mem[];
    float *shared_logits = (float *)shared_mem;
    int32_t *shared_indices = (int32_t *)(shared_logits + vocab_size);
    
    // Copy and scale logits
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        shared_logits[i] = (float)batch_logits[i] / temperature;
        shared_indices[i] = i;
    }
    __syncthreads();
    
    // Find max for numerical stability
    __shared__ float max_logit;
    if (threadIdx.x == 0) max_logit = -1e9f;
    __syncthreads();
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        atomicMaxFloat(&max_logit, shared_logits[i]);
    }
    __syncthreads();
    
    // Compute exp probabilities for all items
    __shared__ float sum_exp;
    if (threadIdx.x == 0) sum_exp = 0.0f;
    __syncthreads();
    
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float exp_logit = expf(shared_logits[i] - max_logit);
        shared_logits[i] = exp_logit;
        atomicAdd(&sum_exp, exp_logit);
    }
    __syncthreads();
    
    // Normalize to get probabilities
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        shared_logits[i] /= (sum_exp + 1e-9f);
    }
    __syncthreads();
    
    // Determine size after top-k filtering
    int k_size = vocab_size;
    if (top_k > 0 && top_k < (int32_t)vocab_size) {
        k_size = top_k;
    }
    
    // Simple selection of top-k (single-threaded for correctness)
    if (threadIdx.x == 0) {
        for (int i = 0; i < k_size; i++) {
            int max_idx = i;
            for (int j = i + 1; j < (int)vocab_size; j++) {
                if (shared_logits[j] > shared_logits[max_idx]) {
                    max_idx = j;
                }
            }
            // Swap logits
            float tmp_logit = shared_logits[i];
            shared_logits[i] = shared_logits[max_idx];
            shared_logits[max_idx] = tmp_logit;
            
            // Swap indices
            int32_t tmp_idx = shared_indices[i];
            shared_indices[i] = shared_indices[max_idx];
            shared_indices[max_idx] = tmp_idx;
        }
    }
    __syncthreads();
    
    // Renormalize after top-k filtering
    __shared__ float sum_topk;
    if (threadIdx.x == 0) sum_topk = 0.0f;
    __syncthreads();
    
    for (int i = threadIdx.x; i < k_size; i += blockDim.x) {
        atomicAdd(&sum_topk, shared_logits[i]);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < k_size; i += blockDim.x) {
        shared_logits[i] /= (sum_topk + 1e-9f);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        // Apply top-p (nucleus) filter
        int p_size = k_size;
        if (top_p > 0.0f && top_p < 1.0f) {
            float cumulative = 0.0f;
            p_size = 0;
            for (int i = 0; i < k_size; i++) {
                cumulative += shared_logits[i];
                p_size = i + 1;
                if (cumulative >= top_p) break;
            }
            
            // Renormalize after top-p filtering
            float total = 0.0f;
            for (int i = 0; i < p_size; i++) {
                total += shared_logits[i];
            }
            for (int i = 0; i < p_size; i++) {
                shared_logits[i] /= (total + 1e-9f);
            }
        }
        
        // Multinomial sampling
        float rand_val = curand_uniform(&state);
        float cumulative = 0.0f;
        int sampled_idx = 0;
        
        for (int i = 0; i < p_size; i++) {
            cumulative += shared_logits[i];
            if (rand_val <= cumulative) {
                sampled_idx = i;
                break;
            }
        }
        
        out[batch_idx] = shared_indices[sampled_idx];
    }
}

void sampling(std::byte *out, const std::byte *logits, llaisysDataType_t type, 
              size_t batch_size, size_t vocab_size, float temperature, 
              int32_t top_k, float top_p, uint64_t seed) {
    
    // Ensure valid parameters
    temperature = fmaxf(temperature, 1e-6f);
    top_p = fmaxf(0.0f, fminf(top_p, 1.0f));
    
    // One block per batch item, enough threads to handle vocab
    int32_t block_size = 256;
    int32_t grid_size = batch_size;
    
    size_t shared_mem_size = (vocab_size * sizeof(float)) + (vocab_size * sizeof(int32_t));
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        sampling_kernel<<<grid_size, block_size, shared_mem_size>>>(
            reinterpret_cast<int64_t *>(out),
            reinterpret_cast<const float *>(logits),
            batch_size, vocab_size, temperature, top_k, top_p, seed);
        break;
    case LLAISYS_DTYPE_BF16:
        sampling_kernel<<<grid_size, block_size, shared_mem_size>>>(
            reinterpret_cast<int64_t *>(out),
            reinterpret_cast<const nv_bfloat16 *>(logits), 
            batch_size, vocab_size, temperature, top_k, top_p, seed);
        break;
    case LLAISYS_DTYPE_F16:
        sampling_kernel<<<grid_size, block_size, shared_mem_size>>>(
            reinterpret_cast<int64_t *>(out),
            reinterpret_cast<const half *>(logits),
            batch_size, vocab_size, temperature, top_k, top_p, seed);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    
    cudaDeviceSynchronize();
}

} // namespace llaisys::ops::nvidia
