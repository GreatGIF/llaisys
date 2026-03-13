#include "sampling_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

template <typename T>
void sampling_kernel_(int64_t *out, const T *logits, size_t batch_size, size_t vocab_size, 
                      float temperature, int32_t top_k, float top_p, uint64_t seed) {
    std::mt19937_64 generator(seed == 0 ? std::random_device{}() : seed);
    
#ifdef ENABLE_OPENMP
    #pragma omp parallel for schedule(static) if (batch_size >= 4)
#endif
    for (size_t b = 0; b < batch_size; b++) {
        // Create indices and corresponding logits for this batch
        std::vector<std::pair<float, int32_t>> logits_with_idx;
        logits_with_idx.reserve(vocab_size);
        
        const T *batch_logits = logits + b * vocab_size;
        
        // Apply temperature scaling
        for (size_t i = 0; i < vocab_size; i++) {
            float scaled_logit = llaisys::utils::cast<float>(batch_logits[i]) / std::max(temperature, 1e-6f);
            logits_with_idx.emplace_back(scaled_logit, static_cast<int32_t>(i));
        }
        
        // Sort by logits in descending order
        std::sort(logits_with_idx.begin(), logits_with_idx.end(),
                  [](const auto &a, const auto &b) { return a.first > b.first; });
        
        // Apply Top-K filter
        int32_t k_size = vocab_size;
        if (top_k > 0 && top_k < static_cast<int32_t>(vocab_size)) {
            k_size = top_k;
        }
        
        // Compute exp(logits) for top-k
        float max_logit = logits_with_idx[0].first;
        std::vector<float> probs;
        probs.reserve(k_size);
        float total_prob = 0.0f;
        
        for (int32_t i = 0; i < k_size; i++) {
            float exp_logit = std::exp(logits_with_idx[i].first - max_logit);
            probs.push_back(exp_logit);
            total_prob += exp_logit;
        }
        
        // Normalize probabilities
        for (auto &p : probs) {
            p /= total_prob;
        }
        
        // Apply Top-P (nucleus) filter
        int32_t p_size = k_size;
        if (top_p > 0.0f && top_p < 1.0f) {
            float cumulative_prob = 0.0f;
            p_size = 0;
            for (int32_t i = 0; i < k_size; i++) {
                cumulative_prob += probs[i];
                p_size = i + 1;
                if (cumulative_prob >= top_p) {
                    break;
                }
            }
            
            // Renormalize after filtering
            total_prob = 0.0f;
            for (int32_t i = 0; i < p_size; i++) {
                total_prob += probs[i];
            }
            for (int32_t i = 0; i < p_size; i++) {
                probs[i] /= total_prob;
            }
        }
        
        // Multinomial sampling
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float rand_val = dist(generator);
        float cumulative = 0.0f;
        int32_t sampled_idx = 0;
        
        for (int32_t i = 0; i < p_size; i++) {
            cumulative += probs[i];
            if (rand_val <= cumulative) {
                sampled_idx = i;
                break;
            }
        }
        
        out[b] = logits_with_idx[sampled_idx].second;
    }
}

namespace llaisys::ops::cpu {

void sampling(std::byte *out, const std::byte *logits, llaisysDataType_t type, 
              size_t batch_size, size_t vocab_size, float temperature, 
              int32_t top_k, float top_p, uint64_t seed) {
    
    // Ensure valid parameters
    temperature = std::max(temperature, 1e-6f);
    top_p = std::max(0.0f, std::min(top_p, 1.0f));
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        sampling_kernel_(reinterpret_cast<int64_t *>(out),
                        reinterpret_cast<const float *>(logits),
                        batch_size, vocab_size, temperature, top_k, top_p, seed);
        break;
    case LLAISYS_DTYPE_BF16:
        sampling_kernel_(reinterpret_cast<int64_t *>(out),
                        reinterpret_cast<const llaisys::bf16_t *>(logits),
                        batch_size, vocab_size, temperature, top_k, top_p, seed);
        break;
    case LLAISYS_DTYPE_F16:
        sampling_kernel_(reinterpret_cast<int64_t *>(out),
                        reinterpret_cast<const llaisys::fp16_t *>(logits),
                        batch_size, vocab_size, temperature, top_k, top_p, seed);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
