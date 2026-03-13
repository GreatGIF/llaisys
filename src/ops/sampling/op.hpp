#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

/**
 * @brief Sampling operator for token generation with various decoding strategies
 * @param out Output token indices (shape: [batch_size])
 * @param logits Input logits (shape: [batch_size, vocab_size])
 * @param temperature Temperature for controlling randomness (default: 1.0)
 * @param top_k Top-K sampling: keep only top K logits (0 to disable)
 * @param top_p Top-P (nucleus) sampling: keep logits until cumulative prob >= top_p (0.0-1.0, 0 to disable)
 * @param seed Random seed for reproducibility
 */
void sampling(tensor_t out, tensor_t logits, float temperature = 1.0f, int32_t top_k = 0, 
              float top_p = 0.0f, uint64_t seed = 0);

} // namespace llaisys::ops
