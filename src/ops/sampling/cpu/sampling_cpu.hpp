#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {

void sampling(std::byte *out, const std::byte *logits, llaisysDataType_t type, 
              size_t batch_size, size_t vocab_size, float temperature, 
              int32_t top_k, float top_p, uint64_t seed);

} // namespace llaisys::ops::cpu
