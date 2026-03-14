#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::mx {
void rope(std::byte *out, const std::byte *in, const std::int64_t *pos_ids,
          llaisysDataType_t type, size_t seq_len, size_t head_num,
          size_t head_dim, float theta);
} // namespace llaisys::ops::mx
