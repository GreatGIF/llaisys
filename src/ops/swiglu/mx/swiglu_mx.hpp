#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::mx {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t seq_len, size_t hid_dim);
} // namespace llaisys::ops::mx
