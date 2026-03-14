#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::mx {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t m, size_t n, float eps);
} // namespace llaisys::ops::mx
