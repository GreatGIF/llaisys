#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void argmax(std::int64_t *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size);
}