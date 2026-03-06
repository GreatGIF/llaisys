#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {

enum class LinearCPUStrategy : uint8_t {
	NAIVE_SMALL = 0,
	OMP_PARALLEL = 1,
	OMP_TILED = 2,
	BLAS_GEMM = 3,
};

LinearCPUStrategy choose_linear_strategy(size_t M, size_t N, size_t K, bool can_use_blas = false);

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type, size_t M, size_t N, size_t K);

} // namespace llaisys::ops::cpu
