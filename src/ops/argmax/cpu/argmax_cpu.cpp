#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t size) {
    *max_idx = 0;
    *max_val = vals[0];
    for (size_t i = 1; i < size; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(*max_val)) {
                *max_idx = i;
                *max_val = vals[i];
            }
        } else {
            if (vals[i] > *max_val) {
                *max_idx = i;
                *max_val = vals[i];
            }
        }
    }
}

namespace llaisys::ops::cpu {
void argmax(std::int64_t *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_(max_idx, reinterpret_cast<float *>(max_val),
                reinterpret_cast<const float *>(vals), size);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_(max_idx, reinterpret_cast<llaisys::bf16_t *>(max_val),
                reinterpret_cast<const llaisys::bf16_t *>(vals), size);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_(max_idx, reinterpret_cast<llaisys::fp16_t *>(max_val),
                reinterpret_cast<const llaisys::fp16_t *>(vals), size);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu


