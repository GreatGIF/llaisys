#include "op.hpp"

template <typename T1, typename T2>
void argmax_(T1 *max_idx, T2*max_val, const T2 *vals, size_t size) {
    *max_idx = 0;
    *max_val = vals[0];
    for (size_t i = 1; i < size; i++) {
        if constexpr (std::is_same_v<T2, llaisys::bf16_t> || std::is_same_v<T2, llaisys::fp16_t>) {
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

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    if (vals->ndim() != 1) {
        throw std::invalid_argument("argmax: vals must be a 1D tensor");
    }
    if (vals->numel() == 0) {
        throw std::invalid_argument("argmax: vals is empty");
    }
    if (max_val->shape() != std::vector<size_t>{1} ||
        max_idx->shape() != std::vector<size_t>{1}) {
        throw std::invalid_argument("argmax: max_val and max_idx must have shape [1]");
    }
    if (max_idx->dtype() != LLAISYS_DTYPE_I64) {
        throw std::invalid_argument("argmax: max_idx must have dtype int64");
    }
    if (max_val->dtype() != vals->dtype()) {
        throw std::invalid_argument("argmax: max_val and vals must have the same dtype");
    }

    llaisysDataType_t type = vals->dtype();
    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_(reinterpret_cast<std::int64_t *>(max_idx->data()), reinterpret_cast<float *>(max_val->data()), 
        reinterpret_cast<float *>(vals->data()), vals->numel());
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_(reinterpret_cast<std::int64_t *>(max_idx->data()), reinterpret_cast<llaisys::bf16_t *>(max_val->data()), 
        reinterpret_cast<llaisys::bf16_t *>(vals->data()), vals->numel());
        break;
    case LLAISYS_DTYPE_F16:
        argmax_(reinterpret_cast<std::int64_t *>(max_idx->data()), reinterpret_cast<llaisys::fp16_t *>(max_val->data()), 
        reinterpret_cast<llaisys::fp16_t *>(vals->data()), vals->numel());
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops
