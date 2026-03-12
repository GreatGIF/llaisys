#include "swiglu_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t seq_len, size_t hid_dim) {
#ifdef ENABLE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if (seq_len * hid_dim >= 256)
#endif
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < hid_dim; j++) {
            float gate_val = 0, up_val = 0;
            size_t idx = i * hid_dim + j;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                gate_val = llaisys::utils::cast<float>(gate[idx]);
                up_val = llaisys::utils::cast<float>(up[idx]);
            } else {
                gate_val = gate[idx];
                up_val = up[idx];
            }
            out[idx] =
                llaisys::utils::cast<T>(up_val * gate_val / (1 + exp(-gate_val)));
        }
    }
}

namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t seq_len, size_t hid_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(gate),
                       reinterpret_cast<const float *>(up), seq_len, hid_dim);
    case LLAISYS_DTYPE_BF16:
        return swiglu_(reinterpret_cast<llaisys::bf16_t *>(out),
                       reinterpret_cast<const llaisys::bf16_t *>(gate),
                       reinterpret_cast<const llaisys::bf16_t *>(up), seq_len,
                       hid_dim);
    case LLAISYS_DTYPE_F16:
        return swiglu_(reinterpret_cast<llaisys::fp16_t *>(out),
                       reinterpret_cast<const llaisys::fp16_t *>(gate),
                       reinterpret_cast<const llaisys::fp16_t *>(up), seq_len,
                       hid_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
