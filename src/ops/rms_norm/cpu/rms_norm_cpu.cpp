#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t m, size_t n,
               float eps) {
    std::vector<float> sum(m, 0);
#ifdef ENABLE_OPENMP
    #pragma omp parallel for schedule(static) if (m >= 16)
#endif
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float x = llaisys::utils::cast<float>(in[i * n + j]);
            sum[i] += x * x;
        }
        sum[i] = std::sqrt(sum[i] / static_cast<float>(n) + eps);
    }
#ifdef ENABLE_OPENMP
    #pragma omp parallel for schedule(static) if (m >= 16)
#endif
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                          std::is_same_v<T, llaisys::fp16_t>) {
                out[i * n + j] = llaisys::utils::cast<T>(
                    llaisys::utils::cast<float>(in[i * n + j]) *
                    llaisys::utils::cast<float>(weight[j]) / sum[i]);
            } else {
                out[i * n + j] = in[i * n + j] * weight[j] / sum[i];
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, size_t m, size_t n, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), m, n, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                         reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), m,
                         n, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                         reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), m,
                         n, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
