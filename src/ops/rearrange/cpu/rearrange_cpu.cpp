#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <vector>

template <typename T>
void rearrange_(T *out, const T *in, const std::vector<size_t> &shape,
                const std::vector<ptrdiff_t> &out_strides,
                const std::vector<ptrdiff_t> &in_strides, size_t ndim) {
    std::vector<size_t> index(ndim, 0);
    size_t total_elems = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_elems *= shape[i];
    }

#ifdef ENABLE_OPENMP
    #pragma omp parallel for schedule(static) if (total_elems >= 4096)
#endif
    for (size_t i = 0; i < total_elems; i++) {
        ptrdiff_t out_offset = 0, in_offset = 0;
        for (size_t j = 0; j < ndim; j++) {
            out_offset += static_cast<ptrdiff_t>(index[j]) * out_strides[j];
            in_offset += static_cast<ptrdiff_t>(index[j]) * in_strides[j];
        }
        out[out_offset] = in[in_offset];

        int dim_idx = static_cast<int>(ndim) - 1;
        while (dim_idx >= 0) {
            if (index[dim_idx] == shape[dim_idx] - 1) {
                index[dim_idx] = 0;
                dim_idx--;
            } else {
                index[dim_idx]++;
                break;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides, size_t ndim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rearrange_(reinterpret_cast<float *>(out),
                          reinterpret_cast<const float *>(in), shape,
                          out_strides, in_strides, ndim);
    case LLAISYS_DTYPE_BF16:
        return rearrange_(reinterpret_cast<llaisys::bf16_t *>(out),
                          reinterpret_cast<const llaisys::bf16_t *>(in), shape,
                          out_strides, in_strides, ndim);
    case LLAISYS_DTYPE_F16:
        return rearrange_(reinterpret_cast<llaisys::fp16_t *>(out),
                          reinterpret_cast<const llaisys::fp16_t *>(in), shape,
                          out_strides, in_strides, ndim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
