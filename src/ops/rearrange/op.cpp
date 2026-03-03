#include "op.hpp"

template <typename T>
void rearrange_(T* out, const T* in, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &out_strides, const std::vector<ptrdiff_t> &in_strides, const size_t ndim) {
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
            out_offset +=  static_cast<ptrdiff_t>(index[j]) * out_strides[j];
            in_offset +=  static_cast<ptrdiff_t>(index[j]) * in_strides[j];
        }
        out[out_offset] = in[in_offset];

        int dim_idx = ndim - 1;
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

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    if (in->ndim() != out->ndim()) {
        throw std::invalid_argument("rearrange: in and out must have the same ndim");
    }
    for (size_t i = 0; i < in->ndim(); i++) {
        if (in->shape()[i] != out->shape()[i]) {
            throw std::invalid_argument("rearrange: in and out must have the same shape");
        }
    }

    llaisysDataType_t type = in->dtype();
    switch (type) {
    case LLAISYS_DTYPE_F32:
        rearrange_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(in->data()), 
        in->shape(), in->strides(), out->strides(), in->ndim());
        break;
    case LLAISYS_DTYPE_BF16:
        rearrange_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(in->data()), 
        in->shape(), in->strides(), out->strides(), in->ndim());
    case LLAISYS_DTYPE_F16:
        rearrange_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(in->data()), 
        in->shape(), in->strides(), out->strides(), in->ndim());
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops
