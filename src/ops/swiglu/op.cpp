#include "op.hpp"
#include <cmath>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, const std::vector<size_t> &shape) {
    size_t seq_len = shape[0];
    size_t hid_dim = shape[1];
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
            out[idx] = llaisys::utils::cast<T>(up_val * gate_val / (1 + exp(-gate_val)));
        }
    }
}

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // TO_BE_IMPLEMENTED();
    if (up->ndim() != gate->ndim() || up->ndim() != out->ndim() || up->ndim() != 2) {
        throw std::invalid_argument("The shape of gate, up and out should be equal to 2.");
    }
    if (up->shape()[0] != gate->shape()[0] || up->shape()[1] != gate->shape()[1]) {
        throw std::invalid_argument("The shape of gate, up should be equal.");
    }
    if (up->dtype() != gate->dtype() || up->dtype() != out->dtype()) {
        throw std::invalid_argument("The dtype of gate, up and out should be equal.");
    }

    llaisysDataType_t type = gate->dtype();
    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const float *>(gate->data()), 
        reinterpret_cast<const float *>(up->data()), gate->shape());
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const bf16_t *>(gate->data()), 
        reinterpret_cast<const bf16_t *>(up->data()), gate->shape());
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const fp16_t *>(gate->data()), 
        reinterpret_cast<const fp16_t *>(up->data()), gate->shape());
        break;
    default:
        throw std::invalid_argument("swiglu: unsupported dtype.");
    }
}
} // namespace llaisys::ops
