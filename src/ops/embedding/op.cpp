#include "op.hpp"

template <typename T1, typename T2>
void embedding_(T1* out, const T2* index, const T1* weight, size_t index_size, size_t row_size) {
#ifdef ENABLE_OPENMP
    #pragma omp parallel for schedule(static) if (index_size >= 4096)
#endif
    for (size_t i = 0; i < index_size; i++) {
        for (size_t j = 0; j < row_size; j++) {
            out[i * row_size + j] = weight[index[i] * row_size + j];
        }
    }
}


namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // TO_BE_IMPLEMENTED();
    if (weight->dtype() != out->dtype()) {
        throw std::invalid_argument("argmax: weight and out must have the same dtype");
    }

    llaisysDataType_t type = weight->dtype();
    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_(reinterpret_cast<float *>(out->data()), reinterpret_cast<const std::int64_t *>(index->data()), 
        reinterpret_cast<const float *>(weight->data()), index->numel(), weight->shape()[1]);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_(reinterpret_cast<bf16_t *>(out->data()), reinterpret_cast<const std::int64_t *>(index->data()), 
        reinterpret_cast<const bf16_t *>(weight->data()), index->numel(), weight->shape()[1]);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_(reinterpret_cast<fp16_t *>(out->data()), reinterpret_cast<const std::int64_t *>(index->data()), 
        reinterpret_cast<const fp16_t *>(weight->data()), index->numel(), weight->shape()[1]);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

}
} // namespace llaisys::ops
