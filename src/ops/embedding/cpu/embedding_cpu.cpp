#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void embedding_(T* out, const std::int64_t* index, const T* weight, size_t index_size, size_t row_size) {
#ifdef ENABLE_OPENMP
    #pragma omp parallel for schedule(static) if (index_size >= 4096)
#endif
    for (size_t i = 0; i < index_size; i++) {
        for (size_t j = 0; j < row_size; j++) {
            out[i * row_size + j] = weight[index[i] * row_size + j];
        }
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::int64_t *index, const std::byte *weight, llaisysDataType_t type, size_t index_size, size_t row_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_(reinterpret_cast<float *>(out),index, 
                   reinterpret_cast<const float *>(weight), index_size, row_size);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_(reinterpret_cast<bf16_t *>(out),index, 
                   reinterpret_cast<const bf16_t *>(weight), index_size, row_size);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_(reinterpret_cast<fp16_t *>(out),index, 
                   reinterpret_cast<const fp16_t *>(weight), index_size, row_size);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}