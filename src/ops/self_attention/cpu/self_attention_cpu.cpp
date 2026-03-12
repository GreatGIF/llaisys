#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v,
                     size_t qlen, size_t kvlen, size_t nh, size_t nkvh,
                     size_t hd, float scale) {
    size_t ng = nh / nkvh;
    std::vector<float> attn_score(nh * qlen * kvlen, 0);

#ifdef ENABLE_OPENMP
    #pragma omp parallel for collapse(4) schedule(static) if (nkvh * ng * qlen * kvlen >= 256)
#endif
    for (size_t head1 = 0; head1 < nkvh; head1++) {
        for (size_t head2 = 0; head2 < ng; head2++) {
            for (size_t ql = 0; ql < qlen; ql++) {
                for (size_t kvl = 0; kvl < kvlen; kvl++) {
                    float score = 0.0f;
                    for (size_t d = 0; d < hd; d++) {
                        size_t idx_q = ql * nh * hd + (head1 * ng + head2) * hd + d; // [ql, head, d]
                        size_t idx_k = kvl * nkvh * hd + head1 * hd + d; // [kvl, head, d]
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            score += llaisys::utils::cast<float>(q[idx_q]) * llaisys::utils::cast<float>(k[idx_k]);
                        } else {
                            score += q[idx_q] * k[idx_k];
                        }
                    }
                    score *= scale;
                    // casual mask
                    size_t idx_attn_score = ((head1 * ng + head2) * qlen + ql) * kvlen + kvl; // [head, ql, kvl]
                    attn_score[idx_attn_score] = (kvl > ql + (kvlen - qlen)) ? -INFINITY : score;
                }
            }
        }
    }

    // safe softmax 减去最大值
#ifdef ENABLE_OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if (nh * qlen >= 128)
#endif
    for (size_t head = 0; head < nh; head++) {
        for (size_t ql = 0; ql < qlen; ql++) {
            float max_val = -std::numeric_limits<float>::infinity();
            // 最大值
            for (size_t kvl = 0; kvl < kvlen; kvl++) {
                max_val = std::max(max_val, attn_score[(head * qlen + ql) * kvlen + kvl]);
            }
            // exp & sum
            float sum = 0.0f;
            for (size_t kvl = 0; kvl < kvlen; kvl++) {
                float shifted =
                    attn_score[(head * qlen + ql) * kvlen + kvl] - max_val;
                float e = std::exp(shifted);
                attn_score[(head * qlen + ql) * kvlen + kvl] = e;
                sum += e;
            }
            // normalize
            for (size_t kvl = 0; kvl < kvlen; kvl++) {
                attn_score[(head * qlen + ql) * kvlen + kvl] /= sum;
            }
        }
    }

    // Q * K^T * V [head, ql, d]
#ifdef ENABLE_OPENMP
    #pragma omp parallel for collapse(4) schedule(static) if (nkvh * ng * qlen * hd >= 256)
#endif
    for (size_t head1 = 0; head1 < nkvh; head1++) {
        for (size_t head2 = 0; head2 < ng; head2++) {
            for (size_t ql = 0; ql < qlen; ql++) {
                for (size_t d = 0; d < hd; d++) {
                    float acc = 0;
                    for (size_t kvl = 0; kvl < kvlen; kvl++) {
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                                      std::is_same_v<T, llaisys::fp16_t>) {
                            acc += attn_score[((head1 * ng + head2) * qlen +
                                               ql) *
                                                  kvlen +
                                              kvl] *
                                   llaisys::utils::cast<float>(
                                       v[(kvl * nkvh + head1) * hd + d]);
                        } else {
                            acc += attn_score[((head1 * ng + head2) * qlen +
                                               ql) *
                                                  kvlen +
                                              kvl] *
                                   v[(kvl * nkvh + head1) * hd + d];
                        }
                    }
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> ||
                                  std::is_same_v<T, llaisys::fp16_t>) {
                        attn_val[(ql * nh + (head1 * ng + head2)) * hd + d] =
                            llaisys::utils::cast<T>(acc);
                    } else {
                        attn_val[(ql * nh + (head1 * ng + head2)) * hd + d] =
                            acc;
                    }
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type, size_t qlen,
                    size_t kvlen, size_t nh, size_t nkvh, size_t hd,
                    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v), qlen, kvlen,
                               nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                               reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                               reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v),
                               qlen, kvlen, nh, nkvh, hd, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
