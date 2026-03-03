#include "op.hpp"
#include <cmath>

template <typename T>
void self_attention_(T* attn_val, const T* q, const T* k, const T* v, const std::vector<size_t> &shape, float scale) {
    size_t qlen = shape[0];
    size_t kvlen = shape[1];
    size_t nh = shape[2];   // num_q_heads
    size_t nkvh = shape[3]; // num_kv_heads
    size_t hd = shape[4];   // head_dim
    size_t ng = nh / nkvh;  // groups
    std::vector<float> attn_score(nh*qlen*kvlen, 0); // [head, ql, kvl]

#ifdef ENABLE_OPENMP
    #pragma omp parallel for collapse(4) schedule(static) if (nkvh * ng * qlen * kvlen >= 256)
#endif
    // Q * K^T [head, ql, kvl]
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
                float shifted = attn_score[(head * qlen + ql) * kvlen + kvl] - max_val;
                float e = std::exp(shifted);
                attn_score[(head * qlen + ql) * kvlen + kvl] = e;
                sum += e;
            }
            // Normalize
            for (size_t kvl = 0; kvl < kvlen; kvl++) {
                attn_score[(head * qlen + ql) * kvlen + kvl] /= sum;                       // [head, ql, kvl]
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
                        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                            acc += attn_score[((head1 * ng + head2) * qlen + ql) * kvlen + kvl] * llaisys::utils::cast<float>(v[(kvl * nkvh + head1) * hd  + d]); 
                        } else {
                            acc += attn_score[((head1 * ng + head2) * qlen + ql) * kvlen + kvl] * v[(kvl * nkvh + head1) * hd  + d];
                        }
                    }
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        attn_val[(ql * nh + (head1 * ng + head2)) * hd + d]  = llaisys::utils::cast<T>(acc);
                    } else {
                        attn_val[(ql * nh + (head1 * ng + head2)) * hd + d]  = acc;
                    }
                }
            }
        }
    }
}

// template <typename T>
// void self_attention_(T* attn_val, const T* q, const T* k, const T* v, const std::vector<size_t> &shape, float scale) {
//     size_t qlen = shape[0];
//     size_t kvlen = shape[1];
//     size_t nh = shape[2];   // num_q_heads
//     size_t nkvh = shape[3]; // num_kv_heads
//     size_t hd = shape[4];   // head_dim
//     size_t ng = nh / nkvh;  // groups

//     // 建议布局: [nh, qlen, kvlen] 以便 softmax 和 reduction
//     std::vector<float> attn_score(nh * qlen * kvlen, 0);

//     // -----------------------------------------------------------
//     // 1. Q * K^T
//     // -----------------------------------------------------------
//     // 优化：将 Head 循环外提，减少 attn_score 的 stride 跳跃，提高缓存命中率
//     // 注意：输入 Q/K 布局假设为 [Seq, Head, Dim]
//     for (size_t head = 0; head < nh; head++) {
//         size_t kv_head = head / ng; // GQA: 映射 Q head 到 KV head
        
//         for (size_t ql = 0; ql < qlen; ql++) {
//             for (size_t kvl = 0; kvl < kvlen; kvl++) {
                
//                 float score = 0.0f;
//                 for (size_t d = 0; d < hd; d++) {
//                     // Q layout: [ql, head, d]
//                     size_t idx_q = ql * nh * hd + head * hd + d; 
//                     // K layout: [kvl, kv_head, d]
//                     size_t idx_k = kvl * nkvh * hd + kv_head * hd + d;

//                     float q_val, k_val;
//                     if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
//                         q_val = llaisys::utils::cast<float>(q[idx_q]);
//                         k_val = llaisys::utils::cast<float>(k[idx_k]);
//                     } else {
//                         q_val = q[idx_q];
//                         k_val = k[idx_k];
//                     }
//                     score += q_val * k_val;
//                 }
//                 score *= scale;

//                 // Causal Masking (Mask logic match with PyTorch tril)
//                 // PyTorch: mask if kvl > ql + (S-L). 
//                 // C++: if kvlen > qlen, history exists. 
//                 if (kvl > ql + (kvlen - qlen)) {
//                     score = -std::numeric_limits<float>::infinity();
//                 }

//                 // attn_score layout: [head, ql, kvl]
//                 attn_score[(head * qlen + ql) * kvlen + kvl] = score;
//             }
//         }
//     }

//     // -----------------------------------------------------------
//     // 2. Softmax (per row: head, ql)
//     // -----------------------------------------------------------
//     for (size_t head = 0; head < nh; head++) {
//         for (size_t ql = 0; ql < qlen; ql++) {
//             float max_val = -std::numeric_limits<float>::infinity();
//             size_t row_offset = (head * qlen + ql) * kvlen;

//             // Find Max
//             for (size_t kvl = 0; kvl < kvlen; kvl++) {
//                 max_val = std::max(max_val, attn_score[row_offset + kvl]);
//             }

//             // Exp & Sum
//             float sum = 0.0f;
//             for (size_t kvl = 0; kvl < kvlen; kvl++) {
//                 float val = attn_score[row_offset + kvl];
//                 // Handle -inf case to avoid NaN if necessary, though std::exp(-inf) is 0
//                 float e = std::exp(val - max_val);
//                 attn_score[row_offset + kvl] = e;
//                 sum += e;
//             }

//             // Normalize
//             for (size_t kvl = 0; kvl < kvlen; kvl++) {
//                 attn_score[row_offset + kvl] /= sum;
//             }
//         }
//     }

//     // -----------------------------------------------------------
//     // 3. Score * V
//     // -----------------------------------------------------------
//     for (size_t head = 0; head < nh; head++) {
//         size_t kv_head = head / ng; // [FIX 1]: GQA Index mapping
        
//         for (size_t ql = 0; ql < qlen; ql++) {
//             for (size_t d = 0; d < hd; d++) {
//                 float acc = 0.0f;
                
//                 for (size_t kvl = 0; kvl < kvlen; kvl++) {
//                     float weight = attn_score[(head * qlen + ql) * kvlen + kvl];
                    
//                     // [FIX 1]: 使用 kv_head 和 nkvh 访问 V
//                     // V layout: [kvl, kv_head, d]
//                     size_t idx_v = kvl * nkvh * hd + kv_head * hd + d;
                    
//                     float v_val;
//                     if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
//                          v_val = llaisys::utils::cast<float>(v[idx_v]);
//                     } else {
//                          v_val = v[idx_v];
//                     }
                    
//                     acc += weight * v_val;
//                 }

//                 // [FIX 2]: Output Layout 必须是 [ql, head, d] (Sequence, Head, Dim)
//                 // 以匹配输入 Q 的布局
//                 size_t idx_out = ql * nh * hd + head * hd + d;

//                 if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
//                     attn_val[idx_out] = llaisys::utils::cast<T>(acc);
//                 } else {
//                     attn_val[idx_out] = acc;
//                 }
//             }
//         }
//     }
// }

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // TO_BE_IMPLEMENTED();
    if (q->dtype() != k->dtype() || q->dtype() != v->dtype()) {
        throw std::runtime_error("dtype of q, k, v must be same");
    }

    std::vector<size_t> shape = {q->shape()[0], k->shape()[0], q->shape()[1], k->shape()[1], q->shape()[2]};
    llaisysDataType_t type = attn_val->dtype();
    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_(reinterpret_cast<float *>(attn_val->data()), reinterpret_cast<const float *>(q->data()), 
        reinterpret_cast<const float *>(k->data()), reinterpret_cast<const float *>(v->data()), shape, scale);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_(reinterpret_cast<bf16_t *>(attn_val->data()), reinterpret_cast<const bf16_t *>(q->data()), 
        reinterpret_cast<const bf16_t *>(k->data()), reinterpret_cast<const bf16_t *>(v->data()), shape, scale);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_(reinterpret_cast<fp16_t *>(attn_val->data()), reinterpret_cast<const fp16_t *>(q->data()), 
        reinterpret_cast<const fp16_t *>(k->data()), reinterpret_cast<const fp16_t *>(v->data()), shape, scale);
        break;
    default:
        throw std::invalid_argument("self_attention: unsupported dtype.");
    }
}
} // namespace llaisys::ops
