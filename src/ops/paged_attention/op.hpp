#pragma once

#include "../../core/paged_kv/paged_kv.hpp"
#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

void store_paged_kv_cache(tensor_t k_cache, tensor_t v_cache, tensor_t k, tensor_t v,
                          const std::vector<int32_t> &slot_mapping);

void paged_attention_prefill(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
                             const core::paged_kv::PrefillBatch &batch,
                             size_t num_kv_heads, float scale);

void paged_attention_decode(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
                            const core::paged_kv::DecodeBatch &batch,
                            size_t num_kv_heads, float scale);

} // namespace llaisys::ops
