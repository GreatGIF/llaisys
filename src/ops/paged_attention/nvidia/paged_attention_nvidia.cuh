#pragma once

#include "../../../core/paged_kv/paged_kv.hpp"
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::nvidia {

void store_paged_kv_cache(std::byte *k_cache, std::byte *v_cache,
                          const std::byte *k, const std::byte *v,
                          llaisysDataType_t type,
                          size_t token_count, size_t num_kv_heads, size_t head_dim,
                          const std::vector<int32_t> &slot_mapping);

void paged_attention_prefill(std::byte *attn_val, const std::byte *q,
                             const std::byte *k_cache, const std::byte *v_cache,
                             llaisysDataType_t type,
                             const core::paged_kv::PrefillBatch &batch,
                             size_t num_heads, size_t num_kv_heads, size_t head_dim,
                             float scale, size_t block_size);

void paged_attention_decode(std::byte *attn_val, const std::byte *q,
                            const std::byte *k_cache, const std::byte *v_cache,
                            llaisysDataType_t type,
                            const core::paged_kv::DecodeBatch &batch,
                            size_t num_heads, size_t num_kv_heads, size_t head_dim,
                            float scale, size_t block_size);

} // namespace llaisys::ops::nvidia
