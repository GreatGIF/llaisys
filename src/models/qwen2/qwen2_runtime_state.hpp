#pragma once

#include "../../core/paged_kv/paged_kv.hpp"
#include "../../tensor/tensor.hpp"
#include "llaisys/models/qwen2.h"

#include <memory>
#include <vector>

namespace llaisys::models {

class Qwen2PagedRuntimeState {
public:
    Qwen2PagedRuntimeState(const LlaisysQwen2Meta &meta,
                           llaisysDeviceType_t device_type,
                           int device_id,
                           size_t block_size);

    std::shared_ptr<core::paged_kv::BlockManager> blockManager() const;
    tensor_t kCache(size_t layer) const;
    tensor_t vCache(size_t layer) const;

private:
    std::vector<tensor_t> _paged_k_cache;
    std::vector<tensor_t> _paged_v_cache;
    std::shared_ptr<core::paged_kv::BlockManager> _block_manager;
};

} // namespace llaisys::models
