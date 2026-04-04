#include "qwen2_runtime_state.hpp"

namespace llaisys::models {

Qwen2PagedRuntimeState::Qwen2PagedRuntimeState(const LlaisysQwen2Meta &meta,
                                               llaisysDeviceType_t device_type,
                                               int device_id,
                                               size_t block_size) {
    const size_t num_kv_blocks = (meta.maxseq + block_size - 1) / block_size;
    _paged_k_cache.reserve(meta.nlayer);
    _paged_v_cache.reserve(meta.nlayer);
    for (size_t i = 0; i < meta.nlayer; ++i) {
        _paged_k_cache.push_back(Tensor::create({num_kv_blocks, block_size, meta.nkvh, meta.dh},
                                                meta.dtype, device_type, device_id));
        _paged_v_cache.push_back(Tensor::create({num_kv_blocks, block_size, meta.nkvh, meta.dh},
                                                meta.dtype, device_type, device_id));
    }
    _block_manager = std::make_shared<core::paged_kv::BlockManager>(num_kv_blocks, block_size);
}

std::shared_ptr<core::paged_kv::BlockManager> Qwen2PagedRuntimeState::blockManager() const {
    return _block_manager;
}

tensor_t Qwen2PagedRuntimeState::kCache(size_t layer) const {
    return _paged_k_cache.at(layer);
}

tensor_t Qwen2PagedRuntimeState::vCache(size_t layer) const {
    return _paged_v_cache.at(layer);
}

} // namespace llaisys::models
