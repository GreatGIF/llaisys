#pragma once

#include "qwen2.hpp"
#include "qwen2_runtime_state.hpp"

#include <memory>
#include <vector>

namespace llaisys::models {

class Qwen2Session {
public:
    Qwen2Session(Qwen2Model &model, size_t seq_id = 0);

    void reset();
    int64_t infer(const std::vector<int64_t> &token_ids, const LlaisysQwen2SamplingParams &params);

    const core::paged_kv::SequenceState *sequence() const;
    Qwen2PagedRuntimeState &runtimeState();

private:
    Qwen2Model &_model;
    Qwen2PagedRuntimeState _runtime_state;
    std::unique_ptr<core::paged_kv::SequenceState> _sequence;
    size_t _seq_id;
};

} // namespace llaisys::models
