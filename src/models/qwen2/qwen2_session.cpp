#include "qwen2_session.hpp"

#include "../../utils.hpp"

namespace llaisys::models {

Qwen2Session::Qwen2Session(Qwen2Model &model, size_t seq_id)
    : _model(model),
      _runtime_state(model.meta(), model.deviceType(), model.deviceId(), Qwen2Model::pagedBlockSize()),
      _seq_id(seq_id) {
}

void Qwen2Session::reset() {
    if (_sequence && !_sequence->blockTable().empty()) {
        _runtime_state.blockManager()->deallocate(*_sequence);
    }
    _sequence.reset();
}

int64_t Qwen2Session::infer(const std::vector<int64_t> &token_ids, const LlaisysQwen2SamplingParams &params) {
    CHECK_ARGUMENT(!token_ids.empty(), "Qwen2Session::infer: token_ids must not be empty");

    size_t old_num_tokens = 0;
    size_t old_num_cached_tokens = 0;
    if (!_sequence) {
        _sequence = std::make_unique<core::paged_kv::SequenceState>(_seq_id, token_ids, Qwen2Model::pagedBlockSize());
        _runtime_state.blockManager()->allocate(*_sequence);
    } else {
        old_num_tokens = _sequence->numTokens();
        old_num_cached_tokens = _sequence->numCachedTokens();
        for (int64_t token : token_ids) {
            _sequence->appendToken(token);
            _runtime_state.blockManager()->prepareAppend(*_sequence);
        }
    }

    const bool use_decode = (token_ids.size() == 1 && old_num_tokens > 0 && old_num_cached_tokens == old_num_tokens);
    return _model.runSequence(_runtime_state, *_sequence, !use_decode, params);
}

const core::paged_kv::SequenceState *Qwen2Session::sequence() const {
    return _sequence.get();
}

Qwen2PagedRuntimeState &Qwen2Session::runtimeState() {
    return _runtime_state;
}

} // namespace llaisys::models
