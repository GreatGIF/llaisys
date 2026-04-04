#include "qwen2_batch_engine.hpp"

#include "../../utils.hpp"

namespace llaisys::models {

Qwen2DynamicBatchEngine::Qwen2DynamicBatchEngine(const LlaisysQwen2Meta &meta,
                                                 llaisysDeviceType_t device,
                                                 int *device_ids,
                                                 int ndevice,
                                                 size_t max_num_seqs,
                                                 size_t max_num_batched_tokens)
    : _model(meta, device, device_ids, ndevice),
      _runtime_state(meta, device, device_ids[0], Qwen2Model::pagedBlockSize()),
      _scheduler(max_num_seqs, max_num_batched_tokens, meta.end_token, _runtime_state.blockManager()) {
    CHECK_ARGUMENT(max_num_batched_tokens <= meta.maxseq,
                   "Qwen2DynamicBatchEngine: max_num_batched_tokens must be <= meta.maxseq");
}

LlaisysQwen2Weights &Qwen2DynamicBatchEngine::weights() {
    return _model.weights();
}

size_t Qwen2DynamicBatchEngine::addRequest(const std::vector<int64_t> &prompt_token_ids,
                                           size_t max_completion_tokens,
                                           const LlaisysQwen2SamplingParams &sampling_params,
                                           bool ignore_eos) {
    CHECK_ARGUMENT(!prompt_token_ids.empty(),
                   "Qwen2DynamicBatchEngine::addRequest: prompt_token_ids must not be empty");
    CHECK_ARGUMENT(prompt_token_ids.size() <= _model.meta().maxseq,
                   "Qwen2DynamicBatchEngine::addRequest: prompt length exceeds meta.maxseq");
    auto sequence = std::make_shared<core::paged_kv::SequenceState>(
        _next_seq_id++, prompt_token_ids, Qwen2Model::pagedBlockSize());
    const size_t seq_id = sequence->seqId();
    _scheduler.add(sequence, max_completion_tokens, ignore_eos);
    _sampling_params[seq_id] = sampling_params;
    return seq_id;
}

std::vector<Qwen2FinishedSequence> Qwen2DynamicBatchEngine::step() {
    if (_scheduler.isFinished()) {
        return {};
    }

    auto scheduled = _scheduler.schedule();
    std::vector<LlaisysQwen2SamplingParams> params;
    params.reserve(scheduled.scheduled.size());
    for (const auto &entry : scheduled.scheduled) {
        auto it = _sampling_params.find(entry->sequence->seqId());
        CHECK_ARGUMENT(it != _sampling_params.end(),
                       "Qwen2DynamicBatchEngine::step: missing sampling params for sequence");
        params.push_back(it->second);
    }

    const auto outputs = _model.runBatch(_runtime_state, scheduled.scheduled, scheduled.is_prefill, params);
    _scheduler.postprocess(scheduled.scheduled, outputs);

    std::vector<Qwen2FinishedSequence> finished;
    for (const auto &entry : scheduled.scheduled) {
        if (entry->isFinished()) {
            finished.push_back(Qwen2FinishedSequence{
                entry->sequence->seqId(),
                entry->sequence->completionTokenIds(),
            });
            _sampling_params.erase(entry->sequence->seqId());
        }
    }
    return finished;
}

bool Qwen2DynamicBatchEngine::isFinished() const {
    return _scheduler.isFinished();
}

Qwen2Model &Qwen2DynamicBatchEngine::model() {
    return _model;
}

} // namespace llaisys::models
