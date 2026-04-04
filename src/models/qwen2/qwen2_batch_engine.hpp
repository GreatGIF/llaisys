#pragma once

#include "qwen2.hpp"
#include "../../core/scheduler/scheduler.hpp"

#include <memory>
#include <unordered_map>
#include <vector>

namespace llaisys::models {

struct Qwen2FinishedSequence {
    size_t seq_id = 0;
    std::vector<int64_t> completion_token_ids;
};

struct Qwen2StepEvent {
    size_t seq_id = 0;
    int64_t token_id = 0;
    bool is_finished = false;
    std::vector<int64_t> completion_token_ids;
};

class Qwen2DynamicBatchEngine {
public:
    Qwen2DynamicBatchEngine(const LlaisysQwen2Meta &meta,
                            llaisysDeviceType_t device,
                            int *device_ids,
                            int ndevice,
                            size_t max_num_seqs,
                            size_t max_num_batched_tokens);

    LlaisysQwen2Weights &weights();
    size_t addRequest(const std::vector<int64_t> &prompt_token_ids,
                      size_t max_completion_tokens,
                      const LlaisysQwen2SamplingParams &sampling_params,
                      bool ignore_eos = false);
    std::vector<Qwen2StepEvent> stepEvents();
    std::vector<Qwen2FinishedSequence> step();
    bool isFinished() const;
    Qwen2Model &model();

private:
    Qwen2Model _model;
    Qwen2PagedRuntimeState _runtime_state;
    core::scheduler::Scheduler _scheduler;
    std::unordered_map<size_t, LlaisysQwen2SamplingParams> _sampling_params;
    size_t _next_seq_id = 0;
};

} // namespace llaisys::models
