#include "../../models/qwen2/qwen2.hpp"
#include "../../models/qwen2/qwen2_batch_engine.hpp"
#include "../../models/qwen2/qwen2_session.hpp"
#include "llaisys/models/qwen2.h"

#include <algorithm>
#include <vector>

using namespace llaisys::models;

__C {
    struct LlaisysQwen2Model {
        Qwen2Model *impl;
        Qwen2Session *compat_session;
    };

    struct LlaisysQwen2Session {
        Qwen2Session *impl;
    };

    struct LlaisysQwen2DynamicBatchEngine {
        Qwen2DynamicBatchEngine *impl;
    };

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        auto *impl = new Qwen2Model(*meta, device, device_ids, ndevice);
        auto *session = new Qwen2Session(*impl, 0);
        return new LlaisysQwen2Model{impl, session};
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete model->compat_session;
        delete model->impl;
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        return &model->impl->weights();
    }

    void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model) {
        model->compat_session->reset();
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken,
                                      const struct LlaisysQwen2SamplingParams *sampling_params) {
        // Create default sampling params if not provided
        LlaisysQwen2SamplingParams params = sampling_params ? *sampling_params : LlaisysQwen2SamplingParams{1.0f, 0, 0.0f, 0};
        return model->compat_session->infer(std::vector<int64_t>(token_ids, token_ids + ntoken), params);
    }

    struct LlaisysQwen2Session *llaisysQwen2SessionCreate(struct LlaisysQwen2Model *model, size_t seq_id) {
        return new LlaisysQwen2Session{new Qwen2Session(*model->impl, seq_id)};
    }

    void llaisysQwen2SessionDestroy(struct LlaisysQwen2Session *session) {
        delete session->impl;
        delete session;
    }

    void llaisysQwen2SessionReset(struct LlaisysQwen2Session *session) {
        session->impl->reset();
    }

    int64_t llaisysQwen2SessionInfer(struct LlaisysQwen2Session *session, int64_t *token_ids, size_t ntoken,
                                     const struct LlaisysQwen2SamplingParams *sampling_params) {
        LlaisysQwen2SamplingParams params = sampling_params ? *sampling_params : LlaisysQwen2SamplingParams{1.0f, 0, 0.0f, 0};
        return session->impl->infer(std::vector<int64_t>(token_ids, token_ids + ntoken), params);
    }

    struct LlaisysQwen2DynamicBatchEngine *llaisysQwen2DynamicBatchEngineCreate(
        const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice,
        size_t max_num_seqs, size_t max_num_batched_tokens) {
        return new LlaisysQwen2DynamicBatchEngine{
            new Qwen2DynamicBatchEngine(*meta, device, device_ids, ndevice, max_num_seqs, max_num_batched_tokens)
        };
    }

    void llaisysQwen2DynamicBatchEngineDestroy(struct LlaisysQwen2DynamicBatchEngine *engine) {
        delete engine->impl;
        delete engine;
    }

    struct LlaisysQwen2Weights *llaisysQwen2DynamicBatchEngineWeights(struct LlaisysQwen2DynamicBatchEngine *engine) {
        return &engine->impl->weights();
    }

    size_t llaisysQwen2DynamicBatchEngineAddRequest(
        struct LlaisysQwen2DynamicBatchEngine *engine, int64_t *token_ids, size_t ntoken,
        size_t max_completion_tokens, const struct LlaisysQwen2SamplingParams *sampling_params, uint8_t ignore_eos) {
        LlaisysQwen2SamplingParams params = sampling_params ? *sampling_params : LlaisysQwen2SamplingParams{1.0f, 0, 0.0f, 0};
        return engine->impl->addRequest(std::vector<int64_t>(token_ids, token_ids + ntoken),
                                        max_completion_tokens, params, static_cast<bool>(ignore_eos));
    }

    struct LlaisysQwen2StepResult *llaisysQwen2DynamicBatchEngineStep(struct LlaisysQwen2DynamicBatchEngine *engine) {
        auto finished = engine->impl->step();
        auto *result = new LlaisysQwen2StepResult{};
        result->nsequence = finished.size();
        result->sequences = finished.empty() ? nullptr : new LlaisysQwen2FinishedSequence[finished.size()];
        for (size_t i = 0; i < finished.size(); ++i) {
            result->sequences[i].seq_id = finished[i].seq_id;
            result->sequences[i].ntoken = finished[i].completion_token_ids.size();
            if (finished[i].completion_token_ids.empty()) {
                result->sequences[i].token_ids = nullptr;
            } else {
                result->sequences[i].token_ids = new int64_t[finished[i].completion_token_ids.size()];
                std::copy(finished[i].completion_token_ids.begin(), finished[i].completion_token_ids.end(),
                          result->sequences[i].token_ids);
            }
        }
        return result;
    }

    void llaisysQwen2StepResultDestroy(struct LlaisysQwen2StepResult *result) {
        if (result == nullptr) {
            return;
        }
        for (size_t i = 0; i < result->nsequence; ++i) {
            delete[] result->sequences[i].token_ids;
        }
        delete[] result->sequences;
        delete result;
    }

    uint8_t llaisysQwen2DynamicBatchEngineIsFinished(struct LlaisysQwen2DynamicBatchEngine *engine) {
        return static_cast<uint8_t>(engine->impl->isFinished());
    }
}
