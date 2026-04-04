#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    // Sampling parameters for token generation
    struct LlaisysQwen2SamplingParams {
        float temperature;  // Temperature for softmax (>0, lower = more deterministic)
        int32_t top_k;      // Keep only top K tokens (0 = disable)
        float top_p;        // Nucleus sampling threshold (0-1, 0 = disable)
        uint64_t seed;      // Random seed for reproducibility (0 = non-deterministic)
    };

    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;
    struct LlaisysQwen2Session;
    struct LlaisysQwen2DynamicBatchEngine;

    struct LlaisysQwen2FinishedSequence {
        size_t seq_id;
        int64_t *token_ids;
        size_t ntoken;
    };

    struct LlaisysQwen2StepEvent {
        size_t seq_id;
        int64_t token_id;
        uint8_t is_finished;
        int64_t *completion_token_ids;
        size_t ncompletion_token;
    };

    struct LlaisysQwen2StepResult {
        struct LlaisysQwen2FinishedSequence *sequences;
        size_t nsequence;
    };

    struct LlaisysQwen2StepEventsResult {
        struct LlaisysQwen2StepEvent *events;
        size_t nevent;
    };

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    __export void llaisysQwen2ModelReset(struct LlaisysQwen2Model * model);

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken,
                                            const struct LlaisysQwen2SamplingParams *sampling_params);

    __export struct LlaisysQwen2Session *llaisysQwen2SessionCreate(struct LlaisysQwen2Model *model, size_t seq_id);
    __export void llaisysQwen2SessionDestroy(struct LlaisysQwen2Session *session);
    __export void llaisysQwen2SessionReset(struct LlaisysQwen2Session *session);
    __export int64_t llaisysQwen2SessionInfer(struct LlaisysQwen2Session *session, int64_t *token_ids, size_t ntoken,
                                              const struct LlaisysQwen2SamplingParams *sampling_params);

    __export struct LlaisysQwen2DynamicBatchEngine *llaisysQwen2DynamicBatchEngineCreate(
        const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice,
        size_t max_num_seqs, size_t max_num_batched_tokens);
    __export void llaisysQwen2DynamicBatchEngineDestroy(struct LlaisysQwen2DynamicBatchEngine *engine);
    __export struct LlaisysQwen2Weights *llaisysQwen2DynamicBatchEngineWeights(struct LlaisysQwen2DynamicBatchEngine *engine);
    __export size_t llaisysQwen2DynamicBatchEngineAddRequest(
        struct LlaisysQwen2DynamicBatchEngine *engine, int64_t *token_ids, size_t ntoken,
        size_t max_completion_tokens, const struct LlaisysQwen2SamplingParams *sampling_params, uint8_t ignore_eos);
    __export struct LlaisysQwen2StepResult *llaisysQwen2DynamicBatchEngineStep(struct LlaisysQwen2DynamicBatchEngine *engine);
    __export void llaisysQwen2StepResultDestroy(struct LlaisysQwen2StepResult *result);
    __export struct LlaisysQwen2StepEventsResult *llaisysQwen2DynamicBatchEngineStepEvents(struct LlaisysQwen2DynamicBatchEngine *engine);
    __export void llaisysQwen2StepEventsResultDestroy(struct LlaisysQwen2StepEventsResult *result);
    __export uint8_t llaisysQwen2DynamicBatchEngineIsFinished(struct LlaisysQwen2DynamicBatchEngine *engine);
}
#endif // LLAISYS_MODELS_QWEN2_H
