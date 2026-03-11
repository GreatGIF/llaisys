#include "../../models/qwen2/qwen2.hpp"
#include "llaisys/models/qwen2.h"

using namespace llaisys::models;

__C {
    struct LlaisysQwen2Model {
        Qwen2Model *impl;
    };

    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        return new LlaisysQwen2Model{new Qwen2Model(*meta, device, device_ids, ndevice)};
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        delete model->impl;
        delete model;
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        return &model->impl->weights();
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        return model->impl->infer(token_ids, ntoken);
    }
}
