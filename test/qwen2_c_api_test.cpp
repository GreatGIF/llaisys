#include "llaisys/models/qwen2.h"
#include "../src/llaisys/llaisys_tensor.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool cond, const std::string &message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

void load_tensor(llaisysTensor_t tensor, const std::vector<float> &values) {
    expect(tensor != nullptr, "load_tensor: null tensor");
    expect(tensor->tensor->numel() == values.size(), "load_tensor: size mismatch");
    tensor->tensor->load(values.data());
}

void fill_tensor(llaisysTensor_t tensor, float value) {
    std::vector<float> values(tensor->tensor->numel(), value);
    load_tensor(tensor, values);
}

std::vector<float> identity_matrix(size_t rows, size_t cols) {
    std::vector<float> values(rows * cols, 0.0f);
    const size_t diag = std::min(rows, cols);
    for (size_t i = 0; i < diag; ++i) {
        values[i * cols + i] = 1.0f;
    }
    return values;
}

void initialize_weights(LlaisysQwen2Weights *weights, size_t nlayer, size_t voc, size_t hs, size_t nkvh, size_t dh) {
    std::vector<float> embed(voc * hs, 0.0f);
    for (size_t token = 0; token < voc; ++token) {
        for (size_t i = 0; i < hs; ++i) {
            embed[token * hs + i] = std::sin(static_cast<float>((token + 1) * (i + 1))) * 0.25f;
        }
    }
    load_tensor(weights->in_embed, embed);
    load_tensor(weights->out_embed, embed);
    fill_tensor(weights->out_norm_w, 1.0f);
    for (size_t layer = 0; layer < nlayer; ++layer) {
        fill_tensor(weights->attn_norm_w[layer], 1.0f);
        load_tensor(weights->attn_q_w[layer], identity_matrix(hs, hs));
        fill_tensor(weights->attn_q_b[layer], 0.0f);

        std::vector<float> k_w(nkvh * dh * hs, 0.0f);
        std::vector<float> v_w(nkvh * dh * hs, 0.0f);
        for (size_t row = 0; row < nkvh * dh; ++row) {
            k_w[row * hs + row] = 1.0f;
            v_w[row * hs + row] = 1.0f;
        }
        load_tensor(weights->attn_k_w[layer], k_w);
        load_tensor(weights->attn_v_w[layer], v_w);
        fill_tensor(weights->attn_k_b[layer], 0.0f);
        fill_tensor(weights->attn_v_b[layer], 0.0f);
        load_tensor(weights->attn_o_w[layer], identity_matrix(hs, hs));
        fill_tensor(weights->mlp_norm_w[layer], 1.0f);
        fill_tensor(weights->mlp_gate_w[layer], 0.0f);
        fill_tensor(weights->mlp_up_w[layer], 0.0f);
        fill_tensor(weights->mlp_down_w[layer], 0.0f);
    }
}

LlaisysQwen2Meta make_meta() {
    LlaisysQwen2Meta meta{};
    meta.dtype = LLAISYS_DTYPE_F32;
    meta.nlayer = 1;
    meta.hs = 4;
    meta.nh = 2;
    meta.nkvh = 1;
    meta.dh = 2;
    meta.di = 4;
    meta.maxseq = 64;
    meta.voc = 64;
    meta.epsilon = 1e-6f;
    meta.theta = 10000.0f;
    meta.end_token = 63;
    return meta;
}

LlaisysQwen2SamplingParams make_params() {
    LlaisysQwen2SamplingParams params{};
    params.temperature = 1.0f;
    params.top_k = 1;
    params.top_p = 0.0f;
    params.seed = 1;
    return params;
}

void test_c_api_session() {
    auto meta = make_meta();
    int device_ids[1] = {0};
    auto *model = llaisysQwen2ModelCreate(&meta, LLAISYS_DEVICE_CPU, device_ids, 1);
    initialize_weights(llaisysQwen2ModelWeights(model), meta.nlayer, meta.voc, meta.hs, meta.nkvh, meta.dh);
    auto *session = llaisysQwen2SessionCreate(model, 1);
    auto params = make_params();
    int64_t prompt[] = {1, 2, 3};
    const int64_t token = llaisysQwen2SessionInfer(session, prompt, 3, &params);
    expect(token >= 0, "C API session infer should produce a token");
    llaisysQwen2SessionDestroy(session);
    llaisysQwen2ModelDestroy(model);
}

void test_c_api_dynamic_batch_engine() {
    auto meta = make_meta();
    int device_ids[1] = {0};
    auto *engine = llaisysQwen2DynamicBatchEngineCreate(&meta, LLAISYS_DEVICE_CPU, device_ids, 1, 2, 32);
    initialize_weights(llaisysQwen2DynamicBatchEngineWeights(engine), meta.nlayer, meta.voc, meta.hs, meta.nkvh, meta.dh);
    auto params = make_params();
    int64_t prompt0[] = {1, 2, 3};
    int64_t prompt1[] = {4, 5};
    const size_t seq0 = llaisysQwen2DynamicBatchEngineAddRequest(engine, prompt0, 3, 2, &params, 0);
    const size_t seq1 = llaisysQwen2DynamicBatchEngineAddRequest(engine, prompt1, 2, 2, &params, 0);
    bool seen0 = false;
    bool seen1 = false;
    while (!llaisysQwen2DynamicBatchEngineIsFinished(engine)) {
        auto *result = llaisysQwen2DynamicBatchEngineStep(engine);
        for (size_t i = 0; i < result->nsequence; ++i) {
            if (result->sequences[i].seq_id == seq0) {
                seen0 = true;
            }
            if (result->sequences[i].seq_id == seq1) {
                seen1 = true;
            }
        }
        llaisysQwen2StepResultDestroy(result);
    }
    expect(seen0 && seen1, "C API dynamic batch engine should finish all requests");
    llaisysQwen2DynamicBatchEngineDestroy(engine);
}

} // namespace

int main() {
    test_c_api_session();
    test_c_api_dynamic_batch_engine();
    std::cout << "qwen2_c_api_test passed" << std::endl;
    return 0;
}
