#include "../src/core/paged_kv/paged_kv.hpp"
#include "../src/core/scheduler/scheduler.hpp"
#include "../src/models/qwen2/qwen2_batch_engine.hpp"
#include "../src/models/qwen2/qwen2.hpp"
#include "../src/models/qwen2/qwen2_session.hpp"
#include "../src/llaisys/llaisys_tensor.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using llaisys::core::paged_kv::BlockManager;
using llaisys::core::paged_kv::SequenceState;
using llaisys::core::scheduler::Scheduler;
using llaisys::models::Qwen2DynamicBatchEngine;
using llaisys::models::Qwen2Model;
using llaisys::models::Qwen2Session;

namespace {

void expect(bool cond, const std::string &message) {
    if (!cond) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void expect_invalid_argument_silent(Fn &&fn, const std::string &message) {
    std::ostringstream sink;
    auto *old_buf = std::cerr.rdbuf(sink.rdbuf());
    bool caught = false;
    try {
        fn();
    } catch (const std::invalid_argument &) {
        caught = true;
    }
    std::cerr.rdbuf(old_buf);
    expect(caught, message);
}

void load_tensor(llaisysTensor_t tensor, const std::vector<float> &values) {
    expect(tensor != nullptr, "load_tensor: null tensor");
    expect(tensor->tensor->dtype() == LLAISYS_DTYPE_F32, "load_tensor: expected f32 tensor");
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

void initialize_test_weights(Qwen2Model &model, size_t nlayer, size_t voc, size_t hs, size_t nkvh, size_t dh) {
    auto &weights = model.weights();

    std::vector<float> embed(voc * hs, 0.0f);
    for (size_t token = 0; token < voc; ++token) {
        for (size_t i = 0; i < hs; ++i) {
            embed[token * hs + i] = std::sin(static_cast<float>((token + 1) * (i + 1))) * 0.25f;
        }
    }
    load_tensor(weights.in_embed, embed);
    load_tensor(weights.out_embed, embed);
    fill_tensor(weights.out_norm_w, 1.0f);

    for (size_t layer = 0; layer < nlayer; ++layer) {
        fill_tensor(weights.attn_norm_w[layer], 1.0f);
        load_tensor(weights.attn_q_w[layer], identity_matrix(hs, hs));
        fill_tensor(weights.attn_q_b[layer], 0.0f);

        std::vector<float> k_w(nkvh * dh * hs, 0.0f);
        std::vector<float> v_w(nkvh * dh * hs, 0.0f);
        for (size_t row = 0; row < nkvh * dh; ++row) {
            k_w[row * hs + row] = 1.0f;
            v_w[row * hs + row] = 1.0f;
        }
        load_tensor(weights.attn_k_w[layer], k_w);
        load_tensor(weights.attn_v_w[layer], v_w);
        fill_tensor(weights.attn_k_b[layer], 0.0f);
        fill_tensor(weights.attn_v_b[layer], 0.0f);

        load_tensor(weights.attn_o_w[layer], identity_matrix(hs, hs));
        fill_tensor(weights.mlp_norm_w[layer], 1.0f);
        fill_tensor(weights.mlp_gate_w[layer], 0.0f);
        fill_tensor(weights.mlp_up_w[layer], 0.0f);
        fill_tensor(weights.mlp_down_w[layer], 0.0f);
    }
}

std::unique_ptr<Qwen2Model> make_test_model() {
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
    int device_ids[1] = {0};
    auto model = std::make_unique<Qwen2Model>(meta, LLAISYS_DEVICE_CPU, device_ids, 1);
    initialize_test_weights(*model, meta.nlayer, meta.voc, meta.hs, meta.nkvh, meta.dh);
    return model;
}

LlaisysQwen2SamplingParams deterministic_sampling() {
    LlaisysQwen2SamplingParams params{};
    params.temperature = 1.0f;
    params.top_k = 1;
    params.top_p = 0.0f;
    params.seed = 1;
    return params;
}

std::vector<int64_t> sequential_generate(const std::vector<int64_t> &prompt, size_t steps) {
    auto model = make_test_model();
    const auto params = deterministic_sampling();
    Qwen2Session session(*model);
    std::vector<int64_t> generated;
    auto current = prompt;
    for (size_t step = 0; step < steps; ++step) {
        int64_t next = session.infer(current, params);
        generated.push_back(next);
        current = {next};
    }
    return generated;
}

std::vector<int64_t> sequential_generate(const std::vector<int64_t> &prompt,
                                         size_t steps,
                                         const LlaisysQwen2SamplingParams &params) {
    auto model = make_test_model();
    Qwen2Session session(*model);
    std::vector<int64_t> generated;
    auto current = prompt;
    for (size_t step = 0; step < steps; ++step) {
        int64_t next = session.infer(current, params);
        generated.push_back(next);
        current = {next};
    }
    return generated;
}

void test_scheduler_model_end_to_end() {
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
    int device_ids[1] = {0};
    Qwen2DynamicBatchEngine engine(meta, LLAISYS_DEVICE_CPU, device_ids, 1, 2, 32);
    initialize_test_weights(engine.model(), meta.nlayer, meta.voc, meta.hs, meta.nkvh, meta.dh);
    const auto params = deterministic_sampling();

    const size_t seq0_id = engine.addRequest({1, 2, 3}, 2, params);
    const size_t seq1_id = engine.addRequest({4, 5}, 2, params);

    std::vector<int64_t> seq0_completion;
    std::vector<int64_t> seq1_completion;
    while (!engine.isFinished()) {
        auto finished = engine.step();
        for (const auto &entry : finished) {
            if (entry.seq_id == seq0_id) {
                seq0_completion = entry.completion_token_ids;
            } else if (entry.seq_id == seq1_id) {
                seq1_completion = entry.completion_token_ids;
            }
        }
    }

    const auto expected0 = sequential_generate({1, 2, 3}, 2);
    const auto expected1 = sequential_generate({4, 5}, 2);

    expect(seq0_completion == expected0,
           "dynamic batch output mismatch for sequence 0");
    expect(seq1_completion == expected1,
           "dynamic batch output mismatch for sequence 1");
}

void test_prefill_then_decode_priority_end_to_end() {
    auto model = make_test_model();
    auto block_manager = std::make_shared<BlockManager>(8, 16);
    Scheduler scheduler(2, 32, 63, block_manager);
    const auto params = deterministic_sampling();

    auto seq0 = scheduler.add(std::make_shared<SequenceState>(0, std::vector<int64_t>{1, 2, 3}, 16), 2);
    auto first = scheduler.schedule();
    expect(first.is_prefill, "first round should be prefill");
    llaisys::models::Qwen2PagedRuntimeState runtime_state(model->meta(), LLAISYS_DEVICE_CPU, 0, Qwen2Model::pagedBlockSize());
    auto out0 = model->runBatch(runtime_state, first.scheduled, first.is_prefill, {params});
    scheduler.postprocess(first.scheduled, out0);

    auto seq1 = scheduler.add(std::make_shared<SequenceState>(1, std::vector<int64_t>{4, 5, 6}, 16), 1);
    auto second = scheduler.schedule();
    expect(second.is_prefill, "new waiting prompt should force prefill before decode");
    expect(second.scheduled.size() == 1 && second.scheduled[0] == seq1,
           "second round should schedule only the new prefill request");

    auto out1 = model->runBatch(runtime_state, second.scheduled, second.is_prefill, {params});
    scheduler.postprocess(second.scheduled, out1);

    auto third = scheduler.schedule();
    expect(!third.is_prefill, "decode should happen only after waiting queue is empty");
    std::vector<LlaisysQwen2SamplingParams> third_params(third.scheduled.size(), params);
    auto out2 = model->runBatch(runtime_state, third.scheduled, third.is_prefill, third_params);
    scheduler.postprocess(third.scheduled, out2);

    expect(seq0->sequence->numCompletionTokens() >= 2 || seq0->isFinished(),
           "sequence 0 should continue decoding after prefill queue drains");
}

void test_dynamic_batch_supports_per_request_sampling_params() {
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
    int device_ids[1] = {0};
    Qwen2DynamicBatchEngine engine(meta, LLAISYS_DEVICE_CPU, device_ids, 1, 2, 32);
    initialize_test_weights(engine.model(), meta.nlayer, meta.voc, meta.hs, meta.nkvh, meta.dh);

    auto params0 = deterministic_sampling();
    auto params1 = deterministic_sampling();
    params1.seed = 17;
    params1.top_k = 4;
    params1.top_p = 0.85f;
    params1.temperature = 0.9f;

    const size_t seq0_id = engine.addRequest({1, 2, 3}, 2, params0);
    const size_t seq1_id = engine.addRequest({4, 5}, 2, params1);

    std::vector<int64_t> seq0_completion;
    std::vector<int64_t> seq1_completion;
    while (!engine.isFinished()) {
        auto finished = engine.step();
        for (const auto &entry : finished) {
            if (entry.seq_id == seq0_id) {
                seq0_completion = entry.completion_token_ids;
            } else if (entry.seq_id == seq1_id) {
                seq1_completion = entry.completion_token_ids;
            }
        }
    }

    expect(seq0_completion == sequential_generate({1, 2, 3}, 2, params0),
           "per-request sampling params mismatch for sequence 0");
    expect(seq1_completion == sequential_generate({4, 5}, 2, params1),
           "per-request sampling params mismatch for sequence 1");
}

void test_dynamic_batch_engine_rejects_invalid_limits_and_prompts() {
    LlaisysQwen2Meta meta{};
    meta.dtype = LLAISYS_DTYPE_F32;
    meta.nlayer = 1;
    meta.hs = 4;
    meta.nh = 2;
    meta.nkvh = 1;
    meta.dh = 2;
    meta.di = 4;
    meta.maxseq = 8;
    meta.voc = 32;
    meta.epsilon = 1e-6f;
    meta.theta = 10000.0f;
    meta.end_token = 31;
    int device_ids[1] = {0};

    expect_invalid_argument_silent(
        [&]() { Qwen2DynamicBatchEngine engine(meta, LLAISYS_DEVICE_CPU, device_ids, 1, 2, 16); },
        "engine should reject max_num_batched_tokens > meta.maxseq");

    Qwen2DynamicBatchEngine engine(meta, LLAISYS_DEVICE_CPU, device_ids, 1, 2, 8);
    initialize_test_weights(engine.model(), meta.nlayer, meta.voc, meta.hs, meta.nkvh, meta.dh);
    auto params = deterministic_sampling();
    expect_invalid_argument_silent(
        [&]() { engine.addRequest({}, 1, params); },
        "engine should reject empty prompt");
    expect_invalid_argument_silent(
        [&]() { engine.addRequest({1,2,3,4,5,6,7,8,9}, 1, params); },
        "engine should reject prompt longer than meta.maxseq");
}

} // namespace

int main() {
    test_scheduler_model_end_to_end();
    test_prefill_then_decode_priority_end_to_end();
    test_dynamic_batch_supports_per_request_sampling_params();
    test_dynamic_batch_engine_rejects_invalid_limits_and_prompts();
    std::cout << "dynamic_batch_test passed" << std::endl;
    return 0;
}
