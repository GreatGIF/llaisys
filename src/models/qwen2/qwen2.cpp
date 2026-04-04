#include "qwen2.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/paged_attention/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/sampling/op.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include "../../device/runtime_api.hpp"
#include "../../utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
namespace llaisys::models {

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
    : _meta(meta), _device_type(device), _device_id(device_ids[0]) {

    // 创建Tensor类, 返回Tensor指针
    auto create_weight = [&](const std::vector<size_t> &shape) {
        return Tensor::create(shape, _meta.dtype, _device_type, _device_id);
    };

    // 将Tensor转换为C-style结构体
    auto to_c = [](tensor_t t) {
        return new LlaisysTensor{t};
    };

    // Allocate global weights
    _in_embed = create_weight({_meta.voc, _meta.hs});
    _out_embed = create_weight({_meta.voc, _meta.hs});
    _out_norm_w = create_weight({_meta.hs});

    // 将权重从Tensor转换为C-style数组, 并绑定到_weights_c
    _weights_c.in_embed = to_c(_in_embed);
    _weights_c.out_embed = to_c(_out_embed);
    _weights_c.out_norm_w = to_c(_out_norm_w);

    // Allocate layer weights
    _attn_norm_w.resize(_meta.nlayer);
    _attn_q_w.resize(_meta.nlayer); _attn_q_b.resize(_meta.nlayer);
    _attn_k_w.resize(_meta.nlayer); _attn_k_b.resize(_meta.nlayer);
    _attn_v_w.resize(_meta.nlayer); _attn_v_b.resize(_meta.nlayer);
    _attn_o_w.resize(_meta.nlayer);
    _mlp_norm_w.resize(_meta.nlayer);
    _mlp_gate_w.resize(_meta.nlayer); _mlp_up_w.resize(_meta.nlayer); _mlp_down_w.resize(_meta.nlayer);

    _c_attn_norm_w.resize(_meta.nlayer);
    _c_attn_q_w.resize(_meta.nlayer); _c_attn_q_b.resize(_meta.nlayer);
    _c_attn_k_w.resize(_meta.nlayer); _c_attn_k_b.resize(_meta.nlayer);
    _c_attn_v_w.resize(_meta.nlayer); _c_attn_v_b.resize(_meta.nlayer);
    _c_attn_o_w.resize(_meta.nlayer);
    _c_mlp_norm_w.resize(_meta.nlayer);
    _c_mlp_gate_w.resize(_meta.nlayer); _c_mlp_up_w.resize(_meta.nlayer); _c_mlp_down_w.resize(_meta.nlayer);

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _attn_norm_w[i] = create_weight({_meta.hs});
        _attn_q_w[i] = create_weight({_meta.nh * _meta.dh, _meta.hs});
        _attn_q_b[i] = create_weight({_meta.nh * _meta.dh});
        _attn_k_w[i] = create_weight({_meta.nkvh * _meta.dh, _meta.hs});
        _attn_k_b[i] = create_weight({_meta.nkvh * _meta.dh});
        _attn_v_w[i] = create_weight({_meta.nkvh * _meta.dh, _meta.hs});
        _attn_v_b[i] = create_weight({_meta.nkvh * _meta.dh});
        _attn_o_w[i] = create_weight({_meta.hs, _meta.nh * _meta.dh});
        _mlp_norm_w[i] = create_weight({_meta.hs});
        _mlp_gate_w[i] = create_weight({_meta.di, _meta.hs});
        _mlp_up_w[i] = create_weight({_meta.di, _meta.hs});
        _mlp_down_w[i] = create_weight({_meta.hs, _meta.di});

        _c_attn_norm_w[i] = to_c(_attn_norm_w[i]);
        _c_attn_q_w[i] = to_c(_attn_q_w[i]); _c_attn_q_b[i] = to_c(_attn_q_b[i]);
        _c_attn_k_w[i] = to_c(_attn_k_w[i]); _c_attn_k_b[i] = to_c(_attn_k_b[i]);
        _c_attn_v_w[i] = to_c(_attn_v_w[i]); _c_attn_v_b[i] = to_c(_attn_v_b[i]);
        _c_attn_o_w[i] = to_c(_attn_o_w[i]);
        _c_mlp_norm_w[i] = to_c(_mlp_norm_w[i]);
        _c_mlp_gate_w[i] = to_c(_mlp_gate_w[i]);
        _c_mlp_up_w[i] = to_c(_mlp_up_w[i]);
        _c_mlp_down_w[i] = to_c(_mlp_down_w[i]);
    }
    
    // 将权重从Tensor转换为C-style数组, 并绑定到_weights_c
    _weights_c.attn_norm_w = _c_attn_norm_w.data();
    _weights_c.attn_q_w = _c_attn_q_w.data(); _weights_c.attn_q_b = _c_attn_q_b.data();
    _weights_c.attn_k_w = _c_attn_k_w.data(); _weights_c.attn_k_b = _c_attn_k_b.data();
    _weights_c.attn_v_w = _c_attn_v_w.data(); _weights_c.attn_v_b = _c_attn_v_b.data();
    _weights_c.attn_o_w = _c_attn_o_w.data();
    _weights_c.mlp_norm_w = _c_mlp_norm_w.data();
    _weights_c.mlp_gate_w = _c_mlp_gate_w.data();
    _weights_c.mlp_up_w = _c_mlp_up_w.data();
    _weights_c.mlp_down_w = _c_mlp_down_w.data();
    // 预分配中间Tensor变量, 通过复用以减少重复创建Tensor
    _x = create_weight({_meta.maxseq, _meta.hs});
    _x_norm = create_weight({_meta.maxseq, _meta.hs});
    _q = create_weight({_meta.maxseq, _meta.nh, _meta.dh});
    _k = create_weight({_meta.maxseq, _meta.nkvh, _meta.dh});
    _v = create_weight({_meta.maxseq, _meta.nkvh, _meta.dh});
    _attn_out = create_weight({_meta.maxseq, _meta.nh, _meta.dh});
    _attn_proj = create_weight({_meta.maxseq, _meta.hs});
    _x_norm_mlp = create_weight({_meta.maxseq, _meta.hs});
    _mlp_gate = create_weight({_meta.maxseq, _meta.di});
    _mlp_up = create_weight({_meta.maxseq, _meta.di});
    _mlp_gate_out = create_weight({_meta.maxseq, _meta.di});
    _mlp_down_out = create_weight({_meta.maxseq, _meta.hs});
    _x_last_norm = create_weight({1, _meta.hs});
    _logits = create_weight({1, _meta.voc});
    // _next_token = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
    // _max_val = Tensor::create({1}, _meta.dtype, LLAISYS_DEVICE_CPU, 0);

}

Qwen2Model::~Qwen2Model() {
    delete _weights_c.in_embed;
    delete _weights_c.out_embed;
    delete _weights_c.out_norm_w;
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        delete _c_attn_norm_w[i];
        delete _c_attn_q_w[i]; delete _c_attn_q_b[i];
        delete _c_attn_k_w[i]; delete _c_attn_k_b[i];
        delete _c_attn_v_w[i]; delete _c_attn_v_b[i];
        delete _c_attn_o_w[i];
        delete _c_mlp_norm_w[i];
        delete _c_mlp_gate_w[i];
        delete _c_mlp_up_w[i];
        delete _c_mlp_down_w[i];
    }
}

int64_t Qwen2Model::runSequence(Qwen2PagedRuntimeState &runtime_state,
                                core::paged_kv::SequenceState &sequence,
                                bool is_prefill,
                                const LlaisysQwen2SamplingParams &params) {
    return infer_batch(runtime_state, {&sequence}, is_prefill, {params})[0];
}

std::vector<int64_t> Qwen2Model::runBatch(
    Qwen2PagedRuntimeState &runtime_state,
    const std::vector<std::shared_ptr<core::scheduler::SequenceEntry>> &entries,
    bool is_prefill,
    const std::vector<LlaisysQwen2SamplingParams> &params) {
    CHECK_ARGUMENT(entries.size() == params.size(), "Qwen2Model::runBatch: entries/params size mismatch");

    std::vector<core::paged_kv::SequenceState *> sequences;
    sequences.reserve(entries.size());
    for (const auto &entry : entries) {
        CHECK_ARGUMENT(entry != nullptr && entry->sequence != nullptr,
                       "Qwen2Model::runBatch: null entry/sequence");
        sequences.push_back(entry->sequence.get());
    }
    return infer_batch(runtime_state, sequences, is_prefill, params);
}

std::vector<int64_t> Qwen2Model::infer_batch(
    Qwen2PagedRuntimeState &runtime_state,
    const std::vector<core::paged_kv::SequenceState *> &sequences,
    bool is_prefill,
    const std::vector<LlaisysQwen2SamplingParams> &params) {
    CHECK_ARGUMENT(!sequences.empty(), "Qwen2Model::infer_batch: sequences must not be empty");
    CHECK_ARGUMENT(sequences.size() == params.size(), "Qwen2Model::infer_batch: sequences/params size mismatch");

    std::vector<core::paged_kv::SequenceState *> prefill_sequences;
    std::vector<LlaisysQwen2SamplingParams> prefill_params;
    std::vector<size_t> prefill_order;
    std::vector<core::paged_kv::SequenceState *> decode_sequences;
    std::vector<LlaisysQwen2SamplingParams> decode_params;
    std::vector<size_t> decode_order;

    for (size_t i = 0; i < sequences.size(); ++i) {
            CHECK_ARGUMENT(sequences[i] != nullptr, "Qwen2Model::infer_batch: null sequence");
        const bool has_prefill_tokens = sequences[i]->numCachedTokens() < sequences[i]->numTokens();
        if (is_prefill && has_prefill_tokens) {
            prefill_sequences.push_back(sequences[i]);
            prefill_params.push_back(params[i]);
            prefill_order.push_back(i);
        } else {
            decode_sequences.push_back(sequences[i]);
            decode_params.push_back(params[i]);
            decode_order.push_back(i);
        }
    }

    std::vector<int64_t> outputs(sequences.size(), 0);

    auto run_group = [&](const std::vector<core::paged_kv::SequenceState *> &group_sequences,
                         const std::vector<LlaisysQwen2SamplingParams> &group_params,
                         bool group_is_prefill) -> std::vector<int64_t> {
        if (group_sequences.empty()) {
            return {};
        }

        core::paged_kv::PrefillBatch prefill_batch;
        core::paged_kv::DecodeBatch decode_batch;
        std::vector<int64_t> compute_input_ids;
        std::vector<int64_t> compute_positions;
        std::vector<size_t> last_token_indices;

        if (group_is_prefill) {
            prefill_batch = core::paged_kv::BlockManager::preparePrefill(group_sequences, PAGED_KV_BLOCK_SIZE);
            compute_input_ids = prefill_batch.input_ids;
            compute_positions = prefill_batch.positions;
            for (size_t i = 0; i + 1 < prefill_batch.cu_seqlens_q.size(); ++i) {
                const size_t q_begin = static_cast<size_t>(prefill_batch.cu_seqlens_q[i]);
                const size_t q_end = static_cast<size_t>(prefill_batch.cu_seqlens_q[i + 1]);
                CHECK_ARGUMENT(q_end > q_begin, "Qwen2Model::infer_batch: prefill sequence has no query tokens");
                last_token_indices.push_back(q_end - 1);
            }
        } else {
            decode_batch = core::paged_kv::BlockManager::prepareDecode(group_sequences, PAGED_KV_BLOCK_SIZE);
            compute_input_ids = decode_batch.input_ids;
            compute_positions = decode_batch.positions;
            for (size_t i = 0; i < decode_batch.input_ids.size(); ++i) {
                last_token_indices.push_back(i);
            }
        }

        const size_t q_token_count = compute_input_ids.size();
        CHECK_ARGUMENT(q_token_count > 0, "Qwen2Model::infer_batch: no query tokens to process");
        CHECK_ARGUMENT(q_token_count <= _meta.maxseq,
                       "Qwen2Model::infer_batch: q_token_count exceeds preallocated model buffers");

        auto tokens_cpu = Tensor::create({q_token_count}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        tokens_cpu->load(compute_input_ids.data());
        auto pos_ids_cpu = Tensor::create({q_token_count}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        pos_ids_cpu->load(compute_positions.data());
        auto tokens_device = (_device_type == LLAISYS_DEVICE_CPU) ? tokens_cpu : tokens_cpu->to(_device_type, _device_id);
        auto pos_ids = (_device_type == LLAISYS_DEVICE_CPU) ? pos_ids_cpu : pos_ids_cpu->to(_device_type, _device_id);

        auto x = _x->view({q_token_count, _meta.hs});
        ops::embedding(x, tokens_device, _in_embed);

        for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
            auto residual = x;
            auto x_norm = _x_norm->view({q_token_count, _meta.hs});
            ops::rms_norm(x_norm, x, _attn_norm_w[layer], _meta.epsilon);

            auto q = _q->view({q_token_count, _meta.nh, _meta.dh});
            auto k = _k->view({q_token_count, _meta.nkvh, _meta.dh});
            auto v = _v->view({q_token_count, _meta.nkvh, _meta.dh});

            ops::linear(q->view({q_token_count, _meta.nh * _meta.dh}), x_norm, _attn_q_w[layer], _attn_q_b[layer]);
            ops::linear(k->view({q_token_count, _meta.nkvh * _meta.dh}), x_norm, _attn_k_w[layer], _attn_k_b[layer]);
            ops::linear(v->view({q_token_count, _meta.nkvh * _meta.dh}), x_norm, _attn_v_w[layer], _attn_v_b[layer]);

            auto q_rope = q;
            auto k_rope = k;
            ops::rope(q_rope, q, pos_ids, _meta.theta);
            ops::rope(k_rope, k, pos_ids, _meta.theta);

            const auto &slot_mapping = group_is_prefill ? prefill_batch.slot_mapping : decode_batch.slot_mapping;
            ops::store_paged_kv_cache(runtime_state.kCache(layer), runtime_state.vCache(layer), k_rope, v, slot_mapping);

            auto attn_out = _attn_out->view({q_token_count, _meta.nh, _meta.dh});
            if (group_is_prefill) {
                ops::paged_attention_prefill(attn_out, q_rope, runtime_state.kCache(layer), runtime_state.vCache(layer),
                                             prefill_batch, _meta.nkvh,
                                             1.0f / sqrtf(static_cast<float>(_meta.dh)));
            } else {
                ops::paged_attention_decode(attn_out, q_rope, runtime_state.kCache(layer), runtime_state.vCache(layer),
                                            decode_batch, _meta.nkvh,
                                            1.0f / sqrtf(static_cast<float>(_meta.dh)));
            }

            auto attn_proj = _attn_proj->view({q_token_count, _meta.hs});
            ops::linear(attn_proj, attn_out->view({q_token_count, _meta.nh * _meta.dh}), _attn_o_w[layer], nullptr);
            ops::add(x, residual, attn_proj);

            residual = x;
            auto x_norm_mlp = _x_norm_mlp->view({q_token_count, _meta.hs});
            ops::rms_norm(x_norm_mlp, x, _mlp_norm_w[layer], _meta.epsilon);

            auto gate = _mlp_gate->view({q_token_count, _meta.di});
            auto up = _mlp_up->view({q_token_count, _meta.di});
            ops::linear(gate, x_norm_mlp, _mlp_gate_w[layer], nullptr);
            ops::linear(up, x_norm_mlp, _mlp_up_w[layer], nullptr);

            auto mlp_gate_out = _mlp_gate_out->view({q_token_count, _meta.di});
            ops::swiglu(mlp_gate_out, gate, up);

            auto mlp_down_out = _mlp_down_out->view({q_token_count, _meta.hs});
            ops::linear(mlp_down_out, mlp_gate_out, _mlp_down_w[layer], nullptr);
            ops::add(x, residual, mlp_down_out);
        }

        for (auto *sequence : group_sequences) {
            sequence->setNumCachedTokens(sequence->numTokens());
        }

        return sample_from_hidden(x, last_token_indices, group_params);
    };

    const auto prefill_outputs = run_group(prefill_sequences, prefill_params, true);
    const auto decode_outputs = run_group(decode_sequences, decode_params, false);

    for (size_t i = 0; i < prefill_outputs.size(); ++i) {
        outputs[prefill_order[i]] = prefill_outputs[i];
    }
    for (size_t i = 0; i < decode_outputs.size(); ++i) {
        outputs[decode_order[i]] = decode_outputs[i];
    }

    return outputs;
}

std::vector<int64_t> Qwen2Model::sample_from_hidden(
    tensor_t hidden_states,
    const std::vector<size_t> &last_token_indices,
    const std::vector<LlaisysQwen2SamplingParams> &params) {
    CHECK_ARGUMENT(last_token_indices.size() == params.size(),
                   "Qwen2Model::sample_from_hidden: indices/params size mismatch");
    const size_t batch_size = last_token_indices.size();
    CHECK_ARGUMENT(batch_size > 0, "Qwen2Model::sample_from_hidden: batch_size must be greater than 0");

    auto x_last = Tensor::create({batch_size, _meta.hs}, _meta.dtype, _device_type, _device_id);
    const size_t row_bytes = _meta.hs * hidden_states->elementSize();
    const auto *src = hidden_states->data();
    auto *dst = x_last->data();
    const LlaisysRuntimeAPI *api = device::getRuntimeAPI(_device_type);
    for (size_t i = 0; i < batch_size; ++i) {
        const size_t row = last_token_indices[i];
        api->memcpy_sync(dst + i * row_bytes, src + row * row_bytes, row_bytes, LLAISYS_MEMCPY_D2D);
    }

    auto x_last_norm = Tensor::create({batch_size, _meta.hs}, _meta.dtype, _device_type, _device_id);
    ops::rms_norm(x_last_norm, x_last, _out_norm_w, _meta.epsilon);

    auto logits = Tensor::create({batch_size, _meta.voc}, _meta.dtype, _device_type, _device_id);
    ops::linear(logits, x_last_norm, _out_embed, nullptr);

    auto next_token_idx = Tensor::create({batch_size}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    for (size_t i = 0; i < batch_size; ++i) {
        auto logits_row = logits->slice(0, i, i + 1);
        auto token_row = next_token_idx->slice(0, i, i + 1);
        ops::sampling(token_row, logits_row, params[i].temperature, params[i].top_k, params[i].top_p, params[i].seed);
    }

    auto next_token_idx_cpu = (_device_type == LLAISYS_DEVICE_CPU) ? next_token_idx : next_token_idx->to(LLAISYS_DEVICE_CPU, 0);
    const int64_t *tokens = reinterpret_cast<const int64_t *>(next_token_idx_cpu->data());
    return std::vector<int64_t>(tokens, tokens + batch_size);
}

} // namespace llaisys::models
