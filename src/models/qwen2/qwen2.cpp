#include "qwen2.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/sampling/op.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include "../../device/runtime_api.hpp"
#include <cmath>

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

    // 静态预分配创建KV cache, 而不是动态分配
    _k_cache.resize(_meta.nlayer);
    _v_cache.resize(_meta.nlayer);
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _k_cache[i] = create_weight({_meta.maxseq, _meta.nkvh, _meta.dh});
        _v_cache[i] = create_weight({_meta.maxseq, _meta.nkvh, _meta.dh});
    }

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

void Qwen2Model::reset() {
    // Reset decode cursor for a new request/session.
    // KV cache tensors are reused and overwritten from position 0 onward.
    _cur_pos = 0;
}

int64_t Qwen2Model::infer(int64_t *token_ids, size_t ntoken, const LlaisysQwen2SamplingParams &params) {

    // _in_embed->slice(0, 1, 10)->debug();
    // _out_embed->slice(0, 1, 10)->debug();
    // exit(1);

    // auto create_tmp = [&](const std::vector<size_t> &shape) {
    //     return Tensor::create(shape, _meta.dtype, _device_type, _device_id);
    // };

    const LlaisysRuntimeAPI *api = device::getRuntimeAPI(_device_type);

    // 1. Tokens to device tensor
    tensor_t tokens_device;
    {
        auto tokens_cpu = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        tokens_cpu->load(token_ids);
        if (_device_type == LLAISYS_DEVICE_CPU) {
            tokens_device = tokens_cpu;
        } else {
            tokens_device = tokens_cpu->to(_device_type, _device_id);
        }
    }

    // 2. Embedding
    // auto x = create_tmp({ntoken, _meta.hs});
    auto x = _x->view({ntoken, _meta.hs});
    ops::embedding(x, tokens_device, _in_embed);

    // 3. Position IDs for RoPE
    tensor_t pos_ids;
    {
        auto pos_ids_cpu = Tensor::create({ntoken}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        int64_t *pos_ptr = (int64_t *)pos_ids_cpu->data();
        for (size_t i = 0; i < ntoken; ++i)
            pos_ptr[i] = _cur_pos + i;
        if (_device_type == LLAISYS_DEVICE_CPU) {
            pos_ids = pos_ids_cpu;
        } else {
            pos_ids = pos_ids_cpu->to(_device_type, _device_id);
        }
    }

    // 4. Layers
    for (size_t i = 0; i < _meta.nlayer; ++i) {
        auto residual = x;

        // Norm
        // auto x_norm = create_tmp({ntoken, _meta.hs});
        auto x_norm = _x_norm->view({ntoken, _meta.hs});
        ops::rms_norm(x_norm, x, _attn_norm_w[i], _meta.epsilon);

        // QKV Projections
        // auto q = create_tmp({ntoken, _meta.nh, _meta.dh});
        // auto k = create_tmp({ntoken, _meta.nkvh, _meta.dh});
        // auto v = create_tmp({ntoken, _meta.nkvh, _meta.dh});
        auto q = _q->view({ntoken, _meta.nh, _meta.dh});
        auto k = _k->view({ntoken, _meta.nkvh, _meta.dh});
        auto v = _v->view({ntoken, _meta.nkvh, _meta.dh});

        ops::linear(q->view({ntoken, _meta.nh * _meta.dh}), x_norm, _attn_q_w[i], _attn_q_b[i]);
        ops::linear(k->view({ntoken, _meta.nkvh * _meta.dh}), x_norm, _attn_k_w[i], _attn_k_b[i]);
        ops::linear(v->view({ntoken, _meta.nkvh * _meta.dh}), x_norm, _attn_v_w[i], _attn_v_b[i]);

        // RoPE
        // auto q_rope = create_tmp({ntoken, _meta.nh, _meta.dh});
        // auto k_rope = create_tmp({ntoken, _meta.nkvh, _meta.dh});
        // in place
        auto q_rope = q;
        auto k_rope = k;
        ops::rope(q_rope, q, pos_ids, _meta.theta);
        ops::rope(k_rope, k, pos_ids, _meta.theta);

        // KV Cache Update
        auto k_cache_slice = _k_cache[i]->slice(0, _cur_pos, _cur_pos + ntoken);
        auto v_cache_slice = _v_cache[i]->slice(0, _cur_pos, _cur_pos + ntoken);

        // Copy to cache
        api->memcpy_sync(k_cache_slice->data(), k_rope->data(), k_rope->numel() * k_rope->elementSize(), LLAISYS_MEMCPY_D2D);
        api->memcpy_sync(v_cache_slice->data(), v->data(), v->numel() * v->elementSize(), LLAISYS_MEMCPY_D2D);

        // Full KV up to now
        auto k_full = _k_cache[i]->slice(0, 0, _cur_pos + ntoken);
        auto v_full = _v_cache[i]->slice(0, 0, _cur_pos + ntoken);

        // Multi-head Attention
        // auto attn_out = create_tmp({ntoken, _meta.nh, _meta.dh});
        auto attn_out = _attn_out->view({ntoken, _meta.nh, _meta.dh});
        ops::self_attention(attn_out, q_rope, k_full, v_full, 1.0f / sqrtf(static_cast<float>(_meta.dh)));

        // Output Projection
        // auto attn_proj = create_tmp({ntoken, _meta.hs})
        auto attn_proj = _attn_proj->view({ntoken, _meta.hs});
        ops::linear(attn_proj, attn_out->view({ntoken, _meta.nh * _meta.dh}), _attn_o_w[i], nullptr);

        // Add
        // auto x_new = create_tmp({ntoken, _meta.hs});
        // ops::add(x_new, residual, attn_proj);
        // x = x_new;
        // add支持inplace
        ops::add(x, residual, attn_proj);

        // MLP
        residual = x;
        // auto x_norm_mlp = create_tmp({ntoken, _meta.hs});
        auto x_norm_mlp = _x_norm_mlp->view({ntoken, _meta.hs});
        ops::rms_norm(x_norm_mlp, x, _mlp_norm_w[i], _meta.epsilon);

        // auto gate = create_tmp({ntoken, _meta.di});
        // auto up = create_tmp({ntoken, _meta.di});
        auto gate = _mlp_gate->view({ntoken, _meta.di});
        auto up = _mlp_up->view({ntoken, _meta.di});
        ops::linear(gate, x_norm_mlp, _mlp_gate_w[i], nullptr);
        ops::linear(up, x_norm_mlp, _mlp_up_w[i], nullptr);

        // auto mlp_gate_out = create_tmp({ntoken, _meta.di});
        auto mlp_gate_out = _mlp_gate_out->view({ntoken, _meta.di});
        ops::swiglu(mlp_gate_out, gate, up);

        // auto mlp_down_out = create_tmp({ntoken, _meta.hs});
        auto mlp_down_out = _mlp_down_out->view({ntoken, _meta.hs});
        ops::linear(mlp_down_out, mlp_gate_out, _mlp_down_w[i], nullptr);

        // auto x_final = create_tmp({ntoken, _meta.hs});
        // ops::add(x_final, residual, mlp_down_out);
        // x = x_final;
        // add支持inplace
        ops::add(x, residual, mlp_down_out);
    }

    _cur_pos += ntoken;

    // 5. Final Norm & LM Head (only for the last token)
    auto x_last = x->slice(0, ntoken - 1, ntoken);
    // auto x_last_norm = create_tmp({1, _meta.hs});
    auto x_last_norm = _x_last_norm;
    ops::rms_norm(x_last_norm, x_last, _out_norm_w, _meta.epsilon);

    // auto logits = create_tmp({1, _meta.voc});
    auto logits = _logits;
    ops::linear(logits, x_last_norm, _out_embed, nullptr);

    // 6. Argmax (on GPU if available, only transfer result back to CPU)
    auto next_token_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, _device_type, _device_id);
    auto max_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id);

    // if (decoding_type == LLAISYS_QWEN2_SAMPLING) {
    //     // Use sampling for decoding
    //     ops::sampling(next_token_idx, logits->view({1, _meta.voc}), 
    //                  params.temperature, params.top_k, params.top_p, params.seed);
    // } else {
    //     // Default: use argmax
    //     auto max_val = Tensor::create({1}, _meta.dtype, _device_type, _device_id);
    //     ops::argmax(next_token_idx, max_val, logits->view({_meta.voc}));
    // }
    ops::sampling(next_token_idx, logits->view({1, _meta.voc}), 
                  params.temperature, params.top_k, params.top_p, params.seed);

    // Transfer result from device to CPU for return
    auto next_token_idx_cpu = next_token_idx->to(LLAISYS_DEVICE_CPU, 0);
    return *(int64_t *)next_token_idx_cpu->data();
}

} // namespace llaisys::models
