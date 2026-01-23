#pragma once
#include "llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include <vector>

namespace llaisys::models {

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    ~Qwen2Model();

    LlaisysQwen2Weights &weights() { return _weights_c; }
    int64_t infer(int64_t *token_ids, size_t ntoken);

private:
    LlaisysQwen2Meta _meta;
    LlaisysQwen2Weights _weights_c;
    
    // Storage for weight tensors to keep shared_ptr alive
    // device tensor
    tensor_t _in_embed;
    tensor_t _out_embed;
    tensor_t _out_norm_w;
    std::vector<tensor_t> _attn_norm_w;
    std::vector<tensor_t> _attn_q_w, _attn_q_b;
    std::vector<tensor_t> _attn_k_w, _attn_k_b;
    std::vector<tensor_t> _attn_v_w, _attn_v_b;
    std::vector<tensor_t> _attn_o_w;
    std::vector<tensor_t> _mlp_norm_w;
    std::vector<tensor_t> _mlp_gate_w, _mlp_up_w, _mlp_down_w;

    // c风格结构体存储经过c封装后的不透明指针, 用于给前端暴露接口
    // for LlaisysQwen2Weights
    std::vector<llaisysTensor_t> _c_attn_norm_w;
    std::vector<llaisysTensor_t> _c_attn_q_w, _c_attn_q_b;
    std::vector<llaisysTensor_t> _c_attn_k_w, _c_attn_k_b;
    std::vector<llaisysTensor_t> _c_attn_v_w, _c_attn_v_b;
    std::vector<llaisysTensor_t> _c_attn_o_w;
    std::vector<llaisysTensor_t> _c_mlp_norm_w;
    std::vector<llaisysTensor_t> _c_mlp_gate_w, _c_mlp_up_w, _c_mlp_down_w;

    // 静态预分配KV Cache, 而不是动态分配
    // todo: 动态分配, 减少内存占用
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;
    size_t _cur_pos = 0;

    // 提前创建中间结果的tensor, 通过复用以减少infer时的内存开销
    // device tensor
    tensor_t _x;               // [ntoken, hs]
    tensor_t _x_norm;          // [ntoken, hs]
    tensor_t _q, _k, _v;       // [ntoken, nh/nkvh, dh] - after RoPE
    tensor_t _attn_out;        // [ntoken, nh, dh]
    tensor_t _attn_proj;       // [ntoken, hs]
    tensor_t _x_norm_mlp;      // [ntoken, hs]
    tensor_t _mlp_gate, _mlp_up; // [ntoken, di]
    tensor_t _mlp_gate_out;    // [ntoken, di]
    tensor_t _mlp_down_out;    // [ntoken, hs]
    tensor_t _x_last_norm;      // [1, hs]
    tensor_t _logits;          // [ntoken, voc]
    // tensor_t _next_token;      // [1]
    // tensor_t _max_val;         // [1]
    
    // Device info
    llaisysDeviceType_t _device_type;
    int _device_id;
};

} // namespace llaisys::models
