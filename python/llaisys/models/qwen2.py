from typing import Sequence
import json
import numpy as np
import ctypes
import torch
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.models import LlaisysQwen2Meta
from ..tensor import Tensor

from pathlib import Path
import safetensors
import secrets


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)

        # 从qwen2的config.json中解析参数, 构造LlaisysQwen2Meta
        meta = LlaisysQwen2Meta()
        meta.dtype = DataType.BF16                              # DeepSeek R1 uses BF16
        meta.nlayer = config["num_hidden_layers"]
        meta.hs = config["hidden_size"]
        meta.nh = config["num_attention_heads"]
        meta.nkvh = config.get("num_key_value_heads", meta.nh)
        meta.dh = meta.hs // meta.nh                            # hidden size per head
        meta.di = config["intermediate_size"]
        meta.maxseq = config.get("max_position_embeddings", 131072)
        meta.voc = config["vocab_size"]
        meta.epsilon = config.get("rms_norm_eps", 1e-6)
        meta.theta = config.get("rope_theta", 1000000.0)
        meta.end_token = config.get("eos_token_id", 151643)
        self.end_token = meta.end_token
        self.tied = config.get("tie_word_embeddings", False)
        # print(f"[llaisys] Tied: {self.tied}, config.tied: {config['tie_word_embeddings']}")

        # 创建c++后端的模型并将指针绑定到self._model
        device_ids = (ctypes.c_int * 1)(0)          # 只支持一个cpu设备
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(meta),
            device,
            device_ids,
            1
        )

        # 通过LlaisysQwen2Weights的结构体指针, 获取c++后端创建的权重tensor的指针
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        weights = weights_ptr.contents

        # 通过map, 将safetensors的key和c++后端的权重tensor对应起来
        name_map = {
            "model.embed_tokens.weight": weights.in_embed,
            "lm_head.weight": weights.out_embed,
            "model.norm.weight": weights.out_norm_w,
        }
        for i in range(meta.nlayer):
            name_map[f"model.layers.{i}.input_layernorm.weight"] = weights.attn_norm_w[i]
            name_map[f"model.layers.{i}.self_attn.q_proj.weight"] = weights.attn_q_w[i]
            name_map[f"model.layers.{i}.self_attn.k_proj.weight"] = weights.attn_k_w[i]
            name_map[f"model.layers.{i}.self_attn.v_proj.weight"] = weights.attn_v_w[i]
            name_map[f"model.layers.{i}.self_attn.q_proj.bias"] = weights.attn_q_b[i]
            name_map[f"model.layers.{i}.self_attn.k_proj.bias"] = weights.attn_k_b[i]
            name_map[f"model.layers.{i}.self_attn.v_proj.bias"] = weights.attn_v_b[i]
            name_map[f"model.layers.{i}.self_attn.o_proj.weight"] = weights.attn_o_w[i]
            name_map[f"model.layers.{i}.post_attention_layernorm.weight"] = weights.mlp_norm_w[i]
            name_map[f"model.layers.{i}.mlp.gate_proj.weight"] = weights.mlp_gate_w[i]
            name_map[f"model.layers.{i}.mlp.up_proj.weight"] = weights.mlp_up_w[i]
            name_map[f"model.layers.{i}.mlp.down_proj.weight"] = weights.mlp_down_w[i]
           
        # 从safetensors中加载权重, 通过map, 将权重加载到c++后端的权重tensor
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if name in name_map:
                        data = f.get_tensor(name)
                        ptr = data.data_ptr()
                        # 使用Tensor类在销毁的时候会自动释放tensor的指针
                        # 而这里的tensor指针同时也由c++后端的Qwen2Model管理, 析构函数中会二次释放
                        # 所以不使用Tensor类, 直接使用c++后端的指针, 避免double free
                        if name == "model.embed_tokens.weight" and self.tied: # Tied embeddings
                            # print("[llaisys] Tied embeddings, loading lm_head.weight to out_embed")
                            # out_embed = Tensor(tensor=weights.out_embed)
                            # out_embed.load(ctypes.c_void_p(ptr))
                            LIB_LLAISYS.tensorLoad(weights.out_embed, ctypes.c_void_p(ptr))
                        # t = Tensor(tensor=name_map[name])
                        # t.load(ctypes.c_void_p(ptr))
                        # print(f"[llaisys] Loading weight: {name} to {t}")
                        LIB_LLAISYS.tensorLoad(name_map[name], ctypes.c_void_p(ptr))
                    else:
                        print(f"Skipping unknown weight: {name}")

    def __del__(self):
        if hasattr(self, "_model") and self._model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 120,
        temperature: float = 1.0,
        top_k: int = 0,                     # 0 to disable
        top_p: float = 0.0,                 # 0 to disable
        seed: int = 0,
        stream: bool = False,
    ):
        # New request/session: clear backend decode cursor/KV-cache position.
        LIB_LLAISYS.llaisysQwen2ModelReset(self._model)

        from ..libllaisys.models import LlaisysQwen2SamplingParams

        # Seed policy:
        # - seed == 0: non-deterministic per generation call (random base seed)
        # - seed != 0: deterministic/reproducible across runs
        base_seed = seed if seed != 0 else secrets.randbits(64)
        sampling_params = LlaisysQwen2SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=base_seed
        )
        sampling_params_ptr = ctypes.byref(sampling_params)

        def _token_generator():
            curr_inputs = list(inputs)
            for step in range(max_new_tokens or 2048):
                # Avoid resetting RNG state every token: advance seed by decoding step.
                # This keeps one run stochastic while preserving reproducibility when
                # user provides a fixed non-zero seed.
                # sampling_params.seed = ctypes.c_uint64((base_seed + step) & 0xFFFFFFFFFFFFFFFF).value

                token_ids = (ctypes.c_int64 * len(curr_inputs))(*curr_inputs)

                # Infer next token
                next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    token_ids,
                    ctypes.c_size_t(len(curr_inputs)),
                    sampling_params_ptr
                )

                next_token = int(next_token)
                yield next_token

                # 检测end_token
                if next_token == self.end_token:
                    break

                # 实现KV cache的时候, 只需传入最新的token
                # todo: KV cache超出的时候, 需要传入历史的token
                curr_inputs = [next_token]

        if stream:
            return _token_generator()

        output_tokens = list(inputs)
        output_tokens.extend(_token_generator())
        return output_tokens
