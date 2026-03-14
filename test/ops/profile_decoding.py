#!/usr/bin/env python3
"""
Profile decoding operators (argmax & sampling) and compare LLAISYS vs PyTorch
on CPU / NVIDIA devices.

Usage examples:
  python test/ops/profile_decoding.py --device cpu
  python test/ops/profile_decoding.py --device nvidia --batch-size 1 --vocab-size 32000
  python test/ops/profile_decoding.py --device all --repeat 50 --warmup 10
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch

from test_utils import random_tensor, llaisys_device, torch_device


def sync_device(device_name: str, api=None):
    if device_name == "nvidia":
        torch.cuda.synchronize()
    if api is not None:
        api.device_synchronize()


def benchmark_ms(
    fn: Callable[[], None],
    device_name: str,
    warmup: int,
    repeat: int,
    api,
) -> float:
    for _ in range(warmup):
        fn()
    sync_device(device_name, api)

    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync_device(device_name, api)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / repeat


def torch_sampling(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> torch.Tensor:
    """Reference sampling implementation in PyTorch for profiling comparison."""
    eps = 1e-6
    scaled = logits / max(temperature, eps)
    probs = torch.softmax(scaled, dim=-1)

    bsz, vocab = probs.shape

    if top_k > 0 and top_k < vocab:
        topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
        filtered = torch.zeros_like(probs)
        filtered.scatter_(dim=-1, index=topk_idx, src=topk_vals)
        probs = filtered
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        cutoff = (cumsum < top_p).sum(dim=-1, keepdim=True)
        arange_idx = torch.arange(vocab, device=logits.device).view(1, vocab)
        keep_sorted = arange_idx <= cutoff

        sorted_mask = keep_sorted.to(sorted_probs.dtype)
        kept_sorted_probs = sorted_probs * sorted_mask

        masked = torch.zeros_like(probs)
        masked.scatter_(dim=-1, index=sorted_idx, src=kept_sorted_probs)
        probs = masked
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-12)

    if seed > 0:
        g = torch.Generator(device=logits.device)
        g.manual_seed(seed)
        idx = torch.multinomial(probs, num_samples=1, generator=g)
    else:
        idx = torch.multinomial(probs, num_samples=1)

    return idx.squeeze(-1)


def profile_one_device(
    device_name: str,
    dtype_name: str,
    batch_size: int,
    vocab_size: int,
    warmup: int,
    repeat: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> List[Dict]:
    print(f"\n=== Device: {device_name} | dtype: {dtype_name} | shape: [{batch_size}, {vocab_size}] ===")

    logits_torch, logits_llaisys = random_tensor((batch_size, vocab_size), dtype_name, device_name)

    api = llaisys.RuntimeAPI(llaisys_device(device_name))

    # ---------- Argmax ----------
    # LLAISYS argmax currently consumes 1D input.
    logits1_torch, logits1_llaisys = random_tensor((vocab_size,), dtype_name, device_name)

    max_idx_ll = llaisys.Tensor((1,), dtype=llaisys.DataType.I64, device=llaisys_device(device_name))
    max_val_ll = llaisys.Tensor((1,), dtype=getattr(llaisys.DataType, dtype_name.upper()), device=llaisys_device(device_name))

    max_idx_torch = torch.zeros((1,), dtype=torch.int64, device=torch_device(device_name))
    max_val_torch = torch.zeros((1,), dtype=logits1_torch.dtype, device=torch_device(device_name))

    def torch_argmax_1d():
        torch.max(logits1_torch, keepdim=True, dim=-1, out=(max_val_torch, max_idx_torch))

    def llaisys_argmax_1d():
        llaisys.Ops.argmax(max_idx_ll, max_val_ll, logits1_llaisys)

    torch_argmax_ms = benchmark_ms(torch_argmax_1d, device_name, warmup, repeat, api)
    llaisys_argmax_ms = benchmark_ms(llaisys_argmax_1d, device_name, warmup, repeat, api)

    # ---------- Sampling ----------
    out_ll = llaisys.Tensor((batch_size,), dtype=llaisys.DataType.I64, device=llaisys_device(device_name))

    def torch_sampling_fn():
        torch_sampling(logits_torch, temperature=temperature, top_k=top_k, top_p=top_p, seed=seed)

    def llaisys_sampling_fn():
        llaisys.Ops.sampling(
            out_ll,
            logits_llaisys,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )

    torch_sampling_ms = benchmark_ms(torch_sampling_fn, device_name, warmup, repeat, api)
    llaisys_sampling_ms = benchmark_ms(llaisys_sampling_fn, device_name, warmup, repeat, api)

    results = [
        {
            "device": device_name,
            "op": "argmax_1d",
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "dtype": dtype_name,
            "torch_ms": torch_argmax_ms,
            "llaisys_ms": llaisys_argmax_ms,
            "speedup_vs_torch": torch_argmax_ms / max(llaisys_argmax_ms, 1e-12),
        },
        {
            "device": device_name,
            "op": "sampling",
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "dtype": dtype_name,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "seed": seed,
            "torch_ms": torch_sampling_ms,
            "llaisys_ms": llaisys_sampling_ms,
            "speedup_vs_torch": torch_sampling_ms / max(llaisys_sampling_ms, 1e-12),
        },
    ]

    return results


def print_table(rows: List[Dict]):
    print("\n" + "=" * 96)
    print("Decoding Profile (ms/op)")
    print("=" * 96)
    print(f"{'device':<8} {'op':<16} {'shape':<16} {'torch':>12} {'llaisys':>12} {'torch/llaisys':>16}")
    print("-" * 96)
    for r in rows:
        shape = f"[{r['batch_size']},{r['vocab_size']}]"
        print(
            f"{r['device']:<8} {r['op']:<16} {shape:<16} "
            f"{r['torch_ms']:>12.4f} {r['llaisys_ms']:>12.4f} {r['speedup_vs_torch']:>16.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Profile argmax/sampling: LLAISYS vs PyTorch")
    parser.add_argument("--device", choices=["cpu", "nvidia", "all"], default="cpu")
    parser.add_argument("--dtype", choices=["f32", "f16", "bf16"], default="f32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    devices = ["cpu", "nvidia", "mx"] if args.device == "all" else [args.device]

    all_rows: List[Dict] = []
    for d in devices:
        if d == "nvidia" and not torch.cuda.is_available():
            print("[WARN] NVIDIA requested but torch.cuda is unavailable. Skip nvidia.")
            continue

        rows = profile_one_device(
            device_name=d,
            dtype_name=args.dtype,
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            warmup=args.warmup,
            repeat=args.repeat,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No profiling results. Check environment/device settings.")

    print_table(all_rows)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
