import argparse
import itertools
import json
import os
import statistics
import subprocess
import sys
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch
from test_utils import random_tensor, check_equal


# --------------------------
# Worker: single BM/BN/BK run
# --------------------------
def run_single(args):
    dtype_name = args.dtype
    device_name = args.device
    M, K, N = args.M, args.K, args.N

    # Build tensors
    x, x_ = random_tensor((M, K), dtype_name, device_name, scale=0.1)
    w, w_ = random_tensor((N, K), dtype_name, device_name, scale=0.01)

    bias, bias_ = None, None
    if args.use_bias:
        bias, bias_ = random_tensor((N,), dtype_name, device_name)

    out, out_ = random_tensor((M, N), dtype_name, device_name)

    # Correctness check once
    torch.nn.functional.linear(x, w, bias, out=out)
    llaisys.Ops.linear(out_, x_, w_, bias_)
    assert check_equal(out_, out, atol=args.atol, rtol=args.rtol), (
        f"correctness check failed: BM={args.bm}, BN={args.bn}, BK={args.bk}"
    )

    api = llaisys.RuntimeAPI(llaisys.DeviceType.CPU)

    # Warmup
    for _ in range(args.warmup):
        llaisys.Ops.linear(out_, x_, w_, bias_)
    api.device_synchronize()

    # Benchmark
    durations = []
    for _ in range(args.repeat):
        t0 = time.perf_counter()
        llaisys.Ops.linear(out_, x_, w_, bias_)
        api.device_synchronize()
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000.0)

    result = {
        "bm": args.bm,
        "bn": args.bn,
        "bk": args.bk,
        "shape": [M, K, N],
        "dtype": dtype_name,
        "use_bias": bool(args.use_bias),
        "avg_ms": statistics.mean(durations),
        "median_ms": statistics.median(durations),
        "min_ms": min(durations),
        "max_ms": max(durations),
    }
    print(json.dumps(result, ensure_ascii=False))


# --------------------------
# Driver: grid search
# --------------------------
def grid_search(args):
    bm_list = [int(x) for x in args.bm_list.split(",") if x.strip()]
    bn_list = [int(x) for x in args.bn_list.split(",") if x.strip()]
    bk_list = [int(x) for x in args.bk_list.split(",") if x.strip()]

    combos = list(itertools.product(bm_list, bn_list, bk_list))
    print(f"[INFO] Search space size: {len(combos)}")

    script_path = os.path.abspath(__file__)
    results = []

    for i, (bm, bn, bk) in enumerate(combos, start=1):
        env = os.environ.copy()
        env["LLAISYS_LINEAR_BM"] = str(bm)
        env["LLAISYS_LINEAR_BN"] = str(bn)
        env["LLAISYS_LINEAR_BK"] = str(bk)

        cmd = [
            sys.executable,
            script_path,
            "--single",
            f"--bm={bm}",
            f"--bn={bn}",
            f"--bk={bk}",
            f"--M={args.M}",
            f"--K={args.K}",
            f"--N={args.N}",
            f"--dtype={args.dtype}",
            f"--device={args.device}",
            f"--warmup={args.warmup}",
            f"--repeat={args.repeat}",
            f"--atol={args.atol}",
            f"--rtol={args.rtol}",
        ]
        if args.use_bias:
            cmd.append("--use-bias")

        proc = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode != 0:
            print(f"[WARN] combo {i}/{len(combos)} failed: BM={bm}, BN={bn}, BK={bk}")
            print(proc.stderr.strip())
            continue

        # last non-empty line should be JSON
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        payload = json.loads(lines[-1])
        results.append(payload)
        print(
            f"[{i:>3}/{len(combos)}] BM={bm:>3} BN={bn:>3} BK={bk:>3} "
            f"avg={payload['avg_ms']:.3f} ms"
        )

    if not results:
        raise RuntimeError("No valid results found. Please check build/runtime environment.")

    results.sort(key=lambda x: x["avg_ms"])
    best = results[0]

    print("\n=== Top Results (by avg_ms) ===")
    for rank, item in enumerate(results[: args.topk], start=1):
        print(
            f"#{rank}: BM={item['bm']}, BN={item['bn']}, BK={item['bk']}, "
            f"avg={item['avg_ms']:.3f} ms, med={item['median_ms']:.3f} ms"
        )

    print("\n=== Best Config ===")
    print(json.dumps(best, ensure_ascii=False, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Full results saved to: {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search best BM/BN/BK for llaisys linear tiled kernel")

    parser.add_argument("--single", action="store_true", help="Worker mode for one BM/BN/BK combination")

    parser.add_argument("--bm", type=int, default=32)
    parser.add_argument("--bn", type=int, default=64)
    parser.add_argument("--bk", type=int, default=64)

    parser.add_argument("--bm-list", type=str, default="16,32,64")
    parser.add_argument("--bn-list", type=str, default="32,64,128")
    parser.add_argument("--bk-list", type=str, default="32,64,128")

    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--N", type=int, default=4096)

    parser.add_argument("--dtype", type=str, default="f32", choices=["f32", "f16", "bf16"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu"])

    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--use-bias", action="store_true")

    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-json", type=str, default="")

    args = parser.parse_args()

    if args.single:
        run_single(args)
    else:
        grid_search(args)
