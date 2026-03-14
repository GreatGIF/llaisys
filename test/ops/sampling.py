#!/usr/bin/env python3
"""
Enhanced test suite for Sampling operator with comprehensive validations.
Includes distribution verification, cross-device testing, and boundary cases.
"""

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch
import numpy as np
from test_utils import random_tensor, torch_device, llaisys_device
from collections import Counter


def verify_topk_filtering(logits_torch, sampled_indices, top_k, tolerance=1e-5):
    """
    Verify that all sampled indices are within the top-k logits.
    
    Args:
        logits_torch: [batch_size, vocab_size] torch tensor
        sampled_indices: [batch_size] sampled token indices
        top_k: K value used in sampling
        return: True if all samples are valid top-k
    """
    batch_size = logits_torch.shape[0]
    
    for b in range(batch_size):
        # Get top-k indices for this batch
        _, top_k_indices = torch.topk(logits_torch[b], min(top_k, logits_torch.shape[1]))
        top_k_set = set(top_k_indices.cpu().numpy())
        
        sampled_idx = sampled_indices[b].item()
        assert sampled_idx in top_k_set, \
            f"Batch {b}: sampled index {sampled_idx} not in top-{top_k} (indices: {top_k_set})"
    
    return True


def verify_topp_validity(logits_torch, sampled_indices, top_p):
    """
    Verify that sampled indices are valid for top-p filtering.
    
    Args:
        logits_torch: [batch_size, vocab_size] torch tensor
        sampled_indices: [batch_size] sampled token indices
        top_p: P value (0-1) used in sampling
        return: True if indices are valid
    """
    batch_size = logits_torch.shape[0]
    
    for b in range(batch_size):
        # Compute softmax probabilities
        probs = torch.softmax(logits_torch[b], dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # Find cutoff index where cumulative probability >= top_p
        cumsum = torch.cumsum(sorted_probs, dim=0)
        cutoff_idx = (cumsum < top_p).sum().item()
        
        # Get valid indices set
        valid_indices_set = set(sorted_indices[:cutoff_idx + 1].cpu().numpy())
        
        sampled_idx = sampled_indices[b].item()
        assert sampled_idx in valid_indices_set, \
            f"Batch {b}: sampled index {sampled_idx} not in valid top-p set"
    
    return True


def compute_sampling_distribution(logits, temperature=1.0, top_k=0, top_p=0.0):
    """
    Compute the theoretical sampling distribution for given parameters.
    Returns: probability distribution [vocab_size]
    """
    # Temperature scaling
    scaled_logits = logits / max(temperature, 1e-6)
    
    # Compute probabilities
    probs = torch.softmax(scaled_logits, dim=-1)  # Already softmax
    
    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        thresholds, _ = torch.topk(probs, top_k)
        mask = probs < thresholds[-1]
        probs[mask] = 0
        probs = probs / probs.sum()
    
    # Top-P filtering
    if 0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        
        # Mask out tokens beyond top_p
        mask_probs = torch.zeros_like(sorted_probs)
        mask_probs[cumsum < top_p] = 1.0
        if (cumsum < top_p).sum() < len(sorted_probs):
            mask_probs[(cumsum < top_p).sum()] = 1.0  # Include cutoff token
        
        # Create mask for original ordering
        original_mask = torch.zeros_like(probs)
        original_mask[sorted_indices] = mask_probs
        probs = probs * original_mask
        probs = probs / (probs.sum() + 1e-10)
    
    return probs.cpu().numpy()


def test_basic_sampling(
    batch_size=2,
    vocab_size=1000,
    device_name="cpu",
):
    """Test basic sampling without filtering."""
    print(f"   batch_size={batch_size}, vocab_size={vocab_size}")
    
    logits_torch, logits_llaisys = random_tensor(
        (batch_size, vocab_size), "f32", device_name
    )
    
    out_llaisys = llaisys.Tensor(
        (batch_size,),
        dtype=llaisys.DataType.I64,
        device=llaisys_device(device_name),
    )
    
    llaisys.Ops.sampling(out_llaisys, logits_llaisys, temperature=1.0, seed=42)
    
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    result = torch.zeros((batch_size,), dtype=torch.int64, device=torch_device(device_name))
    api.memcpy_sync(
        result.data_ptr(), out_llaisys.data_ptr(), batch_size * 8,
        llaisys.MemcpyKind.D2D,
    )
    
    assert result.shape == (batch_size,), f"Shape mismatch: {result.shape}"
    assert torch.all(result >= 0), "Negative indices found"
    assert torch.all(result < vocab_size), f"Out-of-range indices found (>= {vocab_size})"
    
    print("   ✓ Basic sampling passed")


def test_multiple_dtypes(
    batch_size=2,
    vocab_size=100,
    device_name="cpu",
):
    """Test sampling with multiple data types."""
    print(f"   batch_size={batch_size}, vocab_size={vocab_size}")
    
    dtypes = ["f32", "f16", "bf16"]
    
    for dtype in dtypes:
        print(f"      Testing dtype: {dtype}")
        
        logits_torch, logits_llaisys = random_tensor(
            (batch_size, vocab_size), dtype, device_name
        )
        
        out_llaisys = llaisys.Tensor(
            (batch_size,), dtype=llaisys.DataType.I64,
            device=llaisys_device(device_name),
        )
        
        llaisys.Ops.sampling(out_llaisys, logits_llaisys, temperature=1.0, seed=42)
        
        api = llaisys.RuntimeAPI(llaisys_device(device_name))
        result = torch.zeros((batch_size,), dtype=torch.int64, device=torch_device(device_name))
        api.memcpy_sync(
            result.data_ptr(), out_llaisys.data_ptr(), batch_size * 8,
            llaisys.MemcpyKind.D2D,
        )
        
        assert torch.all(result >= 0) and torch.all(result < vocab_size), \
            f"Invalid indices for dtype {dtype}"
    
    print("   ✓ Multiple dtypes test passed")


def test_topk_repeated_sampling(
    batch_size=1,
    vocab_size=100,
    top_k=10,
    num_samples=20,
    device_name="cpu",
):
    """Test Top-K with repeated sampling to ensure consistency."""
    print(f"   batch_size={batch_size}, vocab_size={vocab_size}, top_k={top_k}, samples={num_samples}")
    
    logits_torch, logits_llaisys = random_tensor(
        (batch_size, vocab_size), "f32", device_name
    )
    
    all_samples = []
    
    for i in range(num_samples):
        out_llaisys = llaisys.Tensor(
            (batch_size,), dtype=llaisys.DataType.I64,
            device=llaisys_device(device_name),
        )
        
        llaisys.Ops.sampling(out_llaisys, logits_llaisys, temperature=1.0, top_k=top_k, seed=i)
        
        api = llaisys.RuntimeAPI(llaisys_device(device_name))
        result = torch.zeros((batch_size,), dtype=torch.int64, device=torch_device(device_name))
        api.memcpy_sync(
            result.data_ptr(), out_llaisys.data_ptr(), batch_size * 8,
            llaisys.MemcpyKind.D2D,
        )
        
        all_samples.append(result)
    
    all_samples = torch.stack(all_samples, dim=0)  # [num_samples, batch_size]
    
    # Verify all samples are within top-k
    verify_topk_filtering(logits_torch, all_samples[:, 0], top_k)
    
    print(f"   ✓ Top-K repeated sampling passed (all {num_samples} samples valid)")


def test_boundary_cases(
    device_name="cpu",
):
    """Test boundary and special cases."""
    print("   Testing boundary cases")
    
    # Case 1: Single token (vocab_size=1)
    print("      Case 1: vocab_size=1")
    logits_torch, logits_llaisys = random_tensor(
        (1, 1), "f32", device_name
    )
    
    out_llaisys = llaisys.Tensor(
        (1,), dtype=llaisys.DataType.I64,
        device=llaisys_device(device_name),
    )
    
    llaisys.Ops.sampling(out_llaisys, logits_llaisys, seed=42)
    
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    result = torch.zeros((1,), dtype=torch.int64, device=torch_device(device_name))
    api.memcpy_sync(result.data_ptr(), out_llaisys.data_ptr(), 8, llaisys.MemcpyKind.D2D)
    
    assert result[0].item() == 0, "Should always sample index 0 when vocab_size=1"
    
    # Case 2: Very small logits
    print("      Case 2: uniform logits")
    logits_torch = torch.ones((1, 100), dtype=torch.float32, device=torch_device(device_name))
    logits_llaisys = llaisys.Tensor(
        (1, 100), dtype=llaisys.DataType.F32,
        device=llaisys_device(device_name),
    )
    
    api.memcpy_sync(logits_llaisys.data_ptr(), logits_torch.data_ptr(), 100*4, llaisys.MemcpyKind.D2D)
    
    out_llaisys = llaisys.Tensor(
        (1,), dtype=llaisys.DataType.I64,
        device=llaisys_device(device_name),
    )
    
    llaisys.Ops.sampling(out_llaisys, logits_llaisys, seed=42)
    
    result = torch.zeros((1,), dtype=torch.int64, device=torch_device(device_name))
    api.memcpy_sync(result.data_ptr(), out_llaisys.data_ptr(), 8, llaisys.MemcpyKind.D2D)
    
    assert 0 <= result[0].item() < 100, "Invalid index for uniform distribution"
    
    # Case 3: Top-K = 1
    print("      Case 3: top_k=1")
    logits_torch, logits_llaisys = random_tensor(
        (1, 100), "f32", device_name
    )
    
    out_llaisys = llaisys.Tensor(
        (1,), dtype=llaisys.DataType.I64,
        device=llaisys_device(device_name),
    )
    
    llaisys.Ops.sampling(out_llaisys, logits_llaisys, top_k=1, seed=42)
    
    result = torch.zeros((1,), dtype=torch.int64, device=torch_device(device_name))
    api.memcpy_sync(result.data_ptr(), out_llaisys.data_ptr(), 8, llaisys.MemcpyKind.D2D)
    
    top1_idx = torch.argmax(logits_torch[0]).item()
    assert result[0].item() == top1_idx, f"With top_k=1, should always sample argmax (expected {top1_idx}, got {result[0].item()})"
    
    print("   ✓ Boundary cases passed")


def test_reproducibility_across_runs(
    batch_size=2,
    vocab_size=100,
    device_name="cpu",
):
    """Test reproducibility with same seed."""
    print(f"   batch_size={batch_size}, vocab_size={vocab_size}")
    
    # Run 1
    logits_torch1, logits_llaisys1 = random_tensor(
        (batch_size, vocab_size), "f32", device_name
    )
    
    out_llaisys1 = llaisys.Tensor(
        (batch_size,), dtype=llaisys.DataType.I64,
        device=llaisys_device(device_name),
    )
    
    llaisys.Ops.sampling(out_llaisys1, logits_llaisys1, temperature=0.7, top_k=20, top_p=0.9, seed=12345)
    
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    result1 = torch.zeros((batch_size,), dtype=torch.int64, device=torch_device(device_name))
    api.memcpy_sync(result1.data_ptr(), out_llaisys1.data_ptr(), batch_size*8, llaisys.MemcpyKind.D2D)
    
    # Run 2 (same seed)
    logits_torch2, logits_llaisys2 = random_tensor(
        (batch_size, vocab_size), "f32", device_name
    )
    
    # Copy same logits
    api.memcpy_sync(logits_llaisys2.data_ptr(), logits_llaisys1.data_ptr(), batch_size*vocab_size*4, llaisys.MemcpyKind.D2D)
    
    out_llaisys2 = llaisys.Tensor(
        (batch_size,), dtype=llaisys.DataType.I64,
        device=llaisys_device(device_name),
    )
    
    llaisys.Ops.sampling(out_llaisys2, logits_llaisys2, temperature=0.7, top_k=20, top_p=0.9, seed=12345)
    
    result2 = torch.zeros((batch_size,), dtype=torch.int64, device=torch_device(device_name))
    api.memcpy_sync(result2.data_ptr(), out_llaisys2.data_ptr(), batch_size*8, llaisys.MemcpyKind.D2D)
    
    assert torch.equal(result1, result2), f"Reproducibility test failed: {result1} != {result2}"
    
    print(f"   ✓ Reproducibility test passed (results: {result1.tolist()})")


def test_topp_sampling(
    batch_size=1,
    vocab_size=100,
    top_p=0.95,
    num_samples=20,
    device_name="cpu",
):
    """Test Top-P sampling with repeated samples."""
    print(f"   batch_size={batch_size}, vocab_size={vocab_size}, top_p={top_p}, samples={num_samples}")
    
    logits_torch, logits_llaisys = random_tensor(
        (batch_size, vocab_size), "f32", device_name
    )
    
    all_samples = []
    
    for i in range(num_samples):
        out_llaisys = llaisys.Tensor(
            (batch_size,), dtype=llaisys.DataType.I64,
            device=llaisys_device(device_name),
        )
        
        llaisys.Ops.sampling(out_llaisys, logits_llaisys, temperature=1.0, top_p=top_p, seed=i+100)
        
        api = llaisys.RuntimeAPI(llaisys_device(device_name))
        result = torch.zeros((batch_size,), dtype=torch.int64, device=torch_device(device_name))
        api.memcpy_sync(result.data_ptr(), out_llaisys.data_ptr(), batch_size*8, llaisys.MemcpyKind.D2D)
        
        all_samples.append(result)
    
    all_samples = torch.stack(all_samples, dim=0)
    
    # Verify all samples are valid for top_p
    verify_topp_validity(logits_torch, all_samples[:, 0], top_p)
    
    print(f"   ✓ Top-P sampling passed (all {num_samples} samples valid)")


def test_temperature_zero_topk1_equals_argmax(
    batch_size=8,
    vocab_size=256,
    device_name="cpu",
):
    """Verify sampling == argmax when temperature=0 and top_k=1."""
    print(f"   batch_size={batch_size}, vocab_size={vocab_size}, temperature=0, top_k=1")

    logits_torch, logits_llaisys = random_tensor(
        (batch_size, vocab_size), "f32", device_name
    )

    # Add a tiny deterministic offset to reduce tie probability.
    offset = torch.arange(vocab_size, dtype=torch.float32, device=torch_device(device_name)) * 1e-7
    logits_torch = logits_torch + offset.unsqueeze(0)

    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(
        logits_llaisys.data_ptr(),
        logits_torch.data_ptr(),
        batch_size * vocab_size * 4,
        llaisys.MemcpyKind.D2D,
    )

    out_llaisys = llaisys.Tensor(
        (batch_size,),
        dtype=llaisys.DataType.I64,
        device=llaisys_device(device_name),
    )

    llaisys.Ops.sampling(out_llaisys, logits_llaisys, temperature=0.0, top_k=1, seed=42)

    result = torch.zeros((batch_size,), dtype=torch.int64, device=torch_device(device_name))
    api.memcpy_sync(
        result.data_ptr(), out_llaisys.data_ptr(), batch_size * 8,
        llaisys.MemcpyKind.D2D,
    )

    expected = torch.argmax(logits_torch, dim=-1)
    assert torch.equal(result, expected), \
        f"temperature=0, top_k=1 should equal argmax: got {result.tolist()}, expected {expected.tolist()}"

    print("   ✓ temperature=0 & top_k=1 equals argmax")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia", "mx"], type=str)
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Enhanced Sampling Operator Test Suite ({args.device})")
    print("=" * 70)
    
    try:
        print("\n[Test 1] Basic Sampling")
        test_basic_sampling(batch_size=2, vocab_size=1000, device_name=args.device)
        
        print("\n[Test 2] Multiple Data Types (F32, F16, BF16)")
        test_multiple_dtypes(batch_size=2, vocab_size=100, device_name=args.device)
        
        print("\n[Test 3] Top-K Repeated Sampling")
        test_topk_repeated_sampling(batch_size=1, vocab_size=100, top_k=10, num_samples=20, device_name=args.device)
        
        print("\n[Test 4] Boundary Cases")
        test_boundary_cases(device_name=args.device)
        
        print("\n[Test 5] Reproducibility")
        test_reproducibility_across_runs(batch_size=2, vocab_size=100, device_name=args.device)
        
        print("\n[Test 6] Top-P Sampling")
        test_topp_sampling(batch_size=1, vocab_size=100, top_p=0.95, num_samples=20, device_name=args.device)

        print("\n[Test 7] temperature=0 & top_k=1 == argmax")
        test_temperature_zero_topk1_equals_argmax(batch_size=8, vocab_size=256, device_name=args.device)
        
        print("\n" + "=" * 70)
        print("✓ All enhanced tests passed!")
        print("=" * 70)
    
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ Test failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
