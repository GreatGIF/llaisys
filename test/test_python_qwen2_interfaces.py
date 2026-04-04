import os
import tempfile
from pathlib import Path

import pytest

import llaisys


def make_minimal_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "fake_qwen2"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """{
  "num_hidden_layers": 1,
  "hidden_size": 4,
  "num_attention_heads": 2,
  "num_key_value_heads": 1,
  "intermediate_size": 4,
  "max_position_embeddings": 32,
  "vocab_size": 32,
  "rms_norm_eps": 1e-6,
  "rope_theta": 10000.0,
  "eos_token_id": 31,
  "tie_word_embeddings": false
}""",
        encoding="utf-8",
    )
    return model_dir


def test_python_interfaces_exist():
    assert hasattr(llaisys.models, "Qwen2Session")
    assert hasattr(llaisys.models, "Qwen2DynamicBatchEngine")


def test_python_qwen2_session_constructor(monkeypatch, tmp_path):
    model_dir = make_minimal_model_dir(tmp_path)

    class DummySafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return []

    monkeypatch.setattr("safetensors.safe_open", lambda *args, **kwargs: DummySafeOpen())

    model = llaisys.models.Qwen2(str(model_dir), llaisys.DeviceType.CPU)
    session = llaisys.models.Qwen2Session(model)
    assert session is not None


def test_python_dynamic_batch_engine_constructor(monkeypatch, tmp_path):
    model_dir = make_minimal_model_dir(tmp_path)

    class DummySafeOpen:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def keys(self):
            return []

    monkeypatch.setattr("safetensors.safe_open", lambda *args, **kwargs: DummySafeOpen())

    engine = llaisys.models.Qwen2DynamicBatchEngine(
        str(model_dir),
        llaisys.DeviceType.CPU,
        max_num_seqs=2,
        max_num_batched_tokens=8,
    )
    assert engine is not None
