import threading
import time

from server import ModelManager


class FakeTokenizer:
    def encode(self, text):
        return [ord(ch) - 96 for ch in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(token + 96) for token in token_ids)


class FakeBatchEngine:
    def __init__(self):
        self.pending = []
        self.finished = []
        self.next_id = 0
        self.step_batch_sizes = []

    def add_request(self, inputs, max_completion_tokens, temperature, top_k, top_p, seed, ignore_eos):
        seq_id = self.next_id
        self.next_id += 1
        self.pending.append((seq_id, list(inputs)))
        return seq_id

    def is_finished(self):
        return not self.pending

    def step(self):
        if not self.pending:
            return []
        batch = self.pending[:]
        self.pending.clear()
        self.step_batch_sizes.append(len(batch))
        return [{"seq_id": seq_id, "token_ids": [inputs[-1] + 1]} for seq_id, inputs in batch]


class FailingBatchEngine(FakeBatchEngine):
    def step(self):
        raise RuntimeError("boom")


def test_model_manager_dynamic_batch_path():
    manager = ModelManager(backend="llaisys")
    manager.tokenizer = FakeTokenizer()
    manager.llaisys_use_dynamic_batch = True
    manager.llaisys_batch_engine = FakeBatchEngine()
    manager._llaisys_worker = threading.Thread(target=manager._llaisys_batch_loop, daemon=True)
    manager._llaisys_worker.start()
    try:
        text = manager._generate_llaisys_dynamic_batch(
            prompt="ab",
            max_new_tokens=1,
            temperature=1.0,
            top_p=0.0,
            top_k=1,
            seed=1,
        )
        assert text == "abc"
    finally:
        manager.shutdown()


def test_model_manager_dynamic_batch_aggregates_requests():
    manager = ModelManager(backend="llaisys")
    manager.tokenizer = FakeTokenizer()
    manager.llaisys_use_dynamic_batch = True
    fake_engine = FakeBatchEngine()
    manager.llaisys_batch_engine = fake_engine
    manager._llaisys_worker = threading.Thread(target=manager._llaisys_batch_loop, daemon=True)
    manager._llaisys_worker.start()
    results = {}

    def run_request(name, prompt):
        results[name] = manager._generate_llaisys_dynamic_batch(
            prompt=prompt,
            max_new_tokens=1,
            temperature=1.0,
            top_p=0.0,
            top_k=1,
            seed=1,
        )

    try:
        t1 = threading.Thread(target=run_request, args=("a", "ab"))
        t2 = threading.Thread(target=run_request, args=("b", "cd"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["a"] == "abc"
        assert results["b"] == "cde"
        assert any(size >= 2 for size in fake_engine.step_batch_sizes)
    finally:
        manager.shutdown()


def test_model_manager_dynamic_batch_respects_batch_max_queue():
    manager = ModelManager(backend="llaisys")
    manager.tokenizer = FakeTokenizer()
    manager.llaisys_use_dynamic_batch = True
    manager._llaisys_batch_wait_ms = 50
    manager._llaisys_batch_max_queue = 1
    fake_engine = FakeBatchEngine()
    manager.llaisys_batch_engine = fake_engine
    manager._llaisys_worker = threading.Thread(target=manager._llaisys_batch_loop, daemon=True)
    manager._llaisys_worker.start()
    results = {}

    def run_request(name, prompt):
        results[name] = manager._generate_llaisys_dynamic_batch(
            prompt=prompt,
            max_new_tokens=1,
            temperature=1.0,
            top_p=0.0,
            top_k=1,
            seed=1,
        )

    try:
        t1 = threading.Thread(target=run_request, args=("a", "ab"))
        t2 = threading.Thread(target=run_request, args=("b", "cd"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["a"] == "abc"
        assert results["b"] == "cde"
        assert all(size == 1 for size in fake_engine.step_batch_sizes)
    finally:
        manager.shutdown()


def test_model_manager_dynamic_batch_propagates_worker_errors():
    manager = ModelManager(backend="llaisys")
    manager.tokenizer = FakeTokenizer()
    manager.llaisys_use_dynamic_batch = True
    manager._llaisys_request_timeout_s = 1
    manager.llaisys_batch_engine = FailingBatchEngine()
    manager._llaisys_worker = threading.Thread(target=manager._llaisys_batch_loop, daemon=True)
    manager._llaisys_worker.start()
    try:
        try:
            manager._generate_llaisys_dynamic_batch(
                prompt="ab",
                max_new_tokens=1,
                temperature=1.0,
                top_p=0.0,
                top_k=1,
                seed=1,
            )
            assert False, "expected runtime error"
        except RuntimeError as exc:
            assert "dynamic batch request failed" in str(exc)
    finally:
        manager.shutdown()
