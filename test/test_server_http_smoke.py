import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

import server


class DummyTokenizer:
    def encode(self, text, return_tensors=None):
        if return_tensors is not None:
            raise NotImplementedError
        return [1, 2, 3]

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded"

    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False):
        return "prompt"


class DummyManager:
    def __init__(self):
        self.tokenizer = DummyTokenizer()

    def generate(self, prompt, max_new_tokens, temperature, top_p, top_k, seed, stream):
        if stream:
            def gen():
                yield "a"
                yield "b"
            return gen()
        return "hello"

    def shutdown(self):
        pass


def test_chat_completion_http_smoke(monkeypatch):
    monkeypatch.setattr(server, "create_model_manager_from_env", lambda: DummyManager())
    with TestClient(server.app) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen-1.5b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "max_tokens": 4,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "hello"


def test_chat_completion_stream_http_smoke(monkeypatch):
    monkeypatch.setattr(server, "create_model_manager_from_env", lambda: DummyManager())
    with TestClient(server.app) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwen-1.5b",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "max_tokens": 4,
            },
        ) as resp:
            assert resp.status_code == 200
            body = "".join(line for line in resp.iter_text())
            assert "hello" not in body
            assert "data:" in body
