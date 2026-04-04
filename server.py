"""
HTTP Server for LLM inference with OpenAI Chat Completion API compatibility.
Supports both streaming and non-streaming modes.
"""

import os
import sys
import io
import time
import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from threading import Lock, Thread
from queue import Queue, Empty
from dataclasses import dataclass
from contextlib import asynccontextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# ============================================================================
# Helper Functions
# ============================================================================

def get_llaisys_device(device_name: str):
    """Convert device name string to llaisys device type."""
    try:
        import llaisys
        if device_name == "cpu":
            return llaisys.DeviceType.CPU
        elif device_name == "nvidia":
            return llaisys.DeviceType.NVIDIA
        else:
            raise ValueError(f"Unsupported device name: {device_name}")
    except ImportError:
        raise ImportError("llaisys not installed")


# ============================================================================
# Data Models (OpenAI-compatible)
# ============================================================================

class ChatMessage(BaseModel):
    """Message in a chat conversation."""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    """Chat completion request following OpenAI API format."""
    model: str = Field(default="qwen-1.5b", description="Model name")
    messages: List[ChatMessage] = Field(..., description="List of messages")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    seed: int = Field(default=0, ge=0, description="Sampling seed (0 = non-deterministic)")
    max_tokens: int = Field(default=128, ge=1, le=4096)
    stream: bool = Field(default=False, description="Whether to stream the response")


class ChatCompletionChoice(BaseModel):
    """Choice in a chat completion response."""
    index: int
    message: ChatMessage
    finish_reason: str = Field(default="stop")


class ChatCompletionResponse(BaseModel):
    """Non-streaming chat completion response (OpenAI compatible)."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(default_factory=dict)


class ChatCompletionStreamChoice(BaseModel):
    """Choice in a streaming chat completion response."""
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """Streaming chat completion response chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


# ============================================================================
# Model Management
# ============================================================================

class ModelManager:
    """Manages model loading and inference."""

    def __init__(
        self,
        device: str = "cpu",
        model_path: Optional[str] = None,
        backend: str = "pytorch",
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 50,
        seed: int = 0,
    ):
        self.device = device
        self.model_path = model_path
        self.backend = backend
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.tokenizer = None
        self.model = None
        self.llaisys_session = None
        self.llaisys_batch_engine = None
        self.device_map = self._get_device_map()
        self.cached_token_ids = []
        self._llaisys_lock = Lock()
        self._llaisys_use_dynamic_batch = False
        self._llaisys_worker = None
        self._llaisys_request_queue = Queue()
        self._llaisys_pending = {}
        self._llaisys_stop = False
        self._llaisys_batch_wait_ms = float(os.environ.get("LLAISYS_BATCH_WAIT_MS", "5"))
        self._llaisys_batch_max_queue = int(os.environ.get("LLAISYS_BATCH_MAX_QUEUE", "32"))
        self._llaisys_request_timeout_s = float(os.environ.get("LLAISYS_REQUEST_TIMEOUT_S", "30"))

    def _get_device_map(self) -> str | dict:
        """Get appropriate device mapping for the model."""
        if self.device == "cpu":
            return "cpu"
        elif self.device == "nvidia":
            return "auto"  # auto device map for CUDA
        return "cpu"

    def load_model(self, model_id: str = "Qwen/Qwen2.5-1.5B"):
        """Load tokenizer and model."""
        if self.tokenizer is not None and self.model is not None:
            return  # Already loaded

        print(f"Loading model: {model_id}")
        print(f"Backend: {self.backend}")

        if self.model_path and os.path.isdir(self.model_path):
            print(f"Loading from local path: {self.model_path}")
            model_path = self.model_path
        else:
            print(f"Downloading from Hugging Face Hub: {model_id}")
            model_path = snapshot_download(model_id)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        if self.backend == "pytorch":
            # Load PyTorch model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                device_map=self.device_map,
                trust_remote_code=True,
            )
            print("PyTorch model loaded successfully")
        elif self.backend == "llaisys":
            # Load LLAISYS model
            import llaisys
            llaisys_device_type = get_llaisys_device(self.device)
            self.model = llaisys.models.Qwen2(model_path, llaisys_device_type)
            self.llaisys_session = llaisys.models.Qwen2Session(self.model)
            self.llaisys_use_dynamic_batch = os.environ.get("LLAISYS_DYNAMIC_BATCH", "0") == "1"
            if self.llaisys_use_dynamic_batch:
                max_num_seqs = int(os.environ.get("LLAISYS_MAX_NUM_SEQS", "32"))
                max_num_batched_tokens = int(os.environ.get("LLAISYS_MAX_NUM_BATCHED_TOKENS", "1024"))
                self.llaisys_batch_engine = llaisys.models.Qwen2DynamicBatchEngine(
                    model_path,
                    llaisys_device_type,
                    max_num_seqs=max_num_seqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                )
                self._llaisys_worker = Thread(target=self._llaisys_batch_loop, daemon=True)
                self._llaisys_worker.start()
            print("LLAISYS model loaded successfully")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def shutdown(self):
        self._llaisys_stop = True
        if self._llaisys_worker is not None and self._llaisys_worker.is_alive():
            self._llaisys_worker.join(timeout=1.0)

    def _llaisys_batch_loop(self):
        while not self._llaisys_stop:
            if not self.llaisys_batch_engine:
                time.sleep(0.01)
                continue
            collected = []
            try:
                request = self._llaisys_request_queue.get(timeout=0.01)
                collected.append(request)
                deadline = time.time() + self._llaisys_batch_wait_ms / 1000.0
                while len(collected) < self._llaisys_batch_max_queue:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        collected.append(self._llaisys_request_queue.get(timeout=remaining))
                    except Empty:
                        break
            except Empty:
                pass

            try:
                for request in collected:
                    seq_id = self.llaisys_batch_engine.add_request(
                        request.input_ids,
                        max_completion_tokens=request.max_completion_tokens,
                        temperature=request.temperature,
                        top_k=request.top_k,
                        top_p=request.top_p,
                        seed=request.seed,
                        ignore_eos=request.ignore_eos,
                    )
                    self._llaisys_pending[seq_id] = request

                if self.llaisys_batch_engine.is_finished():
                    continue

                finished = self.llaisys_batch_engine.step()
                for item in finished:
                    req = self._llaisys_pending.pop(item["seq_id"], None)
                    if req is not None:
                        req.result_queue.put(("ok", item["token_ids"]))
            except Exception as exc:
                for request in collected:
                    request.result_queue.put(("error", exc))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        stream: bool = False,
    ):
        """Generate text completion.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (uses default if None)
            top_p: Top-p sampling parameter (uses default if None)
            top_k: Top-k sampling parameter (uses default if None)
            seed: Sampling seed (uses default if None; 0 = non-deterministic)
            stream: Whether to stream tokens as they are generated

        Yields/Returns:
            For streaming: generator of token strings
            For non-streaming: full generated text
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Use provided values or fall back to defaults
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        top_k = top_k if top_k is not None else self.top_k
        seed = seed if seed != 0 else self.seed

        if self.backend == "pytorch":
            return self._generate_pytorch(
                prompt, max_new_tokens, temperature, top_p, top_k, seed, stream
            )
        elif self.backend == "llaisys":
            return self._generate_llaisys(
                prompt, max_new_tokens, temperature, top_p, top_k, seed, stream
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _generate_non_streaming(self, inputs, max_new_tokens, temperature, top_p, top_k, seed):
        """Generate all tokens at once."""
        if seed > 0:
            torch.manual_seed(seed)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p,
                temperature=temperature,
                do_sample=temperature > 0.0,  # Enable sampling if temperature > 0
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text

    def _generate_streaming(self, inputs, max_new_tokens, temperature, top_p, top_k, seed):
        """Generate tokens one by one (streaming)."""
        from transformers import TextIteratorStreamer
        from threading import Thread

        if seed > 0:
            torch.manual_seed(seed)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_special_tokens=True, skip_prompt=True
        )

        generation_kwargs = {
            "inputs": inputs,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k if top_k > 0 else None,
            "top_p": top_p,
            "temperature": temperature,
            "do_sample": temperature > 0.0,  # Enable sampling if temperature > 0
            "streamer": streamer,
        }

        # Run generation in a thread so we can stream the output
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Yield tokens as they become available
        for text in streamer:
            yield text

    def _generate_pytorch(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int, seed: int, stream: bool
    ):
        """Generate using PyTorch backend."""
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        if stream:
            return self._generate_streaming(
                inputs, max_new_tokens, temperature, top_p, top_k, seed
            )
        else:
            return self._generate_non_streaming(
                inputs, max_new_tokens, temperature, top_p, top_k, seed
            )

    def _generate_llaisys(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int, seed: int, stream: bool
    ):
        """Generate using LLAISYS backend with KV cache reuse."""
        if self.llaisys_use_dynamic_batch and not stream:
            return self._generate_llaisys_dynamic_batch(
                prompt, max_new_tokens, temperature, top_p, top_k, seed
            )

        input_ids = self.tokenizer.encode(prompt)

        with self._llaisys_lock:
            if self.cached_token_ids and len(input_ids) >= len(self.cached_token_ids) and \
               input_ids[:len(self.cached_token_ids)] == self.cached_token_ids:
                model_inputs = input_ids[len(self.cached_token_ids):]
                print(f"KV cache matched. Reusing {len(self.cached_token_ids)} tokens, sending {len(model_inputs)} new tokens.")
            else:
                model_inputs = input_ids
                self.llaisys_session.reset()
                print("KV cache not matched or first request. Clearing cache.")

            self.cached_token_ids = input_ids.copy()

            if stream:
                def _token_generator():
                    curr_inputs = list(model_inputs)
                    for _ in range(max_new_tokens or 2048):
                        token_id = self.llaisys_session.infer(
                            curr_inputs,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            seed=seed,
                        )
                        token_id = int(token_id)
                        self.cached_token_ids.append(token_id)
                        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        yield token_text
                        if token_id == self.model.end_token:
                            break
                        curr_inputs = [token_id]

                return _token_generator()
            else:
                output_ids = list(model_inputs)
                curr_inputs = list(model_inputs)
                for _ in range(max_new_tokens or 2048):
                    token_id = self.llaisys_session.infer(
                        curr_inputs,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        seed=seed,
                    )
                    token_id = int(token_id)
                    output_ids.append(token_id)
                    self.cached_token_ids.append(token_id)
                    if token_id == self.model.end_token:
                        break
                    curr_inputs = [token_id]

                full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                return full_text

    def _generate_llaisys_dynamic_batch(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int, seed: int
    ):
        input_ids = self.tokenizer.encode(prompt)
        result_queue = Queue(maxsize=1)
        self._llaisys_request_queue.put(_BatchRequest(
            input_ids=list(input_ids),
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            ignore_eos=False,
            result_queue=result_queue,
        ))
        status, payload = result_queue.get(timeout=self._llaisys_request_timeout_s)
        if status == "error":
            raise RuntimeError(f"LLAISYS dynamic batch request failed: {payload}") from payload
        token_ids = payload
        return self.tokenizer.decode(input_ids + token_ids, skip_special_tokens=True)


@dataclass
class _BatchRequest:
    input_ids: List[int]
    max_completion_tokens: int
    temperature: float
    top_k: int
    top_p: float
    seed: int
    ignore_eos: bool
    result_queue: Queue


# ============================================================================
# FastAPI Application
# ============================================================================

model_manager: Optional[ModelManager] = None


def create_model_manager_from_env() -> ModelManager:
    device = os.environ.get("DEVICE", "cpu")
    model_path = os.environ.get("MODEL_PATH", None)
    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B")
    backend = os.environ.get("BACKEND", "pytorch")
    temperature = float(os.environ.get("TEMPERATURE", "0.8"))
    top_p = float(os.environ.get("TOP_P", "0.8"))
    top_k = int(os.environ.get("TOP_K", "50"))
    seed = int(os.environ.get("SEED", "0"))

    print(f"Initializing model manager")
    print(f"  device={device}, backend={backend}")
    print(f"  temperature={temperature}, top_p={top_p}, top_k={top_k}, seed={seed}")

    model_manager = ModelManager(
        device=device,
        model_path=model_path,
        backend=backend,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
    )
    model_manager.load_model(model_id=model_id)
    return model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_manager
    model_manager = create_model_manager_from_env()
    try:
        yield
    finally:
        if model_manager is not None:
            model_manager.shutdown()
        model_manager = None


app = FastAPI(title="LLM Chat API", version="1.0.0", lifespan=lifespan)


def get_model_manager() -> ModelManager:
    """Get or initialize the global model manager."""
    global model_manager
    if model_manager is None:
        raise RuntimeError("Model manager not initialized")
    return model_manager


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen-1.5b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "llaisys",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint (OpenAI compatible)."""
    manager = get_model_manager()

    # Build prompt from messages
    prompt = _build_prompt_from_messages(manager, request.messages)

    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(manager, request, prompt),
            media_type="text/event-stream",
        )
    else:
        return _non_streaming_chat_completion(manager, request, prompt)


def _non_streaming_chat_completion(
    manager: ModelManager, request: ChatCompletionRequest, prompt: str
):
    """Generate non-streaming chat completion."""
    # Generate response
    response_text = manager.generate(
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        seed=request.seed,
        stream=False,
    )

    # Create response
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    choice = ChatCompletionChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response_text),
        finish_reason="stop",
    )

    response = ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=request.model,
        choices=[choice],
        usage={
            "prompt_tokens": len(manager.tokenizer.encode(prompt)),
            "completion_tokens": len(manager.tokenizer.encode(response_text)),
            "total_tokens": len(manager.tokenizer.encode(prompt + response_text)),
        },
    )

    return response.model_dump()


async def _stream_chat_completion(
    manager: ModelManager, request: ChatCompletionRequest, prompt: str
):
    """Stream chat completion as server-sent events."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # Generate tokens
    for token in manager.generate(
        prompt=prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        seed=request.seed,
        stream=True,
    ):
        # Create streaming response chunk
        chunk = ChatCompletionStreamResponse(
            id=completion_id,
            created=created_time,
            model=request.model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta={"content": token},
                    finish_reason=None,
                )
            ],
        )

        yield f"data: {json.dumps(chunk.model_dump())}\n\n"

    # Send final chunk with finish_reason
    end_chunk = ChatCompletionStreamResponse(
        id=completion_id,
        created=created_time,
        model=request.model,
        choices=[ChatCompletionStreamChoice(index=0, delta={}, finish_reason="stop")],
    )

    yield f"data: {json.dumps(end_chunk.model_dump())}\n\n"
    yield "data: [DONE]\n\n"


def _build_prompt_from_messages(
    manager: ModelManager, messages: List[ChatMessage]
) -> str:
    """Build a prompt string from chat messages using the tokenizer's chat template."""
    conversation = [{"role": msg.role, "content": msg.content} for msg in messages]

    prompt = manager.tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        tokenize=False,
    )

    return prompt


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "LLM Chat Server",
        "version": "1.0.0",
        "documentation": "/docs",
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="LLM Chat Server")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Local model path or Hugging Face model ID",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pytorch", "llaisys"],
        default="pytorch",
        help="Inference backend (default: pytorch)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "nvidia", "mx"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling parameter (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed (default: 0, non-deterministic)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers (default: 1)",
    )

    args = parser.parse_args()

    # Set environment variables from command line arguments
    os.environ["DEVICE"] = args.device
    os.environ["BACKEND"] = args.backend
    os.environ["TEMPERATURE"] = str(args.temperature)
    os.environ["TOP_P"] = str(args.top_p)
    os.environ["TOP_K"] = str(args.top_k)
    os.environ["SEED"] = str(args.seed)
    if args.model:
        os.environ["MODEL_PATH"] = args.model

    print(f"Starting server on {args.host}:{args.port}")
    if args.model:
        print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Device: {args.device}")
    print(f"Sampling: temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, seed={args.seed}")
    uvicorn.run(app, host=args.host, port=args.port, workers=args.workers)
