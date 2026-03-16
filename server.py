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
from dataclasses import asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# Import new session management and KV-Cache pool modules
try:
    from llaisys.session_manager import SessionManager, Message, Conversation
    from llaisys.kv_cache_pool import KVCachePool
    from llaisys.qwen2_with_cache import Qwen2WithKVCachePool
    LLAISYS_MODULES_AVAILABLE = True
except ImportError:
    LLAISYS_MODULES_AVAILABLE = False
    print("[Warning] llaisys session management modules not available")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


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
# Session Management Models
# ============================================================================

class SessionMessage(BaseModel):
    """Message in a session"""
    role: str
    content: str
    token_ids: List[int] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


class CreateSessionRequest(BaseModel):
    """Request to create new session"""
    title: Optional[str] = Field(default=None, description="Session title")


class CreateSessionResponse(BaseModel):
    """Response from session creation"""
    session_id: str
    title: str
    created_at: float


class ListSessionsResponse(BaseModel):
    """Response for listing sessions"""
    sessions: List[Dict[str, Any]]
    total: int


class GetMessagesResponse(BaseModel):
    """Response for getting session messages"""
    session_id: str
    messages: List[Dict[str, Any]]


class SendMessageRequest(BaseModel):
    """Request to send message in session"""
    content: str
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    seed: int = Field(default=0, ge=0)
    stream: bool = Field(default=False)


class SendMessageResponse(BaseModel):
    """Response from sending message"""
    session_id: str
    user_message: str
    assistant_message: str
    stats: Dict[str, Any]


class UpdateMessageRequest(BaseModel):
    """Request to update message"""
    new_content: str
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    seed: int = Field(default=0, ge=0)


class UpdateMessageResponse(BaseModel):
    """Response from updating message"""
    session_id: str
    message_index: int
    assistant_message: str
    stats: Dict[str, Any]


class RegenerateRequest(BaseModel):
    """Request to regenerate from a message"""
    from_message_idx: int = Field(..., ge=0, description="Regenerate from this message index")
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    seed: int = Field(default=0, ge=0)


class RegenerateResponse(BaseModel):
    """Response from regenerate"""
    session_id: str
    assistant_message: str
    stats: Dict[str, Any]


class CacheStatsResponse(BaseModel):
    """Cache statistics"""
    cache_stats: Dict[str, Any]
    session_stats: Dict[str, Any]


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
        self.device_map = self._get_device_map()
        self.cached_token_ids = []
        
        # Initialize session management and KV-Cache pool
        if LLAISYS_MODULES_AVAILABLE:
            self.session_manager = SessionManager(max_sessions=100)
            self.kv_cache_pool = KVCachePool(max_cache_entries=100)
            self.qwen2_with_cache = None  # Will be initialized after model loading
        else:
            self.session_manager = None
            self.kv_cache_pool = None
            self.qwen2_with_cache = None

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
            print("LLAISYS model loaded successfully")
            
            # Initialize Qwen2 with KV-Cache pool support
            if LLAISYS_MODULES_AVAILABLE and self.session_manager and self.kv_cache_pool:
                self.qwen2_with_cache = Qwen2WithKVCachePool(
                    model_path, llaisys_device_type, self.session_manager, self.kv_cache_pool
                )
                self.qwen2_with_cache.tokenizer = self.tokenizer
                print("Qwen2WithKVCachePool initialized successfully")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

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
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)

        # Check for KV cache prefix match
        if self.cached_token_ids and len(input_ids) >= len(self.cached_token_ids) and \
           input_ids[:len(self.cached_token_ids)] == self.cached_token_ids:
            # Prefix matched, we can reuse KV cache
            clear_kv_cache = False
            # Only send the new tokens to the model
            model_inputs = input_ids[len(self.cached_token_ids):]
            print(f"KV cache matched. Reusing {len(self.cached_token_ids)} tokens, sending {len(model_inputs)} new tokens.")
        else:
            # No match or first request, reset cache
            clear_kv_cache = True
            model_inputs = input_ids
            print("KV cache not matched or first request. Clearing cache.")

        # IMPORTANT: Update cache list to include the newly sent tokens!
        self.cached_token_ids = input_ids.copy()

        if stream:
            # True streaming: backend yields token ids as they are generated.
            def _token_generator():
                for token_id in self.model.generate(
                    model_inputs,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    seed=seed,
                    stream=True,
                    clear_kv_cache=clear_kv_cache,
                ):
                    self.cached_token_ids.append(token_id)
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    yield token_text

            return _token_generator()
        else:
            output_ids = self.model.generate(
                model_inputs,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                seed=seed,
                stream=False,
                clear_kv_cache=clear_kv_cache,
            )
            # The output_ids includes the inputs we sent (model_inputs) + the new tokens.
            # We want to extract only the generated tokens to append to our cache.
            generated_ids = output_ids[len(model_inputs):]
            self.cached_token_ids.extend(generated_ids)
            
            full_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            return full_text


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="LLM Chat API", version="1.0.0")

# Global model manager
model_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get or initialize the global model manager."""
    global model_manager
    if model_manager is None:
        raise RuntimeError("Model manager not initialized")
    return model_manager


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model_manager
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
# Session Management API Endpoints
# ============================================================================

@app.post("/v1/sessions/create")
async def create_session(request: CreateSessionRequest):
    """Create a new conversation session"""
    manager = get_model_manager()
    if not manager.session_manager:
        raise HTTPException(status_code=503, detail="Session management not available")
    
    session_id = manager.session_manager.create_session(title=request.title)
    conv = manager.session_manager.get_session(session_id)
    
    return CreateSessionResponse(
        session_id=session_id,
        title=conv.title,
        created_at=conv.created_at,
    )


@app.get("/v1/sessions")
async def list_sessions():
    """List all sessions"""
    manager = get_model_manager()
    if not manager.session_manager:
        raise HTTPException(status_code=503, detail="Session management not available")
    
    sessions = manager.session_manager.list_sessions()
    return ListSessionsResponse(sessions=sessions, total=len(sessions))


@app.get("/v1/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """Get all messages in a session"""
    manager = get_model_manager()
    if not manager.session_manager:
        raise HTTPException(status_code=503, detail="Session management not available")
    
    conv = manager.session_manager.get_session(session_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = [asdict(msg) for msg in conv.messages]
    return GetMessagesResponse(session_id=session_id, messages=messages)


@app.post("/v1/sessions/{session_id}/message")
async def send_message(session_id: str, request: SendMessageRequest):
    """Send a message in a session and get AI response"""
    manager = get_model_manager()
    if not manager.session_manager:
        raise HTTPException(status_code=503, detail="Session management not available")
    
    conv = manager.session_manager.get_session(session_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if request.stream:
        # Return streaming response
        async def _stream_session_message():
            from dataclasses import asdict as dataclass_asdict
            
            # Add user message
            user_token_ids = manager.tokenizer.encode(request.content)
            user_msg = Message(role="user", content=request.content, token_ids=user_token_ids)
            manager.session_manager.add_message(session_id, user_msg)
            
            # Get full input token sequence
            input_token_ids = conv.get_full_token_ids()
            
            # Generate tokens
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created_time = int(time.time())
            
            assistant_content = ""
            
            if manager.qwen2_with_cache and manager.backend == "llaisys":
                # Use KV-Cache pool for streaming
                for token_text in manager.qwen2_with_cache.infer_streaming(
                    input_token_ids, session_id,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    seed=request.seed,
                ):
                    assistant_content += token_text
                    chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=created_time,
                        model="qwen-1.5b",
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": token_text},
                            finish_reason=None,
                        )],
                    )
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            else:
                # Fallback to standard generation
                for token_text in manager.generate(
                    prompt=_build_prompt_from_messages(manager, [ChatMessage(role=msg.role, content=msg.content) for msg in conv.messages] + [ChatMessage(role="user", content=request.content)]),
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    seed=request.seed,
                    stream=True,
                ):
                    assistant_content += token_text
                    chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=created_time,
                        model="qwen-1.5b",
                        choices=[ChatCompletionStreamChoice(
                            index=0,
                            delta={"content": token_text},
                            finish_reason=None,
                        )],
                    )
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            
            # Save assistant message
            output_tokens = manager.tokenizer.encode(assistant_content)
            assistant_msg = Message(role="assistant", content=assistant_content, token_ids=output_tokens)
            manager.session_manager.add_message(session_id, assistant_msg)
            
            # Send final chunk
            end_chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created_time,
                model="qwen-1.5b",
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={},
                    finish_reason="stop",
                )],
            )
            yield f"data: {json.dumps(end_chunk.model_dump())}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(_stream_session_message(), media_type="text/event-stream")
    else:
        # Non-streaming response
        if manager.qwen2_with_cache and manager.backend == "llaisys":
            # Use KV-Cache pool
            try:
                assistant_message, stats = manager.qwen2_with_cache.chat_completion(
                    session_id=session_id,
                    user_message=request.content,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                    seed=request.seed,
                )
                stats["kv_cache_enabled"] = True
            except Exception as e:
                print(f"Error in chat_completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        else:
            # Fallback to standard generation
            user_msg = Message(role="user", content=request.content, token_ids=manager.tokenizer.encode(request.content))
            manager.session_manager.add_message(session_id, user_msg)
            
            prompt = _build_prompt_from_messages(manager, [ChatMessage(role=msg.role, content=msg.content) for msg in conv.messages + [user_msg]])
            
            assistant_message = manager.generate(
                prompt=prompt,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                seed=request.seed,
                stream=False,
            )
            
            output_tokens = manager.tokenizer.encode(assistant_message)
            assistant_msg = Message(role="assistant", content=assistant_message, token_ids=output_tokens)
            manager.session_manager.add_message(session_id, assistant_msg)
            
            stats = {
                "kv_cache_enabled": False,
                "output_tokens": len(output_tokens),
            }
        
        return SendMessageResponse(
            session_id=session_id,
            user_message=request.content,
            assistant_message=assistant_message,
            stats=stats,
        )


@app.put("/v1/sessions/{session_id}/message/{message_idx}")
async def update_message(session_id: str, message_idx: int, request: UpdateMessageRequest):
    """Update a message and regenerate from that point"""
    manager = get_model_manager()
    if not manager.session_manager:
        raise HTTPException(status_code=503, detail="Session management not available")
    
    conv = manager.session_manager.get_session(session_id)
    if not conv or message_idx >= len(conv.messages):
        raise HTTPException(status_code=404, detail="Session or message not found")
    
    if manager.qwen2_with_cache and manager.backend == "llaisys":
        # Use KV-Cache pool for regeneration
        try:
            assistant_message, stats = manager.qwen2_with_cache.update_and_regenerate(
                session_id=session_id,
                message_idx=message_idx,
                new_content=request.new_content,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                seed=request.seed,
            )
            stats["kv_cache_enabled"] = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=503, detail="KV-Cache not available")
    
    return UpdateMessageResponse(
        session_id=session_id,
        message_index=message_idx,
        assistant_message=assistant_message,
        stats=stats,
    )


@app.post("/v1/sessions/{session_id}/regenerate")
async def regenerate_from_message(session_id: str, request: RegenerateRequest):
    """Regenerate response from a specific message"""
    manager = get_model_manager()
    if not manager.session_manager:
        raise HTTPException(status_code=503, detail="Session management not available")
    
    conv = manager.session_manager.get_session(session_id)
    if not conv or request.from_message_idx >= len(conv.messages):
        raise HTTPException(status_code=404, detail="Session or message not found")
    
    if manager.qwen2_with_cache and manager.backend == "llaisys":
        # Use KV-Cache pool
        try:
            assistant_message, stats = manager.qwen2_with_cache.regenerate_from_message(
                session_id=session_id,
                from_message_idx=request.from_message_idx,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                seed=request.seed,
            )
            stats["kv_cache_enabled"] = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=503, detail="KV-Cache not available")
    
    return RegenerateResponse(
        session_id=session_id,
        assistant_message=assistant_message,
        stats=stats,
    )


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session"""
    manager = get_model_manager()
    if not manager.session_manager:
        raise HTTPException(status_code=503, detail="Session management not available")
    
    success = manager.session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clear KV-Cache for this session
    if manager.kv_cache_pool:
        manager.kv_cache_pool.invalidate_session_cache(session_id)
    
    return {"status": "success", "message": f"Session {session_id} deleted"}


@app.get("/v1/debug/cache-stats")
async def get_cache_stats():
    """Get KV-Cache pool statistics"""
    manager = get_model_manager()
    if not manager.kv_cache_pool:
        raise HTTPException(status_code=503, detail="KV-Cache not available")
    
    cache_stats = manager.kv_cache_pool.get_statistics()
    session_stats = manager.session_manager.get_statistics() if manager.session_manager else {}
    
    return CacheStatsResponse(
        cache_stats=cache_stats,
        session_stats=session_stats,
    )


@app.post("/v1/debug/cache-clear")
async def clear_cache():
    """Clear all KV-Cache"""
    manager = get_model_manager()
    if not manager.kv_cache_pool:
        raise HTTPException(status_code=503, detail="KV-Cache not available")
    
    count = manager.kv_cache_pool.clear_all()
    return {"status": "success", "cleared_entries": count}


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
