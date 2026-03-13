"""
Advanced examples for LLM Chat Server.
Demonstrates various use cases and integration patterns.
"""

import requests
import json
import asyncio
from typing import Generator, AsyncGenerator


# ============================================================================
# Example 1: Context-aware Conversation
# ============================================================================

class ChatSession:
    """Maintains conversation context across multiple turns."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.conversation_history = []
        self.system_prompt = "You are a helpful AI assistant."

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def chat(self, user_message: str, stream: bool = False) -> str:
        """Send message and get response, maintaining conversation context."""
        self.add_message("user", user_message)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        payload = {
            "model": "qwen-1.5b",
            "messages": messages,
            "stream": stream,
            "max_tokens": 256,
        }

        url = f"{self.base_url}/v1/chat/completions"

        if stream:
            return self._stream_response(url, payload)
        else:
            response = requests.post(url, json=payload)
            result = response.json()
            assistant_message = result["choices"][0]["message"]["content"]
            self.add_message("assistant", assistant_message)
            return assistant_message

    def _stream_response(self, url: str, payload: dict) -> Generator[str, None, None]:
        """Stream response and update context."""
        full_response = ""

        response = requests.post(url, json=payload, stream=True)

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line.startswith("data: "):
                data_str = line[6:]

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    if data.get("choices"):
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            chunk = delta["content"]
                            full_response += chunk
                            yield chunk
                except json.JSONDecodeError:
                    pass

        # Add assistant response to history
        self.add_message("assistant", full_response)

    def print_history(self):
        """Print conversation history."""
        for msg in self.conversation_history:
            role = msg["role"].upper()
            content = msg["content"]
            print(f"{role}: {content}\n")


# ============================================================================
# Example 2: Streaming with Real-time Processing
# ============================================================================

def process_stream_realtime(user_message: str, base_url: str = "http://localhost:8000"):
    """Process streaming response with real-time token handling."""

    print("User: ", user_message)
    print("Assistant: ", end="", flush=True)

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "qwen-1.5b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
        "max_tokens": 256,
    }

    token_count = 0
    response = requests.post(url, json=payload, stream=True, timeout=300)

    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode("utf-8") if isinstance(line, bytes) else line
        if line.startswith("data: "):
            data_str = line[6:]

            if data_str == "[DONE]":
                print()  # Final newline
                print(f"[Generated {token_count} tokens]")
                break

            try:
                data = json.loads(data_str)
                if data.get("choices"):
                    delta = data["choices"][0].get("delta", {})
                    if "content" in delta:
                        print(delta["content"], end="", flush=True)
                        token_count += 1
            except json.JSONDecodeError:
                pass


# ============================================================================
# Example 3: Batch Processing
# ============================================================================

def batch_process(
    prompts: list[str],
    base_url: str = "http://localhost:8000",
    stream: bool = False,
) -> list[str]:
    """Process multiple prompts sequentially."""

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Processing: {prompt[:50]}...")

        payload = {
            "model": "qwen-1.5b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "max_tokens": 256,
        }

        url = f"{base_url}/v1/chat/completions"

        if stream:
            full_response = ""
            response = requests.post(url, json=payload, stream=True, timeout=300)

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)
                            if data.get("choices"):
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    full_response += delta["content"]
                        except json.JSONDecodeError:
                            pass

            results.append(full_response)
        else:
            response = requests.post(url, json=payload, timeout=300)
            result = response.json()
            response_text = result["choices"][0]["message"]["content"]
            results.append(response_text)

    return results


# ============================================================================
# Example 4: Temperature Sweep (为超参数调优)
# ============================================================================

def temperature_sweep(
    prompt: str,
    temperatures: list[float] = [0.1, 0.5, 0.8, 1.2, 1.8],
    base_url: str = "http://localhost:8000",
):
    """Test same prompt with different temperatures."""

    print(f"Prompt: {prompt}\n")
    print("=" * 60)

    for temp in temperatures:
        print(f"\nTemperature: {temp}")
        print("-" * 40)

        payload = {
            "model": "qwen-1.5b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "stream": False,
            "max_tokens": 128,
        }

        url = f"{base_url}/v1/chat/completions"
        response = requests.post(url, json=payload, timeout=300)
        result = response.json()
        print(result["choices"][0]["message"]["content"])


# ============================================================================
# Example 5: Question Answering with Context
# ============================================================================

class QASystem:
    """Simple Q&A system with context documents."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.documents = []

    def add_document(self, doc_id: str, content: str):
        """Add a document to the knowledge base."""
        self.documents.append({"id": doc_id, "content": content})

    def answer_question(self, question: str, use_documents: bool = True) -> str:
        """Answer a question using available documents."""

        context = ""
        if use_documents:
            # In a real system, you'd use semantic search to find relevant docs
            context = "\n".join([f"Document: {d['content']}" for d in self.documents[0:3]])

        system_prompt = f"""You are a helpful Q&A assistant.
Answer the user's question based on the provided documents.

{context}"""

        payload = {
            "model": "qwen-1.5b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "stream": False,
            "max_tokens": 256,
        }

        url = f"{self.base_url}/v1/chat/completions"
        response = requests.post(url, json=payload, timeout=300)
        result = response.json()
        return result["choices"][0]["message"]["content"]


# ============================================================================
# Example 6: Error Handling and Retry Logic
# ============================================================================

def request_with_retry(
    payload: dict,
    base_url: str = "http://localhost:8000",
    max_retries: int = 3,
    timeout: int = 300,
):
    """Make request with automatic retry on failure."""

    url = f"{base_url}/v1/chat/completions"

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            if attempt == max_retries:
                raise RuntimeError(f"Request timed out after {max_retries} attempts")
            print(f"Attempt {attempt}: Timeout, retrying...")

        except requests.exceptions.ConnectionError:
            if attempt == max_retries:
                raise RuntimeError(f"Connection failed after {max_retries} attempts")
            print(f"Attempt {attempt}: Connection error, retrying...")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")

    raise RuntimeError("Failed after all retries")


# ============================================================================
# Main Examples
# ============================================================================

if __name__ == "__main__":
    print("LLM Chat Server - Advanced Examples\n")

    # Example 1: Context-aware Conversation
    print("=" * 60)
    print("Example 1: Context-aware Conversation")
    print("=" * 60)

    session = ChatSession()
    print(session.chat("你好，你叫什么名字？"))
    print(session.chat("你能做什么？"))
    print(session.chat("用中文回答前面的问题"))

    print("\n\nConversation History:")
    session.print_history()

    # Example 2: Streaming with Real-time Processing
    print("\n" + "=" * 60)
    print("Example 2: Streaming with Real-time Processing")
    print("=" * 60)
    process_stream_realtime("请告诉我机器学习是什么？")

    # Example 3: Batch Processing
    print("\n\n" + "=" * 60)
    print("Example 3: Batch Processing")
    print("=" * 60)

    prompts = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
    ]
    results = batch_process(prompts[:1])  # 只运行第一个以节省时间
    for prompt, result in zip(prompts[:1], results):
        print(f"\nQ: {prompt}")
        print(f"A: {result[:100]}...")

    # Example 4: Q&A System
    print("\n" + "=" * 60)
    print("Example 4: Q&A System")
    print("=" * 60)

    qa = QASystem()
    qa.add_document("doc1", "Machine learning is a subset of artificial intelligence...")
    qa.add_document("doc2", "Deep learning uses neural networks with multiple layers...")

    answer = qa.answer_question("What is machine learning?")
    print(f"Q: What is machine learning?")
    print(f"A: {answer}")

    print("\n\nAll examples completed!")
