"""
Client examples for LLM Chat Server.
Demonstrates both streaming and non-streaming requests.
"""

import requests
import json
import sys


def chat_non_streaming(
    base_url: str = "http://localhost:8000",
    user_message: str = "你好，请介绍一下你自己",
    model: str = "qwen-1.5b",
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 50,
    max_tokens: int = 256,
):
    """Send a non-streaming chat completion request."""
    print(f"\n{'='*60}")
    print(f"Non-streaming Chat Completion")
    print(f"{'='*60}")
    print(f"User: {user_message}\n")

    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()

        # Print response
        if result.get("choices"):
            assistant_message = result["choices"][0]["message"]["content"]
            print(f"Assistant: {assistant_message}")

        # Print usage
        if result.get("usage"):
            print(f"\nUsage:")
            print(f"  - Prompt tokens: {result['usage']['prompt_tokens']}")
            print(f"  - Completion tokens: {result['usage']['completion_tokens']}")
            print(f"  - Total tokens: {result['usage']['total_tokens']}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure it's running on", base_url)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


def chat_streaming(
    base_url: str = "http://localhost:8000",
    user_message: str = "你好，请介绍一下你自己",
    model: str = "qwen-1.5b",
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 50,
    max_tokens: int = 256,
):
    """Send a streaming chat completion request."""
    print(f"\n{'='*60}")
    print(f"Streaming Chat Completion")
    print(f"{'='*60}")
    print(f"User: {user_message}\n")
    print("Assistant: ", end="", flush=True)

    url = f"{base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "stream": True,
    }

    try:
        response = requests.post(url, json=payload, timeout=300, stream=True)
        response.raise_for_status()

        # Process streaming response
        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8") if isinstance(line, bytes) else line
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    if data.get("choices"):
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
                except json.JSONDecodeError:
                    pass

        print()  # Newline after streaming completes

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure it's running on", base_url)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


def list_models(base_url: str = "http://localhost:8000"):
    """List available models."""
    print(f"\n{'='*60}")
    print(f"Available Models")
    print(f"{'='*60}\n")

    try:
        response = requests.get(f"{base_url}/v1/models")
        response.raise_for_status()

        models = response.json().get("data", [])
        for model in models:
            print(f"- {model['id']}")

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


def health_check(base_url: str = "http://localhost:8000"):
    """Check server health."""
    print(f"\n{'='*60}")
    print(f"Health Check")
    print(f"{'='*60}\n")

    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()

        health = response.json()
        print(f"Status: {health.get('status')}")
        print(f"Timestamp: {health.get('timestamp')}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure it's running on", base_url)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Chat Server Client")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the server",
    )
    parser.add_argument(
        "--message",
        default="你好，请告诉我你的名字",
        help="User message to send",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k parameter",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--mode",
        choices=["health", "models", "stream", "non-stream", "all"],
        default="non-stream",
        help="Operating mode",
    )

    args = parser.parse_args()

    if args.mode in ["health", "all"]:
        health_check(args.base_url)

    if args.mode in ["models", "all"]:
        list_models(args.base_url)

    if args.mode in ["non-stream", "all"]:
        chat_non_streaming(
            base_url=args.base_url,
            user_message=args.message,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )

    if args.mode in ["stream", "all"]:
        chat_streaming(
            base_url=args.base_url,
            user_message=args.message,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
        )
