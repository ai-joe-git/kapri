# Kapri

<p align="center">
  <strong>Run AI locally. Beautifully.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/kapri-ai/"><img src="https://img.shields.io/pypi/v/kapri-ai?color=4ADE80" alt="PyPI"></a>
  <a href="https://github.com/ai-joe-git/kapri/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/kapri-ai" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/kapri-ai" alt="Python"></a>
  <a href="https://github.com/ai-joe-git/kapri/releases"><img src="https://img.shields.io/github/v/release/kapri-ai/kapri" alt="GitHub release"></a>
</p>

---

## Why Kapri?

Kapri is a complete drop-in replacement for [Ollama](https://ollama.com). Built on **llama.cpp** + **llama-swap**, it gives you full control with zero cloud dependency.

### Key Features

- **Full GPU Support** — Vulkan (AMD), CUDA (NVIDIA), ROCm (AMD Linux), SYCL (Intel), Metal (Apple Silicon), CPU
- **Any GGUF Model** — Pull directly from HuggingFace, not limited to Ollama's registry
- **Transparent Config** — Plain YAML you can read and edit, no hidden internals  
- **Multi-Model Hot-Swap** — llama-swap with TTL-based auto-unload
- **Universal Install** — `pip install kapri-ai`, works everywhere Python exists

### Kapri vs Ollama

| Feature | Kapri | Ollama |
|---------|------|-------|
| Vulkan Support | Native | Blocked |
| Any HuggingFace Model | Yes | Registry only |
| Full llama-server Flags | Complete | Abstracted |
| Transparent Config | YAML | Hidden |
| Multi-model Hot-swap | Yes | Partial |
| Universal Install | pip | Platform-specific |

---

## Quick Start

```bash
# Install
pip install kapri-ai

# Setup (interactive wizard guides you)
kapri install

# Or use custom paths
kapri install --llama-server /path/to/llama-server --llama-swap /path/to/llama-swap --import-config /path/to/config.yaml

# Pull a model
kapri pull qwen3.5-0.8b

# Start server
kapri serve

# Open web UI chat
kapri run qwen3.5-0.8b

# Or use terminal chat (HTTP through llama-swap)
kapri run qwen3.5-0.8b --tui

# API available at http://localhost:11434
```

Then use with any OpenAI-compatible client:

```bash
curl http://localhost:11434/v1/models
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5-coder", "messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## Pulling Models

Kapri supports multiple ways to pull models:

### 1. From Built-in Registry

```bash
# Default Q4_K_M quantization
kapri pull llama3.2-3b

# Specify quantization
kapri pull llama3.2-3b:Q5_K_M

# List available models
kapri search llama
```

### 2. Direct from HuggingFace

Pull any GGUF model directly from HuggingFace:

```bash
# Unsloth models
kapri pull unsloth/Qwen3.5-0.8B-GGUF
kapri pull unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M

# Bartowski quantized models
kapri pull bartowski/Llama-3.2-3B-Instruct-GGUF
kapri pull bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M

# ggml-org TinyLlama
kapri pull ggml-org/TinyLlama-1.1B-Chat-v1.0-GGUF

# Any other GGUF repo
kapri pull <hf-repo>/<model-name>-GGUF
kapri pull <hf-repo>/<model-name>-GGUF:Q4_K_M
```

**Examples of GGUF repos on HuggingFace:**
- [unsloth/Qwen3.5-0.8B-GGUF](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF)
- [bartowski/Llama-3.2-3B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF)
- [bartowski/Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF)
- [ggml-org/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/ggml-org/TinyLlama-1.1B-Chat-v1.0-GGUF)

---

## Supported Backends

| Backend | Description | GPU Brands |
|---------|------------|----------|
| CUDA | NVIDIA GPUs | NVIDIA |
| Vulkan | AMD GPUs (full speed) | AMD |
| ROCm | AMD GPUs (Linux) | AMD |
| SYCL | Intel GPUs | Intel |
| Metal | Apple Silicon | Apple |
| CPU | Fallback | Any |

### Forcing a Backend

```bash
kapri install --backend cuda    # Force CUDA
kapri install --backend vulkan  # Force Vulkan
kapri install --backend cpu    # Force CPU
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `kapri install` | Install llama-server and llama-swap binaries |
| `kapri pull <model>` | Download a model |
| `kapri serve` | Start the server |
| `kapri stop` | Stop the server |
| `kapri status` | Show server status |
| `kapri list` | List downloaded models |
| `kapri search <query>` | Search registry |
| `kapri remove <model>` | Remove a model |
| `kapri run <model>` | Open web UI chat (default) |
| `kapri run <model> --tui` | Open terminal chat |
| `kapri config show` | Show config |
| `kapri logs` | View logs |

---

## API Endpoints

The server exposes OpenAI-compatible endpoints:

- `GET /v1/models` — List models
- `POST /v1/chat/completions` — Chat completions
- `POST /v1/completions` — Text completions

### Python Example

```python
import httpx

client = httpx.Client(base_url="http://localhost:11434")
response = client.post("/v1/chat/completions", json={
    "model": "qwen2.5-coder",
    "messages": [{"role": "user", "content": "Write a Python hello world"}]
})
print(response.json())
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"
)

chat = client.chat.completions.create(
    model="qwen2.5-coder",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(chat.choices[0].message.content)
```

---

## Architecture

```
~/.kapri/                    # Base directory
├── bin/                    # llama-server, llama-swap binaries
├── models/                # Downloaded GGUF files
├── config.yaml            # llama-swap configuration
├── server.pid            # Server process ID
└── server.log            # Server logs
```

---

## Troubleshooting

### No GPU detected

```bash
kapri install --backend cuda    # Force CUDA
kapri install --backend vulkan  # Force Vulkan (AMD)
```

### Port in use

```bash
kapri serve --port 11435  # Different port
```

### Model not found

Make sure the GGUF file exists on HuggingFace. Try:
```bash
# Use full repo path
kapri pull unsloth/Qwen3.5-0.8B-GGUF
```

---

## Requirements

- Python 3.10+
- No root/sudo required
- GPU optional (CPU fallback works)

---

## Links

- **GitHub:** https://github.com/ai-joe-git/kapri
- **PyPI:** https://pypi.org/project/kapri-ai/
- **Website:** https://kapri.vercel.app
- **Registry API:** https://kapri-registry.vercel.app/models.json

---

<p align="center">Built on llama.cpp + llama-swap</p>