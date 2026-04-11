# kapri

<p align="center">
  <img src="https://raw.githubusercontent.com/kapri-ai/kapri/main/.github/logo.svg" alt="kapri logo" width="120">
</p>

<p align="center">
  <strong>Run AI locally. Beautifully.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/kapri-ai/"><img src="https://img.shields.io/pypi/v/kapri-ai?color=4ADE80" alt="PyPI"></a>
  <a href="https://github.com/kapri-ai/kapri/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/kapri-ai" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/kapri-ai" alt="Python"></a>
  <a href="https://github.com/kapri-ai/kapri/releases"><img src="https://img.shields.io/github/v/release/kapri-ai/kapri" alt="GitHub release"></a>
</p>

---

## Why Kapri?

Kapri is a complete drop-in replacement for Ollama. Built on **llama.cpp** + **llama-swap**, it gives you full control with zero cloud dependency.

| Feature | Kapri | Ollama |
|----------|------|--------|
| Vulkan Support | Native | Blocked |
| Any HuggingFace Model | Yes | Registry only |
| Full llama-server Flags | Complete | Abstracted |
| Transparent Config | YAML | Hidden |
| Multi-model Hot-swap | Yes | Partial |
| Universal Install | pip | Platform-specific |

## Supported Backends

| Backend | Description |
|---------|-------------|
| CUDA | NVIDIA GPUs |
| Vulkan | AMD GPUs (full speed) |
| ROCm | AMD GPUs (Linux) |
| SYCL | Intel GPUs |
| Metal | Apple Silicon (macOS) |
| CPU | Fallback |

## Quick Start

```bash
# Install
pip install kapri-ai

# Install binaries (auto-detects GPU)
kapri install

# Pull a model
kapri pull qwen2.5-coder

# Start server
kapri serve

# API available at http://localhost:11434
```

## Installation

### macOS / Linux / Windows

```bash
pip install kapri-ai
```

Or download binaries from [Releases](https://github.com/kapri-ai/kapri/releases).

## CLI Reference

| Command | Description |
|---------|-------------|
| `kapri install` | Install llama-server and llama-swap |
| `kapri pull <model>` | Download a model |
| `kapri serve` | Start the server |
| `kapri stop` | Stop the server |
| `kapri status` | Show server status |
| `kapri list` | List downloaded models |
| `kapri search <query>` | Search registry |
| `kapri remove <model>` | Remove a model |
| `kapri run <model>` | Interactive chat |
| `kapri config` | Manage config |
| `kapri logs` | View logs |

## API Endpoints

The server exposes OpenAI-compatible endpoints:

- `POST /v1/chat/completions` — Chat completions
- `POST /v1/completions` — Text completions
- `GET /v1/models` — List models

## Example Usage

### curl

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder",
    "messages": [{"role": "user", "content": "Write a Python hello world"}]
  }'
```

### Python

```python
import httpx

client = httpx.Client(base_url="http://localhost:11434")
response = client.post("/v1/chat/completions", json={
    "model": "qwen2.5-coder",
    "messages": [{"role": "user", "content": "Hello!"}]
})
print(response.json())
```

## Architecture

```
kapri/
├── bin/                # llama-server, llama-swap binaries
├── models/             # Downloaded GGUF files
├── config.yaml         # llama-swap configuration
├── server.pid          # Server process ID
└── server.log          # Server logs
```

## Requirements

- Python 3.10+
- No root/sudo required
- GPU with Vulkan/CUDA/ROCm/SYCL support (optional)

## Troubleshooting

### No GPU detected

```bash
# Force backend
kapri install --backend cuda
# or
kapri install --backend vulkan
```

### Port already in use

```bash
# Use different port
kapri serve --port 11435
```

## Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).

---

<p align="center">Built on llama.cpp + llama-swap</p>