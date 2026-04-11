"""Kapri constants and paths."""

import platform
import pathlib

APP_NAME = "kapri"
PYPI_NAME = "kapri-ai"
VERSION = "0.1.0"
DEFAULT_PORT = 11434  # Ollama-compatible drop-in
START_PORT = 5800  # llama-swap internal model ports
REGISTRY_URL = "https://kapri-registry.vercel.app/models.json"
REGISTRY_CACHE_TTL = 3600  # 1 hour in seconds
LLAMACPP_REPO = "ggml-org/llama.cpp"
LLAMASWAP_REPO = "mostlygeek/llama-swap"

# Cross-platform base directory
_system = platform.system()
if _system == "Windows":
    import os

    BASE_DIR = pathlib.Path(os.environ.get("APPDATA", "~")) / "kapri"
else:
    BASE_DIR = pathlib.Path.home() / ".kapri"

BIN_DIR = BASE_DIR / "bin"
MODEL_DIR = BASE_DIR / "models"
CONFIG_FILE = BASE_DIR / "config.yaml"
PID_FILE = BASE_DIR / "server.pid"
LOG_FILE = BASE_DIR / "server.log"
VERSIONS_FILE = BASE_DIR / "versions.json"
MODELS_MANIFEST = BASE_DIR / "models.json"
REGISTRY_CACHE = BASE_DIR / "registry_cache.json"

# Binary names per platform
LLAMASERVER_BIN = "llama-server.exe" if _system == "Windows" else "llama-server"
LLAMASWAP_BIN = "llama-swap.exe" if _system == "Windows" else "llama-swap"
LLAMA_CLI_BIN = "llama-cli.exe" if _system == "Windows" else "llama-cli"
