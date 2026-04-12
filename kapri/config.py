"""Kapri config auto-generation for llama-swap."""

import pathlib
from typing import Optional

import yaml
from rich.console import Console

console = Console()

from .constants import (
    BIN_DIR,
    CONFIG_FILE,
    LLAMASERVER_BIN,
    LLAMASWAP_BIN,
    MODEL_DIR,
    START_PORT,
)
from .models import get_local_models


def model_id_from_filename(filename: str, manifest: Optional[list[dict]] = None) -> str:
    """
    Extract model ID from filename.

    Prefers manifest ID lookup, fallback to filename parsing.
    """
    # Check manifest
    if manifest is None:
        manifest = get_local_models()

    for entry in manifest:
        if entry.get("filename") and filename.endswith(entry["filename"]):
            return entry["id"]

    # Fallback: strip .gguf, lowercase, replace special chars
    base = filename.replace(".gguf", "").lower()
    for char in "._- ":
        base = base.replace(char, "-")
    return base


def get_default_ctx(model_entry: dict) -> int:
    """Get safe default context size."""
    context = model_entry.get("context", 4096)
    return min(context, 32768)


def regenerate_config() -> None:
    """
    Generate llama-swap config.yaml from downloaded models.

    Structure:
    globalTTL: 900
    startPort: 5800
    healthCheckTimeout: 60

    models:
      <model-id>:
        cmd: > /path/to/llama-server --port ${PORT} ...
    """
    models = get_local_models()

    # Get binary path from kapri bin
    server_bin = BIN_DIR / LLAMASERVER_BIN
    if not server_bin.exists():
        # Try common locations
        for name in ["llama-server", "llama-server.exe"]:
            alt = BIN_DIR / name
            if alt.exists():
                server_bin = alt
                break

    server_path = (
        str(server_bin.resolve()) if server_bin.exists() else "/path/to/llama-server"
    )

    # Build config
    config = {
        "globalTTL": 900,
        "startPort": START_PORT,
        "healthCheckTimeout": 60,
        "models": {},
    }

    # Vulkan environment variables (required for GPU inference)
    vulkan_env = [
        "GGML_VK_NO_PIPELINE_CACHE=1",
        "VK_DISABLE_PIPELINE_CACHE=1",
        "GGML_VK_DISABLE_COOPMAT=1",
        "GGML_VK_DISABLE_COOPMAT2=1",
    ]

    for i, entry in enumerate(models):
        model_id = entry["id"]
        model_path = pathlib.Path(entry["path"])

        if not model_path.exists():
            continue

        # Model path is now a folder - find the main GGUF file
        gguf_files = list(model_path.glob("*.gguf"))
        if not gguf_files:
            console.print(f"[yellow]No GGUF files in:[/yellow] {model_path}")
            continue

        # Skip mmproj files
        gguf_files = [f for f in gguf_files if "mmproj" not in f.name.lower()]
        if not gguf_files:
            continue

        # Use first GGUF file as main model
        main_model = gguf_files[0]

        # Resolve path - use forward slashes for YAML
        resolved_path = str(main_model.resolve()).replace("\\", "/")

        # Check for mmproj
        mmproj_path = None
        mmproj_files = [
            f for f in model_path.glob("*.gguf") if "mmproj" in f.name.lower()
        ]
        if mmproj_files:
            mmproj_path = str(mmproj_files[0].resolve()).replace("\\", "/")

        ctx = get_default_ctx(entry)
        port = START_PORT + i

        # Build command with Vulkan-optimized flags and mmproj if available
        if mmproj_path:
            cmd = (
                f"{server_path} --port ${{PORT}} "
                f"--host 0.0.0.0 "
                f"--model {resolved_path} "
                f"--mmproj {mmproj_path} "
                f"--ctx-size {ctx} "
                f"--n-gpu-layers 99 "
                f"--parallel 1"
            )
        else:
            cmd = (
                f"{server_path} --port ${{PORT}} "
                f"--host 0.0.0.0 "
                f"--model {resolved_path} "
                f"--ctx-size {ctx} "
                f"--n-gpu-layers 99 "
                f"--parallel 1"
            )

        config["models"][model_id] = {
            "cmd": cmd,
            "env": vulkan_env,
        }

    # Write atomically
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = CONFIG_FILE.with_suffix(".yaml.tmp")

    with open(tmp_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Atomically replace
    tmp_file.replace(CONFIG_FILE)

    return config


def load_config() -> dict:
    """Load current config."""
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    regenerate_config()
    print(f"Config written to: {CONFIG_FILE}")
