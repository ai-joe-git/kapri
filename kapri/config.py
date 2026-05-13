"""Kapri config auto-generation for llama.cpp router mode."""

import pathlib
import configparser
import io
from typing import Optional

from rich.console import Console

console = Console()

from .constants import (
    BIN_DIR,
    MODELS_INI,
    LLAMASERVER_BIN,
    MODEL_DIR,
)
from .models import get_local_models


def model_id_from_filename(filename: str, manifest: Optional[list[dict]] = None) -> str:
    """
    Extract model ID from filename.

    Prefers manifest ID lookup, fallback to filename parsing.
    """
    if manifest is None:
        manifest = get_local_models()

    for entry in manifest:
        if entry.get("filename") and filename.endswith(entry["filename"]):
            return entry["id"]

    base = filename.replace(".gguf", "").lower()
    for char in "._- ":
        base = base.replace(char, "-")
    return base


def get_default_ctx(model_entry: dict) -> int:
    """Get safe default context size."""
    context = model_entry.get("context", 4096)
    return min(context, 32768)


def regenerate_config() -> configparser.ConfigParser:
    """
    Generate models.ini for llama.cpp router mode.

    Structure:
      version = 1

      [model-id]
      model = /path/to/file.gguf
      n-gpu-layers = 99
      ctx-size = 32768
      ...
    """
    models = get_local_models()
    config = configparser.ConfigParser()

    config.add_section("DEFAULT")
    config.set("DEFAULT", "version", "1")

    server_bin = BIN_DIR / LLAMASERVER_BIN
    if not server_bin.exists():
        for name in ["llama-server", "llama-server.exe"]:
            alt = BIN_DIR / name
            if alt.exists():
                server_bin = alt
                break

    for entry in models:
        model_id = entry["id"]
        model_path = pathlib.Path(entry["path"])

        if not model_path.exists():
            continue

        gguf_files = list(model_path.parent.glob("*.gguf")) if not model_path.parent == model_path else [model_path]
        if not gguf_files:
            continue

        gguf_files = [f for f in gguf_files if "mmproj" not in f.name.lower()]
        if not gguf_files:
            continue

        main_model = model_path if model_path.is_file() else gguf_files[0]
        resolved_path = str(main_model.resolve())

        mmproj_path = None
        if model_path.parent.exists():
            mmproj_files = [
                f for f in model_path.parent.glob("*.gguf") if "mmproj" in f.name.lower()
            ]
            if mmproj_files:
                mmproj_path = str(mmproj_files[0].resolve())

        config.add_section(model_id)
        config.set(model_id, "model", resolved_path)
        config.set(model_id, "n-gpu-layers", "99")
        config.set(model_id, "ctx-size", str(get_default_ctx(entry)))
        config.set(model_id, "parallel", "1")
        config.set(model_id, "no-warmup", "true")
        config.set(model_id, "jinja", "true")

        if mmproj_path:
            config.set(model_id, "mmproj", mmproj_path)

    MODELS_INI.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_INI, "w", encoding="utf-8") as f:
        config.write(f, space_around_delimiters=False)

    return config


def load_config() -> configparser.ConfigParser:
    """Load current models.ini config."""
    config = configparser.ConfigParser()
    if MODELS_INI.exists():
        config.read(MODELS_INI, encoding="utf-8")
    return config


MODELS_PRESET_FILE = MODELS_INI

CONFIG_FILE = MODELS_INI


if __name__ == "__main__":
    regenerate_config()
    print(f"Config written to: {MODELS_INI}")
