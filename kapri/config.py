"""Kapri config auto-generation for llama.cpp router mode."""

import pathlib
import re
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
    context = model_entry.get("context", 4096)
    return min(context, 32768)


def regenerate_config() -> str:
    """
    Generate models.ini for llama.cpp router mode.

    Structure:
      version = 1

      [model-id]
      model = /path/to/file.gguf
      n-gpu-layers = 99
      ...
    """
    models = get_local_models()

    lines = ["version = 1", ""]

    for entry in models:
        model_id = entry["id"]
        model_path = pathlib.Path(entry["path"])

        model_dir = model_path.parent if not model_path.is_dir() else model_path

        gguf_files = [f for f in model_dir.glob("*.gguf") if "mmproj" not in f.name.lower()]
        if not gguf_files:
            continue

        # Prefer Dynamic (UD) quants — unsloth's older non-Dynamic quants produce garbage
        ud_files = [f for f in gguf_files if "-UD-" in f.name or "-UD_" in f.name]
        if ud_files:
            main_model = sorted(ud_files, key=lambda f: f.stat().st_size, reverse=True)[0]
        else:
            main_model = sorted(gguf_files, key=lambda f: f.stat().st_size, reverse=True)[0]
        resolved_path = str(main_model.resolve())

        mmproj_path = None
        mmproj_files = [f for f in model_dir.glob("*.gguf") if "mmproj" in f.name.lower()]
        if mmproj_files:
            mmproj_path = str(mmproj_files[0].resolve())

        lines.append(f"[{model_id}]")
        lines.append(f"model = {resolved_path}")
        lines.append("n-gpu-layers = 99")
        lines.append(f"ctx-size = {get_default_ctx(entry)}")
        lines.append("parallel = 1")
        lines.append("no-warmup = true")
        lines.append("jinja = true")
        lines.append("temp = 0.7")
        lines.append("top-p = 0.8")
        lines.append("top-k = 20")
        lines.append("min-p = 0.0")
        lines.append("reasoning = off")

        if mmproj_path:
            lines.append(f"mmproj = {mmproj_path}")

        lines.append("")

    content = "\n".join(lines)

    MODELS_INI.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_INI, "w", encoding="utf-8") as f:
        f.write(content)

    return content


def load_config() -> dict:
    """Parse models.ini into {section: {key: value}} dict."""
    return _parse_ini(MODELS_INI)


def _parse_ini(path: pathlib.Path) -> dict:
    """Parse INI file manually, handling top-level version line."""
    if not path.exists():
        return {}

    result = {}
    current_section = None

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        section_match = re.match(r"^\[(.+)\]$", line)
        if section_match:
            current_section = section_match.group(1)
            result[current_section] = {}
            continue

        kv_match = re.match(r"^(\S[^=]*?)\s*=\s*(.+)$", line)
        if kv_match and current_section:
            result[current_section][kv_match.group(1).strip()] = kv_match.group(2).strip()

    return result


MODELS_PRESET_FILE = MODELS_INI

CONFIG_FILE = MODELS_INI


if __name__ == "__main__":
    regenerate_config()
    print(f"Config written to: {MODELS_INI}")
