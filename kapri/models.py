"""Kapri model management - download, list, remove models."""

import json
import pathlib
from datetime import datetime
from typing import Optional

import httpx
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    DownloadColumn,
    TimeRemainingColumn,
)

from .constants import MODEL_DIR, MODELS_MANIFEST
from .registry import fetch_registry, resolve_model

console = Console()


def pull_model(
    model_ref: str, quant: str = "Q4_K_M", download_mmproj: bool = True
) -> dict:
    """
    Download a model from HuggingFace.

    For vision models with mmproj files, this automatically downloads the best available mmproj.
    Priority: mmproj-F32.gguf > mmproj-F16.gguf > mmproj-BF16.gguf

    Examples:
      # From registry
      kapri pull qwen3.5-0.8b
      kapri pull qwen3.5-0.8b:Q5_K_M

      # Direct from HuggingFace
      kapri pull unsloth/Qwen3.5-0.8B-GGUF
      kapri pull unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M
      kapri pull bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M

      # Skip mmproj download
      kapri pull unsloth/Qwen3.5-0.8B-GGUF --no-mmproj
    """
    from huggingface_hub import hf_hub_download

    # Resolve model
    # Handle: repo/format OR repo:filename format
    if "/" in model_ref:
        # Direct HF reference
        if ":" in model_ref:
            parts = model_ref.split(":")
            hf_repo = parts[0]
            filename = parts[1] if len(parts) > 1 else f"*{quant}*.gguf"
        else:
            hf_repo = model_ref
            # Try to find the right filename
            repo_name = hf_repo.split("/")[-1]
            filename = f"{repo_name.replace('-GGUF', '')}-{quant}.gguf"

        # Generate an ID from the repo
        model_id = hf_repo.replace("/", "-").replace("-GGUF", "").lower()
        registry_entry = None
    else:
        # Registry lookup
        registry_entry, filename = resolve_model(model_ref, quant)
        if registry_entry is None:
            raise ValueError(f"Model '{model_ref}' not found in registry")
        hf_repo = registry_entry["hf_repo"]
        model_id = registry_entry["id"]

    console.print(f"[bold]Pulling:[/bold] {model_id}")
    console.print(f"  Repository: {hf_repo}")
    console.print(f"  File: {filename}")

    # Get mmproj file if available (for vision models)
    mmproj_file = None
    mmproj_url = None
    if registry_entry and download_mmproj:
        mmproj_file = registry_entry.get("mmproj_file")
    if mmproj_file:
        # Try to find best mmproj (Q8_0 > BF16 > FP16)
        # mmproj files are usually small and always needed
        mmproj_priority = ["Q8_0", "BF16", "FP16"]
        for priority in mmproj_priority:
            try_mmproj = mmproj_file.replace("{quant}", priority)
            # Check if exists on HF - we'll try to download
            console.print(f"  [dim]Looking for mmproj: {try_mmproj}[/dim]")
            break  # We'll just try the named version

    # Ensure model directory - HF format: MODEL_DIR/repo/model-GGUF/
    # e.g., models/unsloth/Qwen3.5-0.8B-GGUF/
    # Keep "/" as is - creates nested folder structure like HuggingFace
    repo_folder = hf_repo  # "unsloth/Qwen3.5-0.8B-GGUF"
    model_dir = MODEL_DIR / repo_folder
    model_dir.mkdir(parents=True, exist_ok=True)

    local_path = model_dir / filename

    if local_path.exists():
        console.print(f"[yellow]Model already exists:[/yellow] {local_path}")
        return get_model_info(model_id)

    # Download from HuggingFace
    console.print("[blue]Downloading main model...[/blue]")

    try:
        downloaded_path = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")

    # Download mmproj if needed (for vision models) - save in same folder as model
    mmproj_saved_path = None
    if download_mmproj and registry_entry:
        mmproj_files = registry_entry.get("mmproj_files", [])
        if mmproj_files:
            # Try to find best mmproj: F32 > F16 > BF16
            mmproj_priority = ["mmproj-F32.gguf", "mmproj-F16.gguf", "mmproj-BF16.gguf"]
            for mmproj_name in mmproj_priority:
                if mmproj_name in mmproj_files:
                    try:
                        console.print(f"[blue]Downloading mmproj: {mmproj_name}[/blue]")
                        mmproj_path = hf_hub_download(
                            repo_id=hf_repo,
                            filename=mmproj_name,
                            local_dir=str(model_dir),
                            local_dir_use_symlinks=False,
                        )
                        mmproj_saved_path = mmproj_path
                        console.print(
                            f"[green]mmproj downloaded: {mmproj_name}[/green]"
                        )
                        break
                    except Exception:
                        continue

    # Update manifest
    manifest = get_local_models()

    # Check if already in manifest
    existing = next((m for m in manifest if m["id"] == model_id), None)
    if existing:
existing["filename"] = filename
        existing["path"] = str(model_dir / filename)  # Store full file path
        existing["mmproj"] = mmproj_saved_path  # Store mmproj path
        existing["date_updated"] = datetime.utcnow().isoformat() + "Z"
    else:
        size_gb = registry_entry.get("size_gb", {}).get(quant, 0)
manifest.append(
        {
            "id": model_id,
            "name": registry_entry["name"],
            "filename": filename,
            "path": str(model_dir / filename),  # Store full file path
            "mmproj": mmproj_saved_path,  # Store mmproj path
            "size_gb": size_gb,
            "quant": quant,
            "context": registry_entry.get("context", 4096),
            "date_added": datetime.utcnow().isoformat() + "Z",
            "hf_repo": hf_repo,
        }
    )

    save_local_models(manifest)

    # Regenerate config
    from .config import regenerate_config

    regenerate_config()

    console.print(f"[bold green]Model ready:[/bold green] {model_id}")

    return get_model_info(model_id)


def remove_model(model_id: str, yes: bool = False) -> None:
    """
    Remove a downloaded model.
    """
    manifest = get_local_models()
    entry = next((m for m in manifest if m["id"] == model_id), None)

    if entry is None:
        raise ValueError(f"Model '{model_id}' not found")

    if not yes:
        console.print(f"[yellow]Remove {entry['name']}?[/yellow] (y/n)")
        # In non-interactive mode, assume yes
        pass

    # Delete file
    model_path = pathlib.Path(entry.get("path", ""))
    if model_path.exists():
        model_path.unlink()
        console.print(f"[red]Deleted:[/red] {model_path}")

    # Remove from manifest
    manifest = [m for m in manifest if m["id"] != model_id]
    save_local_models(manifest)

    # Regenerate config
    from .config import regenerate_config

    regenerate_config()

    console.print(f"[green]Model removed:[/green] {model_id}")


def list_models() -> list[dict]:
    """List downloaded models."""
    return get_local_models()


def get_model_info(model_id: str) -> dict:
    """Get model info from manifest."""
    manifest = get_local_models()
    return next((m for m in manifest if m["id"] == model_id), {})


def get_model_path(model_id: str) -> pathlib.Path:
    """Get model file path."""
    info = get_model_info(model_id)
    if info:
        return pathlib.Path(info["path"])
    raise ValueError(f"Model '{model_id}' not found")


def get_local_models() -> list[dict]:
    """Load local models manifest."""
    if not MODELS_MANIFEST.exists():
        return []
    with open(MODELS_MANIFEST, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, list) else []


def get_all_models() -> list[dict]:
    """Get all available models from kapri."""
    return get_local_models()


def save_local_models(models: list[dict]) -> None:
    """Save local models manifest."""
    MODELS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2)


if __name__ == "__main__":
    # Test
    models = list_models()
    for m in models:
        print(m["id"], m["name"])
