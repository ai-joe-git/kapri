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


def pull_model(model_ref: str, quant: str = "Q4_K_M") -> dict:
    """
    Download a model from HuggingFace.
    """
    # Resolve model
    registry_entry, filename = resolve_model(model_ref, quant)
    if registry_entry is None:
        raise ValueError(f"Model '{model_ref}' not found in registry")

    hf_repo = registry_entry["hf_repo"]
    model_id = registry_entry["id"]

    console.print(f"[bold]Pulling:[/bold] {registry_entry['name']}")
    console.print(f"  Repository: {hf_repo}")
    console.print(f"  File: {filename}")

    # Ensure model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_DIR / filename

    if local_path.exists():
        console.print(f"[yellow]Model already exists:[/yellow] {local_path}")
        return get_model_info(model_id)

    # Download from HuggingFace
    console.print("[blue]Downloading...[/blue]")

    # Use hf_hub
    from huggingface_hub import hf_hub_download

    try:
        downloaded_path = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download: {e}")

    # Update manifest
    manifest = get_local_models()

    # Check if already in manifest
    existing = next((m for m in manifest if m["id"] == model_id), None)
    if existing:
        existing["filename"] = filename
        existing["path"] = str(downloaded_path)
        existing["date_updated"] = datetime.utcnow().isoformat() + "Z"
    else:
        size_gb = registry_entry.get("size_gb", {}).get(quant, 0)
        manifest.append(
            {
                "id": model_id,
                "name": registry_entry["name"],
                "filename": filename,
                "path": downloaded_path,
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
