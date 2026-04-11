"""Kapri model registry."""

import json
from pathlib import Path
from typing import Optional

import httpx

from .constants import (
    REGISTRY_CACHE,
    REGISTRY_CACHE_TTL,
    REGISTRY_URL,
    MODELS_MANIFEST,
)


def get_bundled_registry() -> list[dict]:
    """Load bundled registry from package - look in parent dirs."""
    # Try multiple locations relative to this file
    locations = [
        Path(__file__).parent
        / "registry_models.json",  # ./registry_models.json (bundled in package)
        Path(__file__).parent.parent
        / "registry"
        / "models.json",  # ../registry/models.json
        Path(__file__).parent
        / "registry"
        / "models.json",  # ./registry/models.json (if included)
    ]
    for bundled in locations:
        if bundled.exists():
            with open(bundled, "r", encoding="utf-8") as f:
                return json.load(f)
    return []


def fetch_registry(force_refresh: bool = False) -> list[dict]:
    """
    Fetch registry from remote or load from cache.

    1. If cache exists and age < TTL: return cached data
    2. GET from REGISTRY_URL
    3. On success: write cache, return list
    4. On network error: use bundled registry
    """
    # Check cache first
    if not force_refresh and REGISTRY_CACHE.exists():
        cache_age = Path(REGISTRY_CACHE).stat().st_mtime
        import time

        if time.time() - cache_age < REGISTRY_CACHE_TTL:
            with open(REGISTRY_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)

    # Try remote
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(REGISTRY_URL)
            response.raise_for_status()
            data = response.json()
            # Cache it
            REGISTRY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(REGISTRY_CACHE, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
    except Exception:
        # Fall back to bundled
        return get_bundled_registry()


def search_registry(query: str, registry: Optional[list[dict]] = None) -> list[dict]:
    """
    Case-insensitive search across id, name, description, tags.
    Sort: exact id match > name contains > tag match.
    """
    if registry is None:
        registry = fetch_registry()

    query_lower = query.lower()
    results = []

    for model in registry:
        score = 0
        # Exact id match
        if model.get("id", "").lower() == query_lower:
            score = 100
        # Name contains
        elif query_lower in model.get("name", "").lower():
            score = 50
        # Description contains
        elif query_lower in model.get("description", "").lower():
            score = 30
        # Tag match
        elif any(query_lower in tag.lower() for tag in model.get("tags", [])):
            score = 20

        if score > 0:
            results.append((score, model))

    # Sort by score descending
    results.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in results]


def resolve_model(
    model_ref: str, quant: Optional[str] = None
) -> tuple[Optional[dict], str]:
    """
    Resolve model reference to registry entry and filename.

    Formats:
      "llama3.2-3b" -> lookup by id
      "llama3.2-3b:Q5_K_M" -> lookup + override quant
      "bartowski/Llama-3.2-3B-GGUF:file.gguf" -> raw HF direct

    Returns: (registry_entry_or_None, resolved_hf_filename)
    """
    registry = fetch_registry()

    # Parse model_ref for quant override
    override_quant = quant
    if ":" in model_ref:
        parts = model_ref.split(":", 1)
        model_ref = parts[0]
        if override_quant is None and len(parts) > 1:
            override_quant = parts[1]

    # Check if it's a direct HF reference
    if "/" in model_ref and "GGUF" in model_ref:
        filename = override_quant if override_quant else "*.gguf"
        return None, filename

    # Search registry
    for model in registry:
        if model.get("id", "").lower() == model_ref.lower():
            final_quant = override_quant or model.get("default_quant", "Q4_K_M")
            pattern = model.get("file_pattern", "{id}-{quant}.gguf")
            filename = pattern.replace("{quant}", final_quant)
            return model, filename

    # Not found
    return None, ""


def get_local_models() -> list[dict]:
    """Get models from local manifest."""
    if not MODELS_MANIFEST.exists():
        return []
    with open(MODELS_MANIFEST, "r", encoding="utf-8") as f:
        return json.load(f)


def save_local_models(models: list[dict]) -> None:
    """Save models to local manifest."""
    MODELS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELS_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2)
