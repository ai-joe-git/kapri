#!/usr/bin/env python3
"""Registry update script - schema validation."""

import json
import sys
from pathlib import Path

REQUIRED_FIELDS = [
    "id",
    "name",
    "description",
    "hf_repo",
    "default_quant",
    "quants",
    "file_pattern",
    "size_gb",
    "context",
    "tags",
    "capabilities",
    "license",
]


def validate_registry(registry_path: Path) -> int:
    """Validate registry JSON schema."""
    with open(registry_path, "r", encoding="utf-8") as f:
        models = json.load(f)

    if not isinstance(models, list):
        print("ERROR: Registry must be a JSON array")
        return 1

    errors = []
    for i, model in enumerate(models):
        for field in REQUIRED_FIELDS:
            if field not in model:
                errors.append(
                    f"Model {i} ({model.get('id', 'unknown')}): missing '{field}'"
                )

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        print(f"\nTotal: {len(errors)} errors found")
        return 1

    print(f"OK: {len(models)} models validated")
    return 0


if __name__ == "__main__":
    registry = Path(__file__).parent / "models.json"
    sys.exit(validate_registry(registry))
