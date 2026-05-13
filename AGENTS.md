# AGENTS.md

## Project overview

Kapri is a Python CLI tool — an Ollama-compatible local LLM runner built on llama.cpp. Distributed on PyPI as `kapri-ai`. The CLI is a single Typer app at `kapri/cli.py:main`.

## Package shape

- **`kapri/`** — the Python package (pip-installable, setuptools).
- **`registry/`** — standalone static JSON API deployed on Vercel. Not part of the Python package.
- **`website/`** — standalone static HTML site deployed on Vercel. Not part of the Python package.

Only changes under `kapri/` ship with the pip package.

## Build & install

```sh
pip install -e .          # editable dev install
pip install .             # production install
python -m build           # build wheel + sdist into dist/
```

Entry point is `kapri` → `kapri.cli:main`. No lockfiles, no venv manager — dependencies are declared inline in `pyproject.toml` with version ranges.

## No tests, no lint config

This project has zero test files, zero test dependencies, and no linter/formatter config. The `.ruff_cache/` directory is a stale artifact — there is no `ruff.toml` or `pyproject.toml [tool.ruff]` section. Do not attempt to run tests or lint checks unless explicitly asked to configure them first.

## Registry duplication (critical)

The model registry is duplicated in two places that **must stay in sync**:

- `kapri/registry_models.json` — bundled with the pip package (included via `MANIFEST.in`)
- `registry/models.json` — deployed to `https://kapri-registry.vercel.app/models.json`

The runtime resolves the bundled file via `kapri/registry.py:get_bundled_registry()` which checks `registry_models.json` first, then falls back to `../registry/models.json`.

**Note:** `pyproject.toml` `[tool.setuptools.package-data]` points to `registry/models.json` but `MANIFEST.in` includes `registry_models.json`. This works because both exist and the runtime checks multiple paths, but the canonical bundled file is `registry_models.json`.

## Version bump checklist

Version is hardcoded in two files — both must be updated together:
- `pyproject.toml` → `[project] version =`
- `kapri/__init__.py` → `__version__ =`

## Architecture notes

- **Server port 11434** (Ollama-compatible).
- **Config is auto-generated** by `kapri/config.py` into a `models.ini` file consumed by llama.cpp's native router mode (`--models-preset`). Config regeneration happens after model pull/remove. Do not hand-edit the generated config; it will be overwritten.
- **Server runs llama.cpp directly in router mode** — `llama-server --models-preset models.ini --models-max 1 --host 0.0.0.0 --port 11434 --metrics --tools all`. No separate proxy/routing binary.
- **Backend detection** (`kapri/installer.py:detect_backend()`) auto-selects GPU backend at install time (Metal > CUDA > ROCm > Vulkan > SYCL > CPU). The `kapri backend <name>` command can override.
- **Model downloads** go through `huggingface_hub.hf_hub_download`, not a custom protocol. The registry is just a search index — actual files come from HuggingFace repos.
- **Binary downloads** pull from GitHub releases of `ggml-org/llama.cpp`, matching platform/arch/backend patterns.
- **No Docker** — the app installs binaries directly to `~/.kapri/bin` (Unix) or `%APPDATA%/kapri/bin` (Windows).

## CI

- **Release** (`release.yml`): Triggered by `v*` tags. PyInstaller builds on Linux/Windows/macOS, then creates a GitHub release.
- **Registry update** (`registry-update.yml`): Weekly cron + manual dispatch. Validates registry JSON schema and auto-commits changes.
