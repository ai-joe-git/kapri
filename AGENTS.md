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

## config.py: INI format, not configparser

`models.ini` starts with a top-level `version = 1` line (no section header), which makes Python's `configparser` unusable — it throws `MissingSectionHeaderError`. Do NOT use `configparser` to read or write this file.

- **Writing**: `regenerate_config()` builds the file as raw strings and writes directly.
- **Reading**: `load_config()` uses `_parse_ini()` — a manual regex-based parser in `config.py`.
- **Imports**: `regenerate_config` is imported lazily inside functions in `models.py` (not at module level) to avoid circular imports between `config ↔ models`.

## Unsloth GGUF quirks (critical for model pulling)

- **`-UD-` filename prefix**: unsloth names Dynamic XL/IQ quants with a `-UD-` prefix (e.g., `Qwen3.5-0.8B-UD-Q4_K_XL.gguf`). The registry's `file_pattern` generates without the prefix, so `models.py` has a 404-retry that inserts `-UD-` into the filename.
- **Prefer Dynamic quants**: pre-March-5 non-Dynamic unsloth GGUFs produce `/////` garbage output. When multiple GGUF files exist in a model directory, `regenerate_config()` prefers files with `-UD-` in the name and picks the largest.
- **Deleting old quants**: if both an old non-Dynamic and a new Dynamic GGUF exist, delete the old one manually. The presence of both causes `regenerate_config` confusion.

## Qwen 3.5/3.6 sampling defaults

These are from unsloth's documented non-thinking general-task defaults and MUST be in `models.ini`:

```
temp = 0.7
top-p = 0.8
top-k = 20
min-p = 0.0    ← critical: llama.cpp defaults to 0.1, which kills ~90% of Qwen tokens
reasoning = off
```

Without `min-p = 0.0`, Qwen models output `????` or `////` garbage. Without `reasoning = off`, large Qwen models (27B+) can get stuck in thinking loops burning the full context.

## Vercel deploy

The website (`kapri-ai.vercel.app`) and registry (`kapri-registry.vercel.app`) are separate Vercel projects under `aijoes-projects`. Git integration is configured in the Vercel dashboard (root directories: `./website` and `./registry`). The Vercel CLI auto-links to the first matching project scope and can mis-deploy to wrong aliases — prefer the dashboard for deploys or link carefully. `.vercel` directories are gitignored.

## PyPI publish flow

No CI publishes to PyPI — it's manual:

```pwsh
# 1. Bump version in both pyproject.toml and kapri/__init__.py
# 2. Build
python -m build
# 3. Upload (token via TWINE_PASSWORD env var or .pypirc)
twine upload dist/kapri_ai-<version>*
# 4. Commit version bump + dist/ and push
```

## CI

- **Release** (`release.yml`): Triggered by `v*` tags. PyInstaller builds on Linux/Windows/macOS, then creates a GitHub release.
- **Registry update** (`registry-update.yml`): Weekly cron + manual dispatch. Validates registry JSON schema and auto-commits changes.
