"""Kapri CLI - main entry point."""

import json
import os
import pathlib
import subprocess
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import regenerate_config, load_config, CONFIG_FILE
from .constants import BIN_DIR, DEFAULT_PORT, LLAMA_CLI_BIN
from .installer import install_binaries, detect_backend, get_current_versions
from .models import (
    pull_model,
    remove_model,
    list_models,
    get_model_info,
    get_local_models,
    get_all_models,
)
from .registry import fetch_registry, search_registry, resolve_model
from .server import (
    start_server,
    stop_server,
    server_status,
    tail_logs,
    ensure_server_running,
)
from .constants import DEFAULT_PORT, LOG_FILE

app = typer.Typer(
    name="kapri",
    help="[bold]Kapri[/bold] — Run AI locally. Beautifully.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=True,
)
console = Console()


# ==== Install Helper Functions ====


def find_existing_llama_server() -> Optional[str]:
    """Search for existing llama-server installations in kapri bin first."""
    # Check kapri bin first
    from .constants import BIN_DIR, LLAMASERVER_BIN

    kapri_server = BIN_DIR / LLAMASERVER_BIN
    if kapri_server.exists():
        return str(kapri_server)
    return None


def find_existing_llama_swap() -> Optional[str]:
    """Search for existing llama-swap in kapri bin first."""
    from .constants import BIN_DIR, LLAMASWAP_BIN

    kapri_swap = BIN_DIR / LLAMASWAP_BIN
    if kapri_swap.exists():
        return str(kapri_swap)
    return None


def find_existing_llama_swap_config() -> Optional[str]:
    """Search for existing config in kapri config directory."""
    from .constants import CONFIG_FILE

    if CONFIG_FILE.exists():
        return str(CONFIG_FILE)
    return None


def configure_custom_paths(
    llama_server_path: Optional[str] = None,
    llama_swap_path: Optional[str] = None,
    import_config: Optional[str] = None,
) -> bool:
    """Configure paths to existing installations - COPIES files to kapri directories."""
    import pathlib
    import shutil as shutil_module
    import yaml
    from .config import regenerate_config
    from .constants import BIN_DIR, CONFIG_FILE

    # Ensure kapri directories exist
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Validate and COPY paths
    copied_server_path = None
    copied_swap_path = None

    if llama_server_path:
        source = pathlib.Path(llama_server_path)
        if not source.exists():
            console.print(f"[red]llama-server not found:[/red] {llama_server_path}")
            return False

        # Copy to kapri bin directory
        dest_name = source.name
        dest = BIN_DIR / dest_name
        console.print(f"[dim]Copying llama-server to: {dest}[/dim]")
        shutil_module.copy2(source, dest)
        copied_server_path = str(dest)

    if llama_swap_path:
        source = pathlib.Path(llama_swap_path)
        if not source.exists():
            console.print(f"[red]llama-swap not found:[/red] {llama_swap_path}")
            return False

        # Copy to kapri bin directory
        dest_name = source.name
        dest = BIN_DIR / dest_name
        console.print(f"[dim]Copying llama-swap to: {dest}[/dim]")
        shutil_module.copy2(source, dest)
        copied_swap_path = str(dest)

    if import_config:
        source = pathlib.Path(import_config)
        if not source.exists():
            console.print(f"[red]Config not found:[/red] {import_config}")
            return False

        # Copy to kapri config location
        dest = CONFIG_FILE
        console.print(f"[dim]Copying config to: {dest}[/dim]")
        shutil_module.copy2(source, dest)
        copied_config_path = str(dest)
    else:
        copied_config_path = None

    # Save COPIED paths to settings
    from .constants import VERSIONS_FILE
    import json

    VERSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing versions or create new
    if VERSIONS_FILE.exists():
        with open(VERSIONS_FILE, "r", encoding="utf-8") as f:
            versions = json.load(f)
    else:
        versions = {}

    # Update with COPIED paths (not original paths)
    if copied_server_path:
        versions["custom_llama_server"] = copied_server_path
    if copied_swap_path:
        versions["custom_llama_swap"] = copied_swap_path
    if copied_config_path:
        versions["imported_config"] = copied_config_path

    with open(VERSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2)

    console.print("[green]Files copied successfully![/green]")
    console.print(f"  llama-server: {copied_server_path or 'default'}")
    console.print(f"  llama-swap: {copied_swap_path or 'default'}")
    console.print(f"  config: {copied_config_path or 'default'}")

    console.print("[green]Paths configured successfully![/green]")
    return True


@app.command()
def install(
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="GPU backend: cuda, vulkan, rocm, sycl, metal, cpu",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall"),
    llama_server_path: Optional[str] = typer.Option(
        None,
        "--llama-server",
        help="Path to existing llama-server executable",
    ),
    llama_swap_path: Optional[str] = typer.Option(
        None,
        "--llama-swap",
        help="Path to existing llama-swap executable",
    ),
    import_config: Optional[str] = typer.Option(
        None,
        "--import-config",
        help="Path to existing llama-swap config.yaml to import",
    ),
):
    """
    Install or configure llama-server and llama-swap.

    Options:
      --llama-server <path>    Use existing llama-server binary
      --llama-swap <path>    Use existing llama-swap binary
      --import-config <path>  Import models from existing llama-swap config
      --backend <type>       Backend: auto, vulkan, cuda, rocm, sycl, metal, cpu
      --force                Force reinstall even if already installed

    If no options provided, shows interactive setup wizard.
    """
    # If backend specified, just install that backend
    if backend:
        valid = ["auto", "vulkan", "cuda", "rocm", "sycl", "metal", "cpu"]
        if backend.lower() not in valid:
            console.print(f"[red]Invalid backend: {backend}[/red]")
            console.print(f"Valid: {', '.join(valid)}")
            raise typer.Exit(1)

        versions = install_binaries(force=force, backend=backend.lower())
        console.print(f"[green]Installed {backend}:[/green]")
        console.print(f"  llama.cpp: {versions.get('llamacpp')}")
        console.print(f"  llama-swap: {versions.get('llamaswap')}")
        return

    # Check if any existing path provided
    has_custom_path = llama_server_path or llama_swap_path or import_config

    if has_custom_path:
        # Non-interactive: use provided paths
        result = configure_custom_paths(
            llama_server_path=llama_server_path,
            llama_swap_path=llama_swap_path,
            import_config=import_config,
        )
        if result:
            console.print("[green]Configuration saved![/green]")
        return

    # Interactive wizard
    console.print("[bold cyan]Kapri Setup Wizard[/bold cyan]\n")

    # Check existing installations
    existing_llama_server = find_existing_llama_server()
    existing_llama_swap = find_existing_llama_swap()
    existing_config = find_existing_llama_swap_config()

    console.print("[dim]Checking for existing installations...[/dim]")

    # Step 1: llama-server
    console.print("\n[bold]Step 1: llama-server (llama.cpp)[/bold]")

    use_existing_server = False
    server_path = None

    if existing_llama_server:
        console.print(f"  Found: {existing_llama_server}")
        response = console.input("  Use this? [Y/n]: ").strip().lower()
        if not response or response == "y":
            server_path = existing_llama_server
            use_existing_server = True

    if not use_existing_server:
        console.print("  Options:")
        console.print("    1. Auto-detect (recommended)")
        console.print("    2. Vulkan (AMD GPUs)")
        console.print("    3. CUDA (NVIDIA GPUs)")
        console.print("    4. ROCm (AMD Linux)")
        console.print("    5. SYCL (Intel GPUs)")
        console.print("    6. Metal (Apple Silicon)")
        console.print("    7. CPU only")
        console.print("    8. Specify custom path")
        console.print("    9. Skip (use default)")

        response = console.input("  Choice [1]: ").strip() or "1"

        backend_map = {
            "1": "auto",
            "2": "vulkan",
            "3": "cuda",
            "4": "rocm",
            "5": "sycl",
            "6": "metal",
            "7": "cpu",
        }

        if response in backend_map:
            backend = backend_map[response]
            versions = install_binaries(force=force, backend=backend)
            console.print(f"[green]Downloaded: {versions.get('llamacpp')}[/green]")
        elif response == "8":
            server_path = console.input("  Enter path to llama-server: ").strip()
        # 9 = skip, use default

    # Step 2: llama-swap
    console.print("\n[bold]Step 2: llama-swap[/bold]")

    use_existing_swap = False

    if existing_llama_swap:
        console.print(f"  Found: {existing_llama_swap}")
        response = console.input("  Use this? [Y/n]: ").strip().lower()
        if not response or response == "y":
            use_existing_swap = True

    if not use_existing_swap:
        console.print("  Options:")
        console.print("    1. Download latest")
        console.print("    2. Specify custom path")

        response = console.input("  Choice [1]: ").strip() or "1"

        if response == "1":
            versions = install_binaries(force=force, backend=backend)
            console.print(f"[green]Downloaded: {versions.get('llamaswap')}[/green]")
        elif response == "2":
            swap_path = console.input("  Enter path to llama-swap: ").strip()

    # Step 3: Import config
    console.print("\n[bold]Step 3: Model Config[/bold]")

    if existing_config:
        console.print(f"  Found: {existing_config}")
        response = (
            console.input("  Import models from this config? [Y/n]: ").strip().lower()
        )
        if not response or response == "y":
            # This will be handled by server.py when starting
            console.print(f"[green]Will import from: {existing_config}[/green]")

    console.print("\n[bold green]Setup complete![/green]")
    console.print("Run 'kapri serve' to start the server.")


@app.command()
def pull(
    model: str = typer.Argument(..., help="Model ID (e.g., llama3.2-3b)"),
    quant: str = typer.Option("Q4_K_M", "--quant", "-q", help="Quantization"),
):
    """Download a model from HuggingFace."""
    try:
        result = pull_model(model, quant)
        console.print(f"[green]Model ready:[/green] {result.get('id')}")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def run(
    model: str = typer.Argument(..., help="Model ID to run"),
    ctx: int = typer.Option(4096, "--ctx", help="Context size"),
    ngl: int = typer.Option(99, "--ngl", "-g", help="GPU layers"),
    system: Optional[str] = typer.Option(None, "--system", "-s", help="System prompt"),
    tui: bool = typer.Option(
        False, "--tui", "-t", help="Open terminal chat UI instead of web UI"
    ),
):
    """Start a chat session. Default: opens web UI. Use --tui for terminal chat."""

    # Resolve model ID - use kapri's internal config
    import yaml

    config_path = CONFIG_FILE

    # Find actual model ID in config
    model_id = model
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        models_in_config = config_data.get("models", {})

        # Normalize user input
        model_normalized = (
            model.lower().replace("-", "").replace("_", "").replace("gguf", "")
        )

        # Try exact match first
        if model in models_in_config:
            model_id = model
        else:
            # Try case-insensitive match
            for m_id in models_in_config:
                m_id_normalized = (
                    m_id.lower().replace("-", "").replace("_", "").replace("gguf", "")
                )
                if (
                    model_normalized == m_id_normalized
                    or model_normalized in m_id_normalized
                    or m_id_normalized in model_normalized
                ):
                    model_id = m_id
                    break
            else:
                console.print(f"[yellow]Model not found: {model}[/yellow]")
                console.print(f"Available: {', '.join(models_in_config.keys())}")
                raise typer.Exit(1)

    # Ensure server is running
    ensure_server_running()

    # Default: open webui (llama-server through llama-swap)
    if not tui:
        import webbrowser

        url = f"http://localhost:{DEFAULT_PORT}/upstream/{model_id}/"

        console.print(f"[green]Opening web UI:[/green] {url}")
        console.print("[dim]Model auto-loads through llama-swap[/dim]")

        webbrowser.open(url)

        try:
            while True:
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Web UI closed[/yellow]")
        return

    # --tui flag: use terminal chat via HTTP through llama-swap
    # Terminal chat via llama-swap's /upstream endpoint
    upstream_url = (
        f"http://localhost:{DEFAULT_PORT}/upstream/{model_id}/v1/chat/completions"
    )

    history = []

    if system:
        history.append({"role": "system", "content": system})

    console.print(f"[green]Terminal chat: {model_id}[/green]")
    console.print("[dim]Type /exit to quit, /clear to reset[/dim]")

    while True:
        try:
            user_input = console.input("\n[green]>[/green] ")
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input.strip():
            continue

        if user_input.strip() == "/exit":
            break
        if user_input.strip() == "/clear":
            history = []
            if system:
                history.append({"role": "system", "content": system})
            console.print("[dim]History cleared[/dim]")
            continue

        history.append({"role": "user", "content": user_input})

        # Show loading
        import sys

        sys.stdout.write("[dim]Loading...[/dim]\r")
        sys.stdout.flush()

        try:
            import json as json_module
            import urllib.request

            payload = json_module.dumps(
                {
                    "model": model_id,
                    "messages": history,
                    "stream": True,
                }
            ).encode("utf-8")

            req = urllib.request.Request(
                upstream_url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
                method="POST",
            )

            resp = urllib.request.urlopen(req, timeout=None)

            sys.stdout.write(" " * 20 + "\r")
            sys.stdout.flush()

            content = ""
            sys.stdout.write("[blue]>[/blue] ")
            sys.stdout.flush()

            buffer = ""
            while True:
                try:
                    chunk = resp.read(1024)
                    if not chunk:
                        break

                    text = chunk.decode("utf-8", errors="ignore")
                    buffer += text

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line == "data: [DONE]":
                            break
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if not data_str:
                                continue
                            try:
                                chunk_data = json_module.loads(data_str)
                                if "choices" in chunk_data and chunk_data["choices"]:
                                    delta = chunk_data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        token = delta["content"]
                                        content += token
                                        sys.stdout.write(token)
                                        sys.stdout.flush()
                            except:
                                continue
                except:
                    break

            resp.close()
            sys.stdout.write("\n")
            sys.stdout.flush()
            history.append({"role": "assistant", "content": content})

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

    # Resolve model ID to actual config name - use SAME logic as start_server
    import yaml

    # Load settings to know which config is being used
    from .server import load_settings

    settings = load_settings()

    # Use kapri's internal config (don't reference original llama-swap config)
    # The server already merged everything when it started
    config_path = CONFIG_FILE

    resolved_model = model

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        models_in_config = config_data.get("models", {})

        # Check if exact match
        if model not in models_in_config:
            # Try case-insensitive match
            for m_id in models_in_config:
                if model.lower() == m_id.lower():
                    resolved_model = m_id
                    break
            else:
                # Try partial match (e.g., "qwopus3.5-2b" matches "Qwopus3.5-2B")
                model_lower = model.lower().replace("-", "").replace("_", "")
                for m_id in models_in_config:
                    m_id_normalized = m_id.lower().replace("-", "").replace("_", "")
                    if model_lower in m_id_normalized or m_id_normalized in model_lower:
                        resolved_model = m_id
                        break
                else:
                    console.print(f"[red]Model not found:[/red] {model}")
                    console.print("Available models:")
                    for m_id in models_in_config:
                        console.print(f"  - {m_id}")
                    raise typer.Exit(1)

    # Ensure server is running (llama-swap as daemon)
    ensure_server_running()

    # Terminal chat: run llama-cli.exe directly in terminal
    # Get model config from kapri's config
    import yaml

    model_config = config_data.get("models", {}).get(resolved_model, {})
    model_cmd = model_config.get("cmd", "")
    model_env = model_config.get("env", [])

    # Get llama-cli from kapri bin
    llama_cli = BIN_DIR / LLAMA_CLI_BIN

    if not llama_cli.exists():
        console.print(f"[red]llama-cli not found: {llama_cli}[/red]")
        raise typer.Exit(1)

    # Build args from config - remove server args, keep model args
    # Replace ${llama_server} with nothing since we run directly
    cmd_args = (
        model_cmd.replace("${llama_server}", "")
        .replace("${llama_cli}", "")
        .replace("${listen_args}", "")
        .replace("--host 0.0.0.0", "")
        .replace(" --port ${PORT}", "")
        .replace("--port ${PORT}", "")
    )

    # Parse args
    import shlex

    try:
        args_list = shlex.split(cmd_args)
    except:
        args_list = cmd_args.split()

    # Build CLI args - interactive mode
    cli_args = ["-cnv"]
    skip_next = False
    for i, arg in enumerate(args_list):
        if skip_next:
            skip_next = False
            continue
        # Skip server args
        if arg in ["--host", "--port", "-p"]:
            skip_next = True
            continue
        cli_args.append(arg)

    # Add system prompt
    if system:
        cli_args.extend(["--system", system])

    # Build env
    env = os.environ.copy()
    for e in model_env:
        if "=" in e:
            key, val = e.split("=", 1)
            env[key] = val

    console.print(f"[green]llama.cpp interactive chat[/green]")
    console.print(f"[dim]Model: {resolved_model}[/dim]")
    console.print("[dim]Type /exit to quit[/dim]")

    # Run llama-cli directly in terminal
    try:
        subprocess.run([str(llama_cli)] + cli_args, env=env)
    except KeyboardInterrupt:
        console.print("\n[yellow]Chat ended[/yellow]")
    return


@app.command()
def serve(
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Port"),
    foreground: bool = typer.Option(
        False, "--foreground", "-f", help="Run in foreground"
    ),
):
    """Start the Kapri server."""
    try:
        start_server(port=port, foreground=foreground)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def stop():
    """Stop the server."""
    stop_server()


@app.command()
def status(json_output: bool = typer.Option(False, "--json", help="JSON output")):
    """Show server status."""
    status = server_status()

    if json:
        console.print(json.dumps(status, indent=2))
        return

    if status["running"]:
        console.print(f"[green]Server running[/green]")
        console.print(f"  PID: {status['pid']}")
        console.print(f"  Port: {status['port']}")
        if status["uptime_seconds"]:
            secs = status["uptime_seconds"]
            mins = secs // 60
            console.print(f"  Uptime: {mins}m")
        if status["loaded_models"]:
            console.print(f"  Models: {', '.join(status['loaded_models'])}")
    else:
        console.print("[yellow]Server stopped[/yellow]")


@app.command()
def list(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show details"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
    local_only: bool = typer.Option(
        False, "--local-only", help="Only show kapri-downloaded models"
    ),
):
    """List available models (includes config models + kapri-downloaded)."""
    import yaml

    # Get kapri-downloaded models
    kapri_models = get_local_models()

    # Get models from config
    config_models = []
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        for model_id, model_data in config_data.get("models", {}).items():
            # Skip external models
            if model_id in ["whisper-large-v3-turbo", "omnivoice-tts"]:
                continue
            config_models.append(
                {
                    "id": model_id,
                    "name": model_data.get("name", model_id),
                    "source": "config",
                    "path": model_data.get("cmd", ""),
                }
            )

    # Combine (config models first, then kapri models)
    models = config_models + kapri_models

    if json_output:
        console.print(json.dumps(models, indent=2))
        return

    if not models:
        console.print("[yellow]No models available[/yellow]")
        console.print("Options:")
        console.print("  - kapri pull <model>    # Download a model")
        console.print(
            "  - kapri install --import-config <path>  # Import existing config"
        )
        return

    table = Table(title=f"Models ({len(models)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Size")
    table.add_column("Source")
    table.add_column("Context")

    for m in models:
        source = m.get("source", "kapri")
        size = m.get("size_gb", "-")
        if size != "-":
            size = f"{size}GB"
        table.add_row(
            m.get("id", "-"),
            m.get("name", "-")[:30],
            size,
            source,
            str(m.get("context", "-")),
        )

    console.print(table)

    if verbose:
        console.print("\n[bold]Details:[/bold]")
        for m in models:
            console.print(f"  {m.get('id')}: {m.get('path')}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Search model registry."""
    results = search_registry(query)

    if json_output:
        console.print(json.dumps(results, indent=2))
        return

    if not results:
        console.print(f"[yellow]No models found for:[/yellow] {query}")
        return

    table = Table(title=f"Results ({len(results)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Quant")
    table.add_column("Size")
    table.add_column("Tags")

    for m in results:
        size = m.get("size_gb", {}).get(m.get("default_quant", "Q4_K_M"), 0)
        table.add_row(
            m.get("id", "-"),
            m.get("name", "-")[:25],
            m.get("default_quant", "-"),
            f"{size:.1f}GB",
            ", ".join(m.get("tags", [])[:3]),
        )

    console.print(table)


@app.command()
def remove(
    model: str = typer.Argument(..., help="Model ID to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirm"),
):
    """Remove a downloaded model."""
    try:
        remove_model(model, yes=yes)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def info(
    model: str = typer.Argument(..., help="Model ID"),
):
    """Show model information."""
    info = get_model_info(model)

    if not info:
        console.print(f"[red]Model not found:[/red] {model}")
        raise typer.Exit(1)

    table = Table(title=info.get("name", model))
    table.add_column("Property")
    table.add_column("Value")

    for key, value in info.items():
        table.add_row(key, str(value))

    console.print(table)


@app.command("update")
def update(
    everything: bool = typer.Option(
        False, "--all", "-a", help="Update kapri package and all binaries"
    ),
):
    """Update kapri and/or binaries."""
    import subprocess

    if everything:
        # Update kapri package
        console.print("[blue]Updating kapri package...[/blue]")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "kapri-ai"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            console.print("[green]Kapri updated[/green]")
        else:
            console.print(f"[yellow]Warning: {result.stderr}[/yellow]")

        # Update binaries
        console.print("[blue]Updating binaries...[/blue]")
        versions = install_binaries(force=True, backend=None)
        console.print(
            f"[green]Updated: {versions.get('llamacpp')}, {versions.get('llamaswap')}[/green]"
        )
    else:
        # Just update binaries
        versions = install_binaries(force=True, backend=None)
        console.print(
            f"[green]Binaries updated: {versions.get('llamacpp')}, {versions.get('llamaswap')}[/green]"
        )


@app.command("backend")
def set_backend(
    backend: str = typer.Argument(
        ...,
        help="Backend: auto, vulkan, cuda, rocm, sycl, metal, cpu",
    ),
    force: bool = typer.Option(True, "--force", "-f", help="Force reinstall"),
):
    """Change GPU backend and re-download binaries."""
    valid_backends = ["auto", "vulkan", "cuda", "rocm", "sycl", "metal", "cpu"]

    backend = backend.lower()
    if backend not in valid_backends:
        console.print(f"[red]Invalid backend: {backend}[/red]")
        console.print(f"Valid: {', '.join(valid_backends)}")
        raise typer.Exit(1)

    console.print(f"[blue]Installing {backend} backend...[/blue]")
    versions = install_binaries(force=force, backend=backend)

    # Save backend preference
    settings_path = VERSIONS_FILE
    if settings_path.exists():
        with open(settings_path, "r", encoding="utf-8") as f:
            versions_data = json.load(f)
    else:
        versions_data = {}

    versions_data["backend"] = backend
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(versions_data, f, indent=2)

    console.print(f"[green]Backend set to {backend}[/green]")
    console.print(f"  llama.cpp: {versions.get('llamacpp')}")
    console.print(f"  llama-swap: {versions.get('llamaswap')}")


@app.command()
def logs(
    lines: int = typer.Option(50, "--lines", "-n", help="Number of lines"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Stream new lines"),
):
    """Show server logs."""
    if follow:
        console.print("[dim]Following... (Ctrl+C to quit)[/dim]")
        try:
            import time

            with open(LOG_FILE, "r") as f:
                f.seek(0, 2)  # EOF
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.5)
                        continue
                    console.print(line.rstrip())
        except KeyboardInterrupt:
            pass
    else:
        content = tail_logs(lines)
        console.print(content)


@app.command("config")
def config_cmd(
    show: bool = typer.Option(False, "--show", help="Show config"),
    path: bool = typer.Option(False, "--path", help="Show config path"),
    reset: bool = typer.Option(False, "--reset", help="Regenerate config"),
):
    """Manage configuration."""
    if path:
        console.print(str(CONFIG_FILE))
        return

    if show or (not show and not reset):
        cfg = load_config()
        if cfg:
            import yaml

            console.print(yaml.dump(cfg))
        else:
            console.print("[yellow]No config found[/yellow]")

    if reset:
        regenerate_config()
        console.print("[green]Config regenerated[/green]")


# Model config subcommand
model_app = typer.Typer(help="Manage model configurations")


@model_app.command("show")
def model_config_show(
    model: str = typer.Argument(..., help="Model ID or name"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
):
    """
    Show model configuration details.

    Example:
        kapri model config show gpt-oss-20b
    """
    import yaml

    config_path = CONFIG_FILE
    if not config_path.exists():
        console.print("[red]Config file not found[/red]")
        raise typer.Exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    models = config_data.get("models", {})

    # Search for model (case-insensitive)
    found = None
    for m_id, m_data in models.items():
        if model.lower() == m_id.lower():
            found = (m_id, m_data)
            break

    if not found:
        console.print(f"[red]Model not found:[/red] {model}")
        raise typer.Exit(1)

    m_id, m_data = found

    if json_output:
        console.print(json.dumps({m_id: m_data}, indent=2))
    else:
        table = Table(title=f"Model: {m_id}")
        table.add_column("Property")
        table.add_column("Value")

        table.add_row("name", str(m_data.get("name", "-")))
        table.add_row("description", str(m_data.get("description", "-")))

        env = m_data.get("env", [])
        if env:
            table.add_row("env", "\n".join(env))

        cmd = m_data.get("cmd", "-")
        if len(cmd) > 100:
            cmd = cmd[:100] + "..."
        table.add_row("cmd", cmd)

        console.print(table)


@model_app.command("search")
def model_config_search(
    query: str = typer.Argument(..., help="Search query"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
):
    """
    Search models in config by name or description.

    Example:
        kapri model config search qwen
    """
    import yaml

    config_path = CONFIG_FILE
    if not config_path.exists():
        console.print("[red]Config file not found[/red]")
        raise typer.Exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    models = config_data.get("models", {})

    query_lower = query.lower()
    results = []

    for m_id, m_data in models.items():
        if (
            query_lower in m_id.lower()
            or query_lower in m_data.get("name", "").lower()
            or query_lower in m_data.get("description", "").lower()
        ):
            results.append((m_id, m_data))

    if not results:
        console.print(f"[yellow]No models found matching:[/yellow] {query}")
        return

    if json_output:
        console.print(json.dumps({m_id: m_data for m_id, m_data in results}, indent=2))
    else:
        table = Table(title=f"Results ({len(results)})")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Description")

        for m_id, m_data in results:
            table.add_row(
                m_id,
                m_data.get("name", "-")[:25],
                m_data.get("description", "-")[:40],
            )

        console.print(table)


@model_app.command("edit")
def model_config_edit(
    model: str = typer.Argument(..., help="Model ID or name"),
):
    """
    Edit model configuration (opens in editor).

    Example:
        kapri model config edit gpt-oss-20b
    """
    import subprocess

    editor = os.environ.get("EDITOR", "notepad")

    console.print(f"[yellow]Opening config in editor:[/yellow] {editor}")
    console.print(f"[dim]Config file: {CONFIG_FILE}[/dim]")
    console.print("[dim]Edit the model section manually, then save and exit.[/dim]")

    subprocess.run([editor, str(CONFIG_FILE)])


# Register as subcommand
app.add_typer(model_app, name="model")


@app.command()
def version():
    """Show version."""
    console.print(f"Kapri {__version__}")


# Entry point
if __name__ == "__main__":
    app()


# Register callbacks
def get_cli():
    return app


# For pip install -e
def main():
    app()


if __name__ == "__main__":
    main()
