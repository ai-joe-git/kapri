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
from .config import regenerate_config, load_config, MODELS_PRESET_FILE
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
    from .constants import BIN_DIR, LLAMASERVER_BIN

    kapri_server = BIN_DIR / LLAMASERVER_BIN
    if kapri_server.exists():
        return str(kapri_server)
    return None


def configure_custom_paths(
    llama_server_path: Optional[str] = None,
) -> bool:
    """Configure path to existing llama-server installation - COPIES file to kapri directory."""
    import shutil as shutil_module
    from .constants import BIN_DIR

    BIN_DIR.mkdir(parents=True, exist_ok=True)

    if llama_server_path:
        source = pathlib.Path(llama_server_path)
        if not source.exists():
            console.print(f"[red]llama-server not found:[/red] {llama_server_path}")
            return False

        dest_name = source.name
        dest = BIN_DIR / dest_name
        console.print(f"[dim]Copying llama-server to: {dest}[/dim]")
        shutil_module.copy2(source, dest)

        from .constants import VERSIONS_FILE

        VERSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if VERSIONS_FILE.exists():
            with open(VERSIONS_FILE, "r", encoding="utf-8") as f:
                versions = json.load(f)
        else:
            versions = {}

        versions["custom_llama_server"] = str(dest)
        with open(VERSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(versions, f, indent=2)

        console.print("[green]Path configured successfully![/green]")
        console.print(f"  llama-server: {dest}")
        return True

    return False


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
):
    """
    Install llama-server binary.

    Options:
      --llama-server <path>    Use existing llama-server binary
      --backend <type>         Backend: auto, vulkan, cuda, rocm, sycl, metal, cpu
      --force                  Force reinstall even if already installed

    If no options provided, shows interactive setup wizard.
    """
    if backend:
        valid = ["auto", "vulkan", "cuda", "rocm", "sycl", "metal", "cpu"]
        if backend.lower() not in valid:
            console.print(f"[red]Invalid backend: {backend}[/red]")
            console.print(f"Valid: {', '.join(valid)}")
            raise typer.Exit(1)

        versions = install_binaries(force=force, backend=backend.lower())
        console.print(f"[green]Installed {backend}:[/green]")
        console.print(f"  llama.cpp: {versions.get('llamacpp')}")
        return

    if llama_server_path:
        result = configure_custom_paths(llama_server_path=llama_server_path)
        if result:
            console.print("[green]Configuration saved![/green]")
        return

    # Interactive wizard
    console.print("[bold cyan]Kapri Setup Wizard[/bold cyan]\n")

    existing_llama_server = find_existing_llama_server()
    backend = None

    console.print("[dim]Checking for existing installations...[/dim]")

    console.print("\n[bold]llama-server (llama.cpp)[/bold]")

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

    console.print("\n[bold green]Setup complete![/green]")
    console.print("Run 'kapri serve' to start the server.")


@app.command()
def pull(
    model: str = typer.Argument(..., help="Model ID (e.g., llama3.2-3b)"),
    quant: str = typer.Option("Q4_K_XL", "--quant", "-q", help="Quantization"),
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

    ensure_server_running()

    models_in_config = load_config()

    # Fallback to local manifest if no models in INI
    if not models_in_config:
        for m in get_local_models():
            models_in_config[m["id"]] = {"model": m.get("path", "")}

    model_id = model
    model_normalized = (
        model.lower().replace("-", "").replace("_", "").replace("gguf", "")
    )

    if model in models_in_config:
        model_id = model
    else:
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

    ensure_server_running()

    if not tui:
        import webbrowser

        url = f"http://localhost:{DEFAULT_PORT}/"

        console.print(f"[green]Opening web UI:[/green] {url}")
        console.print(f"[dim]Server port: {DEFAULT_PORT}, model: {model_id}[/dim]")

        webbrowser.open(url)

        try:
            while True:
                import time

                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Web UI closed[/yellow]")
        return

    chat_url = f"http://localhost:{DEFAULT_PORT}/v1/chat/completions"

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
                chat_url,
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

    if json_output:
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
):
    """List downloaded models."""
    models = get_local_models()

    if json_output:
        console.print(json.dumps(models, indent=2))
        return

    if not models:
        console.print("[yellow]No models downloaded[/yellow]")
        console.print("Run 'kapri pull <model>' to download a model.")
        return

    table = Table(title=f"Models ({len(models)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Size")
    table.add_column("Context")

    for m in models:
        size = m.get("size_gb", "-")
        if size != "-":
            size = f"{size}GB"
        table.add_row(
            m.get("id", "-"),
            m.get("name", "-")[:30],
            size,
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
        False, "--all", "-a", help="Update kapri package and binary"
    ),
):
    """Update kapri and/or binary."""
    if everything:
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

        console.print("[blue]Updating binary...[/blue]")
        versions = install_binaries(force=True, backend=None)
        console.print(
            f"[green]Updated: {versions.get('llamacpp')}[/green]"
        )
    else:
        versions = install_binaries(force=True, backend=None)
        console.print(
            f"[green]Binary updated: {versions.get('llamacpp')}[/green]"
        )


@app.command("backend")
def set_backend(
    backend: str = typer.Argument(
        ...,
        help="Backend: auto, vulkan, cuda, rocm, sycl, metal, cpu",
    ),
    force: bool = typer.Option(True, "--force", "-f", help="Force reinstall"),
):
    """Change GPU backend and re-download binary."""
    valid_backends = ["auto", "vulkan", "cuda", "rocm", "sycl", "metal", "cpu"]

    backend = backend.lower()
    if backend not in valid_backends:
        console.print(f"[red]Invalid backend: {backend}[/red]")
        console.print(f"Valid: {', '.join(valid_backends)}")
        raise typer.Exit(1)

    console.print(f"[blue]Installing {backend} backend...[/blue]")
    versions = install_binaries(force=force, backend=backend)

    from .constants import VERSIONS_FILE

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


# Config subcommand group
config_app = typer.Typer(name="config", help="Manage configuration")


@config_app.command("show-all")
def show_all_config():
    """Show full models preset."""
    if MODELS_PRESET_FILE.exists():
        console.print(MODELS_PRESET_FILE.read_text())
    else:
        console.print("[yellow]No config found[/yellow]")


@config_app.command("show")
def show_full(
    path: bool = typer.Option(False, "--path", help="Show config path"),
    reset: bool = typer.Option(False, "--reset", help="Regenerate config"),
    model: Optional[str] = typer.Argument(None, help="Model ID to show"),
):
    """Show config, or use --path, --reset. Optionally show a specific model."""
    if path:
        console.print(str(MODELS_PRESET_FILE))
        return

    if reset:
        regenerate_config()
        console.print("[green]Config regenerated[/green]")
        return

    if model:
        cfg = load_config()
        if not cfg:
            console.print("[red]Config file not found[/red]")
            raise typer.Exit(1)

        found = None
        for s in cfg:
            if model.lower() == s.lower():
                found = s
                break
        if not found:
            model_lower = model.lower().replace("-", "").replace("_", "")
            for s in cfg:
                s_norm = s.lower().replace("-", "").replace("_", "")
                if model_lower in s_norm or s_norm in model_lower:
                    found = s
                    break
        if not found:
            console.print(f"[red]Model not found:[/red] {model}")
            raise typer.Exit(1)

        table = Table(title=f"Model: {found}")
        table.add_column("Property")
        table.add_column("Value")
        for key, value in cfg[found].items():
            table.add_row(key, value)
        console.print(table)
    else:
        if MODELS_PRESET_FILE.exists():
            console.print(MODELS_PRESET_FILE.read_text())
        else:
            console.print("[yellow]No config found[/yellow]")


app.add_typer(config_app, name="config")


@config_app.command("edit")
def config_edit_model(
    model: str = typer.Argument(..., help="Model ID to edit"),
):
    """Edit model configuration in editor."""
    if not MODELS_PRESET_FILE.exists():
        console.print("[red]Config file not found[/red]")
        raise typer.Exit(1)

    editor = os.environ.get("EDITOR", "notepad")
    console.print(f"[yellow]Opening config in editor:[/yellow] {editor}")
    console.print(f"[dim]Config file: {MODELS_PRESET_FILE}[/dim]")

    subprocess.run([editor, str(MODELS_PRESET_FILE)])


@config_app.command("search")
def config_search_model(
    query: str = typer.Argument(..., help="Search query"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
):
    """Search models in config by name."""
    cfg = load_config()

    query_lower = query.lower()
    results = []

    for section, items in cfg.items():
        if query_lower in section.lower():
            results.append((section, items))
            continue

    if not results:
        console.print(f"[yellow]No models found matching:[/yellow] {query}")
        return

    if json_output:
        console.print(json.dumps({m_id: m_data for m_id, m_data in results}, indent=2))
    else:
        table = Table(title=f"Results ({len(results)})")
        table.add_column("ID")
        table.add_column("Model Path")
        for m_id, m_data in results:
            table.add_row(m_id, m_data.get("model", "-")[:60])
        console.print(table)


# Model config subcommand (backward compat)
model_app = typer.Typer(help="Manage model configurations")


@model_app.command("show")
def model_config_show(
    model: str = typer.Argument(..., help="Model ID or name"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
):
    """
    Show model configuration details.

    Example:
        kapri model config show qwen3.5-0.8b
    """
    cfg = load_config()

    found = None
    for s in cfg:
        if model.lower() == s.lower():
            found = s
            break

    if not found:
        console.print(f"[red]Model not found:[/red] {model}")
        raise typer.Exit(1)

    items = cfg[found]

    if json_output:
        console.print(json.dumps({found: items}, indent=2))
    else:
        table = Table(title=f"Model: {found}")
        table.add_column("Property")
        table.add_column("Value")
        for key, value in items.items():
            table.add_row(key, value)
        console.print(table)


@model_app.command("search")
def model_config_search(
    query: str = typer.Argument(..., help="Search query"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
):
    """
    Search models in config by name.

    Example:
        kapri model config search qwen
    """
    cfg = load_config()

    query_lower = query.lower()
    results = []

    for section, items in cfg.items():
        if query_lower in section.lower():
            results.append((section, items))

    if not results:
        console.print(f"[yellow]No models found matching:[/yellow] {query}")
        return

    if json_output:
        console.print(json.dumps({m_id: m_data for m_id, m_data in results}, indent=2))
    else:
        table = Table(title=f"Results ({len(results)})")
        table.add_column("ID")
        table.add_column("Model Path")
        for m_id, m_data in results:
            table.add_row(m_id, m_data.get("model", "-")[:60])
        console.print(table)


@model_app.command("edit")
def model_config_edit(
    model: str = typer.Argument(..., help="Model ID or name"),
):
    """
    Edit model configuration (opens in editor).

    Example:
        kapri model config edit qwen3.5-0.8b
    """
    editor = os.environ.get("EDITOR", "notepad")

    console.print(f"[yellow]Opening config in editor:[/yellow] {editor}")
    console.print(f"[dim]Config file: {MODELS_PRESET_FILE}[/dim]")

    subprocess.run([editor, str(MODELS_PRESET_FILE)])


app.add_typer(model_app, name="model")


@app.command()
def version():
    """Show version."""
    console.print(f"Kapri {__version__}")


# For pip install -e
def main():
    app()


if __name__ == "__main__":
    main()
