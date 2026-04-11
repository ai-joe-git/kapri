"""Kapri CLI - main entry point."""

import json
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import regenerate_config, load_config, CONFIG_FILE
from .installer import install_binaries, detect_backend, get_current_versions
from .models import (
    pull_model,
    remove_model,
    list_models,
    get_model_info,
    get_local_models,
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


@app.command()
def install(
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        "-b",
        help="GPU backend: cuda, vulkan, rocm, sycl, metal, cpu",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall"),
):
    """Install llama-server and llama-swap binaries."""
    if backend:
        valid = ["cuda", "vulkan", "rocm", "sycl", "metal", "cpu"]
        if backend not in valid:
            console.print(f"[red]Invalid backend: {backend}[/red]")
            console.print(f"Valid: {', '.join(valid)}")
            raise typer.Exit(1)

    versions = install_binaries(force=force, backend=backend)

    table = Table(title="Installed Binaries")
    table.add_column("Component")
    table.add_column("Version")
    table.add_column("Backend")

    table.add_row(
        "llama.cpp", versions.get("llamacpp", "?"), versions.get("backend", "?")
    )
    table.add_row("llama-swap", versions.get("llamaswap", "?"), "-")

    console.print(table)


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
):
    """Start an interactive chat session."""
    # Ensure server running
    ensure_server_running()

    console.print(f"[blue]Starting chat with {model}...[/blue]")
    console.print("[dim]Type /exit to quit, /clear to reset[/dim]")

    import httpx

    history = []

    if system:
        history.append({"role": "system", "content": system})

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

        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"http://localhost:{DEFAULT_PORT}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": history,
                        "stream": False,
                    },
                )
                if resp.status_code != 200:
                    console.print(f"[red]Error:[/red] {resp.text}")
                    continue

                data = resp.json()
                assistant_msg = data["choices"][0]["message"]
                content = assistant_msg["content"]

                console.print(f"\n[blue]Assistant:[/blue] {content}")
                history.append(assistant_msg)

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
def status(json: bool = typer.Option(False, "--json", help="JSON output")):
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
):
    """List downloaded models."""
    models = get_local_models()

    if json_output:
        console.print(json.dumps(models, indent=2))
        return

    if not models:
        console.print("[yellow]No models downloaded[/yellow]")
        console.print("Run: kapri pull <model>")
        return

    table = Table(title=f"Models ({len(models)})")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Size")
    table.add_column("Quant")
    table.add_column("Context")

    for m in models:
        table.add_row(
            m.get("id", "-"),
            m.get("name", "-")[:30],
            f"{m.get('size_gb', 0):.1f}GB",
            m.get("quant", "-"),
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
def update():
    """Update binaries and registry."""
    versions = install_binaries(force=True)
    console.print("[green]Update complete[/green]")


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
