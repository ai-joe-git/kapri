"""Kapri server management - start, stop, status."""

import json
import os
import pathlib
import socket
import subprocess
import time
from typing import Optional

import psutil
from rich.console import Console

from .config import regenerate_config, MODELS_PRESET_FILE
from .constants import (
    BIN_DIR,
    DEFAULT_PORT,
    LLAMASERVER_BIN,
    LOG_FILE,
    PID_FILE,
)
from .models import get_local_models

console = Console()


def load_settings() -> dict:
    """Load kapri settings from versions file."""
    from .constants import VERSIONS_FILE

    if not VERSIONS_FILE.exists():
        return {}
    with open(VERSIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def start_server(port: int = DEFAULT_PORT, foreground: bool = False) -> None:
    """
    Start llama.cpp server in router mode.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            s.close()
        except OSError:
            console.print(
                f"[yellow]Port {port} already in use. Checking for existing server...[/yellow]"
            )

    regenerate_config()

    kapri_models = get_local_models()
    if not kapri_models:
        console.print(
            "[yellow]Warning: No models. Run 'kapri pull <model>' first.[/yellow]"
        )

    server_bin = BIN_DIR / LLAMASERVER_BIN

    if not server_bin.exists():
        from .installer import install_binaries

        console.print("[dim]Installing llama-server...[/dim]")
        install_binaries(force=False, backend=None)
        server_bin = BIN_DIR / LLAMASERVER_BIN

    if not server_bin.exists():
        raise RuntimeError(f"llama-server not found. Run 'kapri install' first.")

    env = os.environ.copy()
    env["GGML_VK_NO_PIPELINE_CACHE"] = "1"
    env["GGML_VK_DISABLE_COOPMAT"] = "1"
    env["GGML_VK_DISABLE_COOPMAT2"] = "1"

    args = [
        str(server_bin),
        "--models-preset", str(MODELS_PRESET_FILE),
        "--models-max", "1",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--metrics",
        "--tools", "all",
    ]

    if foreground:
        subprocess.run(args, env=env)
        return

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as logf:
        proc = subprocess.Popen(
            args,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    PID_FILE.write_text(str(proc.pid))

    console.print(f"[blue]Starting server...[/blue]")
    for i in range(16):
        time.sleep(0.5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
    else:
        if LOG_FILE.exists():
            last_lines = LOG_FILE.read_text().splitlines()[-20:]
            console.print("[red]Server failed to start[/red]")
            console.print("\n".join(last_lines))
        raise RuntimeError("Server timeout")

    console.print(f"[bold green]Kapri is running[/bold green]")
    console.print(f" Port: {port}")
    console.print(f" Preset: {MODELS_PRESET_FILE}")
    console.print(f" Logs: {LOG_FILE}")


def stop_server() -> None:
    """Stop server process."""
    if not PID_FILE.exists():
        console.print("[yellow]Server is not running[/yellow]")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        proc = psutil.Process(pid)
        proc.terminate()

        try:
            proc.wait(timeout=3)
        except psutil.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)

    except (ValueError, FileNotFoundError, psutil.NoSuchProcess):
        pass
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()

    console.print("[green]Server stopped[/green]")


def server_status() -> dict:
    """
    Get server status.
    """
    status = {
        "running": False,
        "pid": None,
        "port": DEFAULT_PORT,
        "uptime_seconds": None,
        "loaded_models": [],
    }

    if not PID_FILE.exists():
        return status

    try:
        pid = int(PID_FILE.read_text().strip())
        if not psutil.pid_exists(pid):
            return status

        proc = psutil.Process(pid)
        if "llama" not in proc.name().lower():
            return status

        status["running"] = True
        status["pid"] = pid

        started = proc.create_time()
        status["uptime_seconds"] = int(time.time() - started)

        try:
            import httpx

            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"http://localhost:{DEFAULT_PORT}/v1/models")
                if resp.status_code == 200:
                    data = resp.json()
                    status["loaded_models"] = [m["id"] for m in data.get("data", [])]
        except Exception:
            pass

    except (ValueError, FileNotFoundError, psutil.NoSuchProcess):
        pass

    return status


def ensure_server_running() -> None:
    """Start server if not running."""
    if not server_status()["running"]:
        start_server()


def tail_logs(lines: int = 50) -> str:
    """Get last N lines from log file."""
    if not LOG_FILE.exists():
        return ""

    content = LOG_FILE.read_text()
    log_lines = content.splitlines()
    return "\n".join(log_lines[-lines:])


if __name__ == "__main__":
    status = server_status()
    print(json.dumps(status, indent=2))
