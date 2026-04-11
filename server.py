"""Kapri server management - start, stop, status."""

import json
import pathlib
import socket
import subprocess
import time
from datetime import datetime
from typing import Optional

import psutil
from rich.console import Console

from .config import CONFIG_FILE, regenerate_config
from .constants import (
    BIN_DIR,
    DEFAULT_PORT,
    LLAMASWAP_BIN,
    LOG_FILE,
    PID_FILE,
    START_PORT,
)
from .models import get_local_models

console = Console()


def start_server(port: int = DEFAULT_PORT, foreground: bool = False) -> None:
    """
    Start llama-swap server.
    """
    # Check if already running
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            if psutil.pid_exists(pid):
                # Check process name
                try:
                    proc = psutil.Process(pid)
                    if "llama" in proc.name().lower():
                        console.print(
                            f"[yellow]Server already running on port {port}[/yellow]"
                        )
                        return
                except psutil.NoSuchProcess:
                    pass
            # Stale PID
            PID_FILE.unlink()
        except (ValueError, FileNotFoundError):
            pass

    # Ensure config exists
    if not CONFIG_FILE.exists():
        regenerate_config()

    # Check models
    models = get_local_models()
    if not models:
        console.print(
            "[yellow]Warning: No models downloaded. "
            "Run 'kapri pull <model>' first.[/yellow]"
        )

    # Get binary
    swap_bin = BIN_DIR / LLAMASWAP_BIN
    if not swap_bin.exists():
        for name in ["llama-swap", "llama-swap.exe"]:
            alt = BIN_DIR / name
            if alt.exists():
                swap_bin = alt
                break

    if not swap_bin.exists():
        raise RuntimeError(f"llama-swap not found. Run 'kapri install' first.")

    # Start process
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as logf:
        proc = subprocess.Popen(
            [
                str(swap_bin),
                "--config",
                str(CONFIG_FILE),
                "--listen",
                f"0.0.0.0:{port}",
            ],
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Write PID
    PID_FILE.write_text(str(proc.pid))

    # Wait for port
    console.print(f"[blue]Starting server...[/blue]")
    for i in range(16):
        time.sleep(0.5)
        if socket.connect_ex(("127.0.0.1", port)) == 0:
            break
    else:
        # Show logs
        if LOG_FILE.exists():
            last_lines = LOG_FILE.read_text().splitlines()[-20:]
            console.print("[red]Server failed to start[/red]")
            console.print("\n".join(last_lines))
        raise RuntimeError("Server timeout")

    console.print(f"[bold green]Kapri is running[/bold green]")
    console.print(f"  Port: {port}")
    console.print(f"  Config: {CONFIG_FILE}")
    console.print(f"  Logs: {LOG_FILE}")


def stop_server() -> None:
    """Stop server process."""
    if not PID_FILE.exists():
        console.print("[yellow]Server is not running[/yellow]")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        proc = psutil.Process(pid)
        proc.terminate()

        # Wait 3s
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

        # Get uptime
        started = proc.create_time()
        status["uptime_seconds"] = int(time.time() - started)

        # Get loaded models
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
