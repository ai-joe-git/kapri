"""Kapri server management - start, stop, status."""

import json
import os
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
    LLAMASERVER_BIN,
    LLAMASWAP_BIN,
    LOG_FILE,
    PID_FILE,
    START_PORT,
)
from .models import get_local_models

console = Console()


def load_settings() -> dict:
    """Load kapri settings from versions file."""
    import json
    from .constants import VERSIONS_FILE

    if not VERSIONS_FILE.exists():
        return {}

    with open(VERSIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def start_server(port: int = DEFAULT_PORT, foreground: bool = False) -> None:
    """
    Start llama-swap server.
    """
    # Load settings
    settings = load_settings()

    # Check if already running by checking the port
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            # Port is free, we can start
            s.close()
        except OSError:
            # Port is in use - check if it's a llama-swap process
            console.print(
                f"[yellow]Port {port} already in use. Checking for existing server...[/yellow]"
            )
            # Continue - the old server will be replaced

    # Determine config source - use internal config, import from original only once at install time

    imported_config_path = settings.get("imported_config")

    # Prefer imported config, fallback to kapri's own config
    if imported_config_path and pathlib.Path(imported_config_path).exists():
        local_config = pathlib.Path(imported_config_path)
    else:
        # Use kapri's own config - don't reference original llama-swap config
        local_config = None

    # Load config
    import yaml

    # Always use kapri's config - create if needed
    config_data = {}
    if local_config and local_config.exists():
        with open(local_config, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        # Keep copy of original for extracting hooks/groups later
        original_config_data = dict(config_data)

        # Remove startPort from config to use --listen parameter instead
        # This ensures llama-swap listens on the port we specify (DEFAULT_PORT = 11434)
        if "startPort" in config_data:
            del config_data["startPort"]

        # Ensure macros are defined for consistent model config format
        if "macros" not in config_data:
            config_data["macros"] = {
                "llama_server": '"${env.LLAMA_SERVER}"',
                "listen_args": "--host 0.0.0.0 --port ${PORT}",
            }
        # Don't add llama_cli macro - not used anymore

        # Increase healthCheckTimeout for slower models (especially vision models)
        config_data["healthCheckTimeout"] = 60000  # 60 seconds

        # Add globalTTL for time-based model unloading
        # Models stay loaded while in use, unload after idle to free VRAM
        # llama-swap stays running in background (light process)
        config_data["globalTTL"] = 600  # 10 minutes - unload models after 10min idle

        # Filter out non-llama.cpp models (whisper STT, omnivoice TTS)
        # These are external servers, not chat models
        external_models = ["whisper-large-v3-turbo", "omnivoice-tts"]
        models_to_remove = [
            m for m in external_models if m in config_data.get("models", {})
        ]
        for m in models_to_remove:
            del config_data["models"][m]
            console.print(f"[dim]Skipped external model: {m}[/dim]")

        # Remove hooks/preload for external models
        if "hooks" in config_data and isinstance(config_data.get("hooks"), dict):
            hooks = config_data["hooks"]
            if "on_startup" in hooks and isinstance(hooks["on_startup"], dict):
                on_startup = hooks["on_startup"]
                if "preload" in on_startup and isinstance(on_startup["preload"], list):
                    preload = on_startup["preload"]
                    filtered_preload = [p for p in preload if p not in external_models]
                    on_startup["preload"] = filtered_preload
                    if not filtered_preload:
                        del config_data["hooks"]

        # Remove groups for external models
        if "groups" in config_data and isinstance(config_data["groups"], dict):
            groups_to_remove = []
            for g_name, g_data in config_data["groups"].items():
                members = g_data.get("members", [])
                if isinstance(members, list) and any(
                    m in external_models for m in members
                ):
                    groups_to_remove.append(g_name)
            for g in groups_to_remove:
                del config_data["groups"][g]
            if not config_data["groups"]:
                del config_data["groups"]

        # Get kapri-downloaded models

        # Get kapri-downloaded models
        kapri_models = get_local_models()

        # Add kapri models to config if not already there
        for km in kapri_models:
            model_id = km["id"]
            if model_id not in config_data.get("models", {}):
                model_path = pathlib.Path(km["path"])
                if model_path.exists():
                    # Resolve path with forward slashes
                    resolved_path = str(model_path.resolve()).replace("\\", "/")

                    # Get llama-server from settings or use kapri bin
                    custom_server = settings.get("custom_llama_server")
                    if custom_server:
                        local_llama_server = pathlib.Path(custom_server)
                    else:
                        # Use kapri's downloaded llama-server
                        local_llama_server = BIN_DIR / LLAMASERVER_BIN

                    ctx = km.get("context", 32768)
                    # Use smaller context for faster loading
                    ctx = min(ctx, 32768)

                    # Build command in the same format as user's config
                    # Using macros like ${llama_server} ${listen_args}
                    vulkan_env = [
                        "GGML_VK_NO_PIPELINE_CACHE=1",
                        "VK_DISABLE_PIPELINE_CACHE=1",
                        "GGML_VK_DISABLE_COOPMAT=1",
                        "GGML_VK_DISABLE_COOPMAT2=1",
                    ]

                    # Use the exact format from user's config
                    model_entry = {
                        "name": km.get("name", model_id),
                        "description": f"Kapri model: {km.get('name', model_id)}",
                        "env": vulkan_env,
                        "cmd": (
                            f"${{llama_server}} ${{listen_args}} "
                            f'-m "{resolved_path}" '
                            f"-ngl 99 "
                            f"--jinja "
                            f"-fa on "
                            f"--temp 0.7 "
                            f"-c {ctx} "
                            f"--top-p 1.0 "
                            f"--top-k 0 "
                            f"--parallel 1 "
                            f"--no-warmup"
                        ),
                    }

                    config_data["models"][model_id] = model_entry
                    console.print(f"[dim]Added kapri model to config: {model_id}[/dim]")

        # Copy hooks from original config if they exist
        # BUT skip whisper/tts hooks - these are external STT/TTS servers, not llama.cpp models
        if "hooks" in original_config_data:
            hooks = original_config_data["hooks"]
            if "on_startup" in hooks and "preload" in hooks["on_startup"]:
                # Filter out non-llama.cpp models (whisper, omnivoice, etc.)
                preload = hooks["on_startup"]["preload"]
                chat_models = [
                    m
                    for m in preload
                    if m not in ["whisper-large-v3-turbo", "omnivoice-tts"]
                ]
                if chat_models:
                    config_data["hooks"] = {"on_startup": {"preload": chat_models}}
                    console.print("[dim]Copied only chat model hooks[/dim]")

        # Copy groups from original config (for persistent models like whisper)
        # BUT skip - these are external STT/TTS groups, not for kapri
        # if "groups" in original_config_data:
        #     config_data["groups"] = original_config_data["groups"]
        #     console.print("[dim]Copied groups from original config[/dim]")

        # Write merged config to kapri's config location
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        console.print(f"[dim]Using merged config: {CONFIG_FILE}[/dim]")
        console.print(
            f"[dim]globalTTL: {config_data.get('globalTTL', 'default')}s[/dim]"
        )

    # Ensure config exists - regenerate if needed
    config_path = CONFIG_FILE
    if not config_path.exists():
        regenerate_config()

    # Check models - from both kapri and config
    kapri_models = get_local_models()
    config_has_models = False
    if CONFIG_FILE.exists():
        import yaml

        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        config_models = config_data.get("models", {})
        # Exclude external models
        config_has_models = any(
            m not in ["whisper-large-v3-turbo", "omnivoice-tts"] for m in config_models
        )

    if not kapri_models and not config_has_models:
        console.print(
            "[yellow]Warning: No models. "
            "Run 'kapri pull <model>' or 'kapri install --import-config <path>'[/yellow]"
        )

    # Get binary - use kapri's bin directory
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    swap_bin = BIN_DIR / LLAMASWAP_BIN

    # If not in kapri bin, try to install
    if not swap_bin.exists():
        from .installer import install_binaries

        console.print("[dim]Installing llama-swap...[/dim]")
        install_binaries(force=False, backend=None)
        swap_bin = BIN_DIR / LLAMASWAP_BIN

    if not swap_bin.exists():
        raise RuntimeError(f"llama-swap not found. Run 'kapri install' first.")

    # Get llama-server from kapri bin
    llama_server_bin = BIN_DIR / LLAMASERVER_BIN

    if not llama_server_bin.exists():
        from .installer import install_binaries

        console.print("[dim]Installing llama-server...[/dim]")
        install_binaries(force=False, backend=None)

    # Build environment
    env = os.environ.copy()
    if llama_server_bin.exists():
        env["LLAMA_SERVER"] = str(llama_server_bin.resolve())

    # Start process
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as logf:
        proc = subprocess.Popen(
            [
                str(swap_bin),
                "--config",
                str(config_path),
                "--listen",
                f"0.0.0.0:{port}",
            ],
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    # Write PID
    PID_FILE.write_text(str(proc.pid))

    # Wait for port
    console.print(f"[blue]Starting server...[/blue]")
    for i in range(16):
        time.sleep(0.5)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
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
