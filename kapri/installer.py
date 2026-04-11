"""Kapri installer - GPU detection and binary download."""

import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from typing import Optional

import httpx
from rich.console import Console
from rich.progress import (
    Progress,
    DownloadColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .constants import (
    BASE_DIR,
    BIN_DIR,
    LLAMACPP_REPO,
    LLAMASWAP_REPO,
    VERSIONS_FILE,
    LLAMASERVER_BIN,
    LLAMASWAP_BIN,
)

console = Console()


# ==== GPU Detection ====


def detect_backend() -> str:
    """
    Detect GPU backend. Priority: CUDA > ROCm > Vulkan > SYCL > CPU
    macOS: always return 'metal' (standard build includes it)
    """
    system = platform.system()

    # macOS - Metal included in standard build
    if system == "Darwin":
        return "metal"

    # CUDA
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # ROCm
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "rocm"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Vulkan
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Filter out software renderers
            output = result.stdout
            if "llvmpipe" in output.lower() or "softpipe" in output.lower():
                pass  # Fall through to next check
            elif "microsoft basic render driver" in output.lower():
                pass  # Fall through to next check
            else:
                # Check for real GPU
                if "deviceName" in output or "GPU" in output:
                    return "vulkan"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # SYCL
    if os.environ.get("ONEAPI_ROOT"):
        return "sycl"
    try:
        result = subprocess.run(
            ["icpx", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return "sycl"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # CPU fallback
    return "cpu"


def get_platform() -> str:
    """Returns 'windows', 'linux', or 'macos'."""
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Darwin":
        return "macos"
    return "linux"


def get_arch() -> str:
    """Returns 'x64' or 'arm64'."""
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    elif machine in ("aarch64", "arm64"):
        return "arm64"
    return "x64"


# ==== Asset Name Resolution ====


def get_llamacpp_asset_name(backend: str, os_name: str, arch: str) -> str:
    """
    Map backend/os/arch to exact llama.cpp asset name.

    Current release: b8762
    """
    # macOS: standard build includes Metal
    if os_name == "macos":
        if arch == "arm64":
            return "llama-b8762-bin-macos-arm64.tar.gz"
        return "llama-b8762-bin-macos-x64.tar.gz"

    # Linux
    if os_name == "linux":
        if backend == "vulkan":
            return f"llama-b8762-bin-ubuntu-vulkan-{arch}.tar.gz"
        elif backend == "rocm":
            return "llama-b8762-bin-ubuntu-rocm-7.2-x64.tar.gz"
        elif backend == "cpu":
            if arch == "arm64":
                return "llama-b8762-bin-ubuntu-arm64.tar.gz"
            return "llama-b8762-bin-ubuntu-x64.tar.gz"

    # Windows
    if os_name == "windows":
        if backend == "cuda":
            return "llama-b8762-bin-win-cuda-13.1-x64.zip"
        elif backend == "vulkan":
            return "llama-b8762-bin-win-vulkan-x64.zip"
        elif backend == "sycl":
            return "llama-b8762-bin-win-sycl-x64.zip"
        elif backend == "cpu":
            if arch == "arm64":
                return "llama-b8762-bin-win-cpu-arm64.zip"
            return "llama-b8762-bin-win-cpu-x64.zip"

    # Default to CPU
    if os_name == "linux":
        return "llama-b8762-bin-ubuntu-x64.tar.gz"
    return "llama-b8762-bin-win-cpu-x64.zip"


def get_llamaswap_asset_name(os_name: str, arch: str) -> str:
    """
    Map os/arch to exact llama-swap asset name.

    Based on research from Step 1B (release v200):
    """
    if os_name == "macos":
        if arch == "arm64":
            return "llama-swap_200_darwin_arm64.tar.gz"
        return "llama-swap_199_darwin_amd64.tar.gz"

    if os_name == "windows":
        return "llama-swap_200_windows_amd64.zip"

    # Linux
    if arch == "arm64":
        return "llama-swap_200_linux_arm64.tar.gz"
    return "llama-swap_200_linux_amd64.tar.gz"


# ==== GitHub Release Fetcher ====


def fetch_latest_release_asset_url(repo: str, asset_name: str) -> str:
    """
    Fetch asset download URL from GitHub releases.
    """
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"

    with httpx.Client(timeout=30.0) as client:
        response = client.get(api_url)
        response.raise_for_status()
        data = response.json()

        # Find asset
        for asset in data.get("assets", []):
            if asset["name"] == asset_name:
                return asset["browser_download_url"]

        # Asset not found - list available
        available = [a["name"] for a in data.get("assets", [])]
        raise ValueError(f"Asset '{asset_name}' not found. Available: {available}")


def get_current_versions() -> dict:
    """Read current installed versions."""
    if not VERSIONS_FILE.exists():
        return {}
    import json

    with open(VERSIONS_FILE, "r") as f:
        return json.load(f)


def save_versions(versions: dict) -> None:
    """Save installed versions."""
    import json
    from datetime import datetime

    versions["installed_at"] = datetime.utcnow().isoformat() + "Z"
    VERSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(VERSIONS_FILE, "w") as f:
        json.dump(versions, f, indent=2)


# ==== Download + Extract ====


def download_and_extract(
    url: str, dest_dir: pathlib.Path, progress: bool = True
) -> pathlib.Path:
    """
    Download file and extract if needed.
    Returns path to the extracted binary.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Get filename from URL
    filename = url.split("/")[-1]
    temp_path = dest_dir / filename

    # Download with progress
    with httpx.Client(follow_redirects=True, timeout=300.0) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            with open(temp_path, "wb") as f:
                if progress and total_size > 0:
                    with Progress(
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(),
                        DownloadColumn(),
                        TextColumn("{task.percentage:>3.0f}%"),
                        TimeRemainingColumn(),
                        console=console,
                    ) as p:
                        task = p.add_task("Downloading...", total=total_size)
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            p.update(task, advance=len(chunk))
                else:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)

    # Extract
    extracted_bin = None
    if filename.endswith(".zip"):
        with zipfile.ZipFile(temp_path, "r") as zf:
            # Find binary
            for name in zf.namelist():
                if "llama-server" in name or "llama-swap" in name:
                    ext = ".exe" if get_platform() == "windows" else ""
                    if name.endswith(ext) and not name.endswith("/"):
                        zf.extract(name, dest_dir)
                        extracted_bin = dest_dir / name
                        break
    elif filename.endswith(".tar.gz"):
        with tarfile.open(temp_path, "r:gz") as tf:
            for member in tf.getmembers():
                if "llama-server" in member.name or "llama-swap" in member.name:
                    ext = ".exe" if get_platform() == "windows" else ""
                    if member.name.endswith(ext) and not member.isdir():
                        tf.extract(member, dest_dir)
                        extracted_bin = dest_dir / member.name
                        break

    # Clean up archive
    if temp_path.exists():
        temp_path.unlink()

    if extracted_bin:
        # Make executable
        if get_platform() != "windows":
            extracted_bin.chmod(0o755)
        return extracted_bin

    raise RuntimeError(f"Could not extract binary from {filename}")


# ==== Main Install Function ====


def install_binaries(force: bool = False, backend: Optional[str] = None) -> dict:
    """
    Install llama-server and llama-swap binaries.
    """
    # Detect platform
    os_name = get_platform()
    arch = get_arch()

    # Detect or validate backend
    if backend is None:
        backend = detect_backend()

    console.print(f"[bold]Detected:[/bold] {os_name}, {arch}, {backend}")

    # Check if already installed
    if not force:
        versions = get_current_versions()
        if versions.get("llamacpp") and versions.get("llamaswap"):
            console.print(
                f"[green]Binaries already installed:[/green] "
                f"llama.cpp {versions.get('llamacpp')}, "
                f"llama-swap {versions.get('llamaswap')}"
            )
            return versions

    # Ensure directories exist
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    # Get asset names
    llamacpp_asset = get_llamacpp_asset_name(backend, os_name, arch)
    llamaswap_asset = get_llamaswap_asset_name(os_name, arch)

    console.print(f"[bold]Installing Kapri binaries...[/bold]")
    console.print(f"  llama.cpp: {llamacpp_asset}")
    console.print(f"  llama-swap: {llamaswap_asset}")

    # Fetch URLs
    llamacpp_url = fetch_latest_release_asset_url(LLAMACPP_REPO, llamacpp_asset)
    llamaswap_url = fetch_latest_release_asset_url(LLAMASWAP_REPO, llamaswap_asset)

    # Download llama-server
    console.print("[blue]Downloading llama-server...[/blue]")
    server_bin = download_and_extract(llamacpp_url, BIN_DIR)

    # Download llama-swap
    console.print("[blue]Downloading llama-swap...[/blue]")
    swap_bin = download_and_extract(llamaswap_url, BIN_DIR)

    # Get version info from asset names
    import re

    llamacpp_version = re.search(r"b(\d+)", llamacpp_asset)
    llamaswap_version = re.search(r"_(\d+)", llamaswap_asset)

    versions = {
        "llamacpp": f"b{llamacpp_version.group(1) if llamacpp_version else 'unknown'}",
        "llamaswap": f"v{llamaswap_version.group(1) if llamaswap_version else 'unknown'}",
        "backend": backend,
    }

    save_versions(versions)

    # Success summary
    console.print("\n[bold green]Installation complete![/bold green]")
    console.print(f"  Binary directory: {BIN_DIR}")
    console.print(f"  Backend: {backend}")

    return versions


if __name__ == "__main__":
    install_binaries()
