"""Kapri utilities."""

import os
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def ensure_dir(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def get_terminal_size() -> tuple[int, int]:
    """Get terminal size (cols, rows)."""
    try:
        from shutil import get_terminal_size

        size = get_terminal_size()
        return size.columns, size.lines
    except Exception:
        return 80, 24


def confirm(prompt: str, default: bool = False) -> bool:
    """Ask for confirmation."""
    suffix = "Y/n" if default else "y/n"
    while True:
        response = console.input(f"{prompt} [{suffix}]: ").lower().strip()
        if not response:
            return default
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False


def error(msg: str) -> None:
    """Print error message."""
    console.print(f"[red]Error:[/red] {msg}")


def warn(msg: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]Warning:[/yellow] {msg}")


def info(msg: str) -> None:
    """Print info message."""
    console.print(f"[blue]Info:[/blue] {msg}")


def success(msg: str) -> None:
    """Print success message."""
    console.print(f"[green]Success:[/green] {msg}")
