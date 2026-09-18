"""Locating files, which differs between running from source and running frozen.

A PyInstaller bundle is read-only and is launched with the working directory set
to ``/``, so neither the bundled ``configs/`` nor a relative ``out/`` path means
what it does during development. This module gives the rest of the GUI two
answers it can rely on:

* :func:`bundled` -- read-only files shipped inside the app.
* :func:`workspace` -- a writable directory for configs, datasets and
  checkpoints, which is the repo itself from source and ``~/LLMS-Studio`` frozen.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

APP_NAME = "LLMS-Studio"


def is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def bundled(*parts: str) -> Path:
    """Path to a read-only resource shipped with the app."""
    if is_frozen():
        # PyInstaller unpacks datas into _MEIPASS.
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).resolve().parent.parent
    return base.joinpath(*parts)


def workspace() -> Path:
    """The writable directory that datasets, checkpoints and configs live in."""
    if is_frozen():
        return Path.home() / APP_NAME
    return Path(__file__).resolve().parent.parent


def prepare_workspace() -> Path:
    """Create the workspace and seed it with the bundled configs.

    Existing files are never overwritten, so a config the user has edited
    survives an app update.
    """
    ws = workspace()
    if not is_frozen():
        return ws

    (ws / "configs").mkdir(parents=True, exist_ok=True)
    src = bundled("configs")
    if src.is_dir():
        for config in src.glob("*.yaml"):
            dest = ws / "configs" / config.name
            if not dest.exists():
                shutil.copyfile(config, dest)
    return ws


def child_command(subcommand: str, args: list[str]) -> list[str]:
    """Build the argv for re-invoking this program to do background work.

    Frozen, ``sys.executable`` is the app binary and takes the subcommand
    directly. From source it is the interpreter, which needs the launcher
    script's path first.
    """
    if is_frozen():
        return [sys.executable, subcommand, *args]
    launcher = Path(__file__).resolve().parent.parent / "llms_studio.py"
    return [sys.executable, str(launcher), subcommand, *args]
