#!/usr/bin/env python3
"""Entry point for LLMS Studio.

This is both the GUI launcher and the dispatcher the GUI re-invokes for
background work.

Why the dispatch: inside a PyInstaller bundle ``sys.executable`` is the app's
own binary, not a Python interpreter, so the GUI cannot spawn
``python -m llms.train``. Instead it spawns *this same program* with a
subcommand, and the branch below routes to the right CLI. Running from source
works identically, with ``sys.executable`` being the real interpreter.

Why the logging: a windowed bundle has no console, so an exception during
startup is invisible -- the app just fails to appear. Everything is mirrored to
``~/LLMS-Studio/llms-studio.log`` so a failed launch can actually be diagnosed.
"""

from __future__ import annotations

import datetime as _dt
import sys
import traceback
from pathlib import Path

SUBCOMMANDS = ("train", "sample")


def _log_path() -> Path:
    directory = Path.home() / "LLMS-Studio"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "llms-studio.log"


class _Tee:
    """Write to the log file and, when there is one, the real stream."""

    def __init__(self, handle, passthrough) -> None:
        self._handle = handle
        self._passthrough = passthrough

    def write(self, text: str) -> int:
        self._handle.write(text)
        self._handle.flush()
        if self._passthrough is not None:
            try:
                self._passthrough.write(text)
                self._passthrough.flush()
            except (ValueError, OSError):
                self._passthrough = None  # closed under us; keep the file going
        return len(text)

    def flush(self) -> None:
        self._handle.flush()

    def isatty(self) -> bool:
        return False


def _start_logging():
    """Mirror stdout/stderr into the log file. Returns the open handle."""
    handle = _log_path().open("a", encoding="utf-8", buffering=1)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    handle.write(f"\n=== LLMS Studio start {stamp} (frozen={getattr(sys, 'frozen', False)}) ===\n")
    sys.stdout = _Tee(handle, sys.__stdout__)
    sys.stderr = _Tee(handle, sys.__stderr__)
    return handle


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in SUBCOMMANDS:
        # Child processes inherit the GUI's pipes, which already reach the log.
        subcommand, rest = argv[0], argv[1:]
        if subcommand == "train":
            from llms.train import main as train_main

            train_main(rest)
        else:
            from llms.sample import main as sample_main

            sample_main(rest)
        return

    _start_logging()
    try:
        _serve(argv)
    except BaseException:
        print("FATAL: the app failed to start\n" + traceback.format_exc())
        raise


def _serve(argv: list[str]) -> None:
    """Start the local server and point a browser at it."""
    import threading
    import webbrowser

    import uvicorn

    from gui.server import HOST, find_free_port
    from gui.server import app as fastapi_app

    port = find_free_port()
    url = f"http://{HOST}:{port}/"
    print(f"serving on {url}")

    if "--no-browser" not in argv:
        # Wait for uvicorn to bind before opening the tab, or the browser races
        # the server and shows a connection error.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(fastapi_app, host=HOST, port=port, log_level="warning")


if __name__ == "__main__":
    main()
