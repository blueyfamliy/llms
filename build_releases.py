#!/usr/bin/env python3
"""Build the LLMS Studio desktop app.

    python build_releases.py

Everything about the build lives in ``LLMS-Studio.spec`` -- entry point, bundled
data, hidden imports, icon, Info.plist. This script only regenerates the icon
and invokes PyInstaller on that spec, so there is one definition of the build
rather than two that can drift apart.

Output lands in ``dist/``: ``LLMS-Studio.app`` on macOS, a ``LLMS-Studio/``
directory elsewhere.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "LLMS-Studio.spec"
ICON = ROOT / "assets" / "icon.icns"


def ensure_icon() -> None:
    """Regenerate the icon if it is missing. Needs Pillow; not fatal without it."""
    if ICON.exists():
        return
    print("icon not found, generating...")
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "make_icon.py")], check=True, cwd=ROOT
        )
    except subprocess.CalledProcessError:
        print("could not generate the icon (is Pillow installed?); building without it")


def build() -> int:
    print(f"building for {platform.system()} ({platform.machine()})")

    if shutil.which("pyinstaller") is None:
        # Prefer the module form: it always matches the interpreter running this.
        pyinstaller = [sys.executable, "-m", "PyInstaller"]
    else:
        pyinstaller = ["pyinstaller"]

    ensure_icon()

    for stale in (ROOT / "build", ROOT / "dist"):
        if stale.exists():
            shutil.rmtree(stale)

    try:
        subprocess.run([*pyinstaller, str(SPEC), "--noconfirm"], check=True, cwd=ROOT)
    except FileNotFoundError:
        print("PyInstaller not found. Install it with: pip install pyinstaller")
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"build failed with exit code {exc.returncode}")
        return exc.returncode

    produced = sorted(p.name for p in (ROOT / "dist").iterdir())
    print(f"\nbuilt: {', '.join(produced)}  (in dist/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
