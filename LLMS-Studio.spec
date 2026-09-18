# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for LLMS Studio.

Build with:  pyinstaller LLMS-Studio.spec --noconfirm

Notes on the choices here, each of which was a bug in the first version:

* The entry point is ``llms_studio.py``, not ``gui/app.py``. PyInstaller runs the
  entry script as ``__main__`` with its own directory first on ``sys.path``, so a
  script inside ``gui/`` cannot do ``from gui.x import y`` -- that is exactly the
  ``No module named 'gui'`` crash. A top-level launcher keeps both packages
  importable by their real names.
* ``llms`` is collected as a *package*, not as ``datas``. As data its imports were
  never analyzed, so torch was silently left out and nothing could actually train.
* ``upx=False``: UPX corrupts signed Mach-O binaries and torch's dylibs.
* No Tk anywhere. Apple's system Tcl/Tk is 8.5 and on macOS 26+ a plain
  ``tkinter.Tk().update()`` never returns, so the UI is served over HTTP to the
  browser instead. That also keeps the bundle free of a GUI toolkit.
"""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("llms") + collect_submodules("gui")

a = Analysis(
    ["llms_studio.py"],
    pathex=[],
    binaries=[],
    # Configs are read-only templates copied into the user's workspace on first
    # run; see gui/paths.py.
    datas=[("configs", "configs"), ("gui/index.html", "gui")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "ruff", "PyInstaller", "tkinter", "customtkinter"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LLMS-Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.icns",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="LLMS-Studio",
)

app = BUNDLE(
    coll,
    name="LLMS-Studio.app",
    icon="assets/icon.icns",
    bundle_identifier="com.github.blueyfamliy.llms-studio",
    version="0.1.0",
    info_plist={
        "CFBundleName": "LLMS Studio",
        "CFBundleDisplayName": "LLMS Studio",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        # Tk apps must not be treated as background-only or the window never shows.
        "LSBackgroundOnly": False,
        "LSMinimumSystemVersion": "11.0",
        # Tk does not participate in AppKit's automatic termination. Without
        # this, AppKit sees "no windows open" and kills the app after ~20s.
        "NSSupportsAutomaticTermination": False,
        "NSSupportsSuddenTermination": False,
        "NSRequiresAquaSystemAppearance": False,
    },
)
