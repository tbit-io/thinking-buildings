# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Thinking Buildings.

Two profiles controlled by the PROFILE environment variable:
  PROFILE=onnx    → Lightweight ONNX-only build (~100MB), excludes PyTorch
  PROFILE=full    → Full build with ultralytics/torch (~2GB)

Usage:
  PROFILE=onnx pyinstaller installer/thinking_buildings.spec
  PROFILE=full pyinstaller installer/thinking_buildings.spec
"""

import os
import sys
from pathlib import Path

PROFILE = os.environ.get("PROFILE", "onnx").lower()
ROOT = Path(SPEC).parent.parent

# --- Excludes per profile ------------------------------------------------

TORCH_EXCLUDES = [
    "torch",
    "torchvision",
    "torchaudio",
    "torch._C",
    "torch.cuda",
    "torchgen",
    "sympy",
    "networkx",
    "jinja2",
    "mpmath",
    "filelock",
]

COMMON_EXCLUDES = [
    "tkinter",
    "unittest",
    "pydoc",
    "doctest",
    "xmlrpc",
    "IPython",
    "jupyter",
    "notebook",
    "matplotlib",
    "scipy",
    "pandas",
    "PIL",
]

if PROFILE == "onnx":
    excludes = TORCH_EXCLUDES + COMMON_EXCLUDES + ["ultralytics"]
elif PROFILE == "full":
    excludes = COMMON_EXCLUDES
else:
    raise ValueError(f"Unknown PROFILE: {PROFILE!r}. Use 'onnx' or 'full'.")

# --- Collect data ---------------------------------------------------------

datas = []
if PROFILE == "full":
    # ultralytics needs its cfg/ directory
    try:
        import ultralytics
        ul_path = Path(ultralytics.__file__).parent
        datas.append((str(ul_path / "cfg"), "ultralytics/cfg"))
    except ImportError:
        pass

# --- Hidden imports -------------------------------------------------------

hiddenimports = [
    "thinking_buildings",
    "thinking_buildings.cli",
    "thinking_buildings.config",
    "thinking_buildings.events",
    "thinking_buildings.capture",
    "thinking_buildings.detector",
    "thinking_buildings.alerter",
    "thinking_buildings.display",
    "thinking_buildings.camera_probe",
    "thinking_buildings.model_manager",
    "thinking_buildings.logger_setup",
    "thinking_buildings.backends",
    "thinking_buildings.backends.base",
    "thinking_buildings.backends.ultralytics_backend",
    "thinking_buildings.face_recognizer",
    "thinking_buildings.face_db",
    "yaml",
    "cv2",
    "numpy",
    "onnxruntime",
]

if PROFILE == "full":
    hiddenimports.extend([
        "ultralytics",
        "ultralytics.nn",
        "ultralytics.nn.tasks",
    ])

# --- Analysis -------------------------------------------------------------

a = Analysis(
    [str(ROOT / "thinking_buildings" / "cli.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="thinking-buildings",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon=str(ROOT / "installer" / "icon.ico") if (ROOT / "installer" / "icon.ico").exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=f"thinking-buildings-{PROFILE}",
)
