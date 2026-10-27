"""Headless test helper for importing PyQt5 without a system libGL.

CI/sandbox environments often lack ``libGL.so.1`` even though the test only
needs the offscreen Qt platform. When the normal import fails because of that
single missing library, this helper compiles a tiny stub shared object (gcc)
and preloads it globally so the Qt libraries resolve their GL symbols.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import tempfile

STUB_SOURCE = r"""
void glMatrixMode(void) {}
void glLoadIdentity(void) {}
void glLoadMatrixf(void) {}
void glOrtho(void) {}
"""


def ensure_pyqt5():
    import importlib
    try:
        # Importing a widget submodule actually loads libQt5Gui/libGL, so this
        # probe detects a missing GL library rather than just the package.
        importlib.import_module("PyQt5.QtWidgets")
        return True
    except ImportError as exc:
        if "libGL" not in str(exc) or shutil.which("gcc") is None:
            raise
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    temp_dir = tempfile.mkdtemp(prefix="chessai-glstub-")
    source = os.path.join(temp_dir, "stubgl.c")
    library = os.path.join(temp_dir, "libGL.so.1")
    with open(source, "w", encoding="utf-8") as handle:
        handle.write(STUB_SOURCE)
    subprocess.run(
        ["gcc", "-shared", "-fPIC", "-Wl,-soname,libGL.so.1",
         "-o", library, source],
        check=True,
    )
    ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
    importlib.import_module("PyQt5.QtWidgets")
    return True
