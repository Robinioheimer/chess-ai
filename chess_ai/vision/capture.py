"""Screen capture with a swappable backend.

``mss`` is fast, pure-Python and works on Windows/macOS/Linux. The capture
class is deliberately tiny so tests can inject synthetic frames instead of
grabbing a real desktop.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..logging_setup import get_logger

LOGGER = get_logger(__name__)


class ScreenCapture:
    """Thin wrapper around ``mss`` returning BGR frames (OpenCV convention)."""

    def __init__(self, monitor_index: int = 0) -> None:
        self.monitor_index = monitor_index
        self._sct = None

    def __enter__(self) -> "ScreenCapture":
        import mss
        factory = getattr(mss, "MSS", None) or getattr(mss, "mss")
        self._sct = factory()
        return self

    def __exit__(self, *exc) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def grab(self, region: Optional[dict] = None) -> np.ndarray:
        """Capture the full virtual screen (or a region) as a BGR image."""
        if self._sct is None:
            import mss
            factory = getattr(mss, "MSS", None) or getattr(mss, "mss")
            self._sct = factory()
        if region is None:
            monitors = self._sct.monitors
            region = monitors[min(self.monitor_index, len(monitors) - 1)]
        shot = self._sct.grab(region)
        frame = np.asarray(shot)  # BGRA
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    @staticmethod
    def frame_from_file(path: str) -> np.ndarray:
        """Load a screenshot from disk (used for offline troubleshooting)."""
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image
