"""Background screen-watching loop.

A dedicated thread periodically captures the screen, runs the recognition
pipeline, feeds the position tracker and publishes results through plain
callbacks. No Qt imports here: the GUI layer wraps the callbacks in Qt
signals, which keeps this component independently testable.
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

from ..config import VisionConfig
from ..core.tracker import PositionTracker, PositionUpdate
from ..logging_setup import get_logger
from ..vision.capture import ScreenCapture
from ..vision.pipeline import RecognitionPipeline, RecognitionResult

LOGGER = get_logger(__name__)

RecognitionCallback = Callable[[RecognitionResult], None]
PositionCallback = Callable[[PositionUpdate, RecognitionResult], None]
ErrorCallback = Callable[[str], None]


class ScreenWatcher:
    """Polls the screen and emits recognized chess positions."""

    def __init__(
        self,
        pipeline: RecognitionPipeline,
        tracker: PositionTracker,
        config: VisionConfig,
        capture: Optional[ScreenCapture] = None,
        on_recognition: Optional[RecognitionCallback] = None,
        on_position: Optional[PositionCallback] = None,
        on_error: Optional[ErrorCallback] = None,
    ) -> None:
        self.pipeline = pipeline
        self.tracker = tracker
        self.config = config
        self.capture = capture or ScreenCapture()
        self.on_recognition = on_recognition or (lambda result: None)
        self.on_position = on_position or (lambda update, result: None)
        self.on_error = on_error or (lambda message: LOGGER.warning(message))

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._paused = False
        self._owns_capture = capture is None
        self._frame_override: Optional[np.ndarray] = None

    # -------------------------------------------------------------- control
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="screen-watcher", daemon=True)
        self._thread.start()
        self._wake_event.set()  # first tick runs immediately, not after one interval

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._owns_capture:
            self.capture.__exit__(None, None, None)

    def pause(self) -> None:
        self._paused = True
        LOGGER.info("Screen watcher paused")

    def resume(self) -> None:
        self._paused = False
        self._wake_event.set()
        LOGGER.info("Screen watcher resumed")

    @property
    def paused(self) -> bool:
        return self._paused

    def trigger_now(self) -> None:
        """Wake the loop immediately (e.g. after a manual correction)."""
        self._wake_event.set()

    def inject_frame(self, frame: np.ndarray) -> None:
        """Test/debug hook: process a specific frame instead of a screenshot."""
        self._frame_override = frame
        self._wake_event.set()

    # ------------------------------------------------------------------ loop
    def _run(self) -> None:
        interval = max(0.2, self.config.poll_interval_ms / 1000.0)
        while not self._stop_event.is_set():
            if self._wake_event.wait(timeout=interval):
                self._wake_event.clear()
            if self._stop_event.is_set():
                break
            if self._paused:
                continue
            try:
                self.tick()
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.exception("Screen watcher tick failed")
                self.on_error(f"Board recognition error: {exc}")

    def tick(self) -> Optional[RecognitionResult]:
        """Perform one capture/recognize/track cycle."""
        if self._frame_override is not None:
            frame = self._frame_override
            self._frame_override = None
        else:
            frame = self.capture.grab()
        result = self.pipeline.process_stable(frame)
        self.on_recognition(result)
        if not result.success or result.fen is None:
            self.on_error(result.error or "Board recognition failed.")
            return result
        update = self.tracker.observe_placement_fen(result.fen)
        if update.accepted and update.changed:
            self.on_position(update, result)
        return result
