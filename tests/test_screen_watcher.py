"""Tests for the screen-watching loop and hotkey service."""
import time

import chess

from chess_ai.app.hotkeys import HotkeyService
from chess_ai.app.screen_watcher import ScreenWatcher
from chess_ai.config import VisionConfig
from chess_ai.core.fen_utils import board_to_grid
from chess_ai.core.tracker import PositionTracker
from chess_ai.vision.pipeline import RecognitionPipeline
from chess_ai.vision.synthetic import render_board


def _watcher(stable_readings=1):
    pipeline = RecognitionPipeline(stable_readings=stable_readings, manual_only=False)
    tracker = PositionTracker()
    events = {"recognition": [], "position": [], "error": []}
    watcher = ScreenWatcher(
        pipeline,
        tracker,
        VisionConfig(poll_interval_ms=500),
        on_recognition=events["recognition"].append,
        on_position=lambda update, result: events["position"].append(update),
        on_error=events["error"].append,
    )
    return watcher, tracker, events


def test_single_tick_recognizes_starting_board():
    watcher, tracker, events = _watcher()
    frame = render_board(board_to_grid(chess.Board()), square_px=60)
    watcher.inject_frame(frame)
    watcher.tick()
    assert tracker.has_position
    assert tracker.board.board_fen() == chess.Board().board_fen()
    assert events["recognition"][0].success
    assert events["position"][0].changed


def test_watcher_thread_publishes_position_and_pauses():
    watcher, tracker, events = _watcher()
    frame = render_board(board_to_grid(chess.Board()), square_px=60)
    watcher.inject_frame(frame)
    watcher.start()
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not tracker.has_position:
            time.sleep(0.05)
        assert tracker.has_position

        watcher.pause()
        assert watcher.paused
        # While paused, injected frames must not alter the tracker.
        watcher.inject_frame(render_board(board_to_grid(chess.Board()), square_px=60))
        watcher.trigger_now()
        time.sleep(0.2)
        assert tracker.board.board_fen() == chess.Board().board_fen()

        watcher.resume()
        assert not watcher.paused
    finally:
        watcher.stop()
    assert watcher._thread is not None and not watcher._thread.is_alive()


def test_missing_board_reports_error_without_crashing():
    import numpy as np
    watcher, tracker, events = _watcher()
    watcher.inject_frame(np.full((400, 400, 3), 12, dtype=np.uint8))
    result = watcher.tick()
    assert not result.success
    assert events["error"]
    assert not tracker.has_position


def test_hotkey_service_without_pynput_gracefully_disables(monkeypatch):
    # Simulate pynput being unavailable.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pynput"):
            raise ImportError("simulated missing pynput")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    service = HotkeyService("<ctrl>+<alt>+s", lambda: None)
    started = service.start()
    assert started is False
    assert service.available is False
    # Reconfiguring the hotkey must also fail gracefully, not raise.
    assert service.restart_with("<ctrl>+<alt>+h") is False
    service.stop()  # must not raise
