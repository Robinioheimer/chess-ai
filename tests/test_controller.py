"""End-to-end controller test with the fake UCI engine and offscreen Qt."""
import sys
import time
from pathlib import Path

import chess
import pytest

from tests.gl_stub import ensure_pyqt5

try:
    ensure_pyqt5()
    PYQT_AVAILABLE = True
except Exception:  # pragma: no cover
    PYQT_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYQT_AVAILABLE,
                                reason="PyQt5 or a GL library is unavailable")

from PyQt5 import QtWidgets  # noqa: E402

from chess_ai.app import controller as controller_module  # noqa: E402
from chess_ai.config import AppConfig, EngineConfig  # noqa: E402
from chess_ai.engine.stockfish_setup import StockfishInfo  # noqa: E402
from chess_ai.core.fen_utils import board_to_grid  # noqa: E402
from chess_ai.vision.synthetic import render_board  # noqa: E402

FAKE_ENGINE = Path(__file__).parent / "fake_engine" / "fake_uci_engine.py"


@pytest.fixture
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait_for(predicate, timeout=15.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_controller_end_to_end(qapp, monkeypatch, tmp_path):
    config = AppConfig()
    config.engine = EngineConfig(depth=4, multipv=3, threads=1, hash_mb=16)
    config.vision.poll_interval_ms = 500

    fake_command = f"{sys.executable} {FAKE_ENGINE}"
    monkeypatch.setattr(
        controller_module, "ensure_stockfish",
        lambda path="": StockfishInfo(fake_command, "FakeStockfish", "test"),
    )
    monkeypatch.setattr(controller_module, "probe_stockfish", lambda path: "FakeStockfish")
    # Keep templates inside the test sandbox.
    monkeypatch.setattr(controller_module, "TEMPLATE_PATH", tmp_path / "templates.npz")

    class FakeCapture:
        def __init__(self, frame):
            self._frame = frame

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def grab(self, region=None):
            return self._frame

    frame = render_board(board_to_grid(chess.Board()), square_px=60)
    app = controller_module.AppController(config, capture=FakeCapture(frame))
    # This test exercises automatic detection on synthetic frames; the real app
    # deliberately keeps manual-only board selection enabled by default.
    app.pipeline.manual_only = False
    app.start()
    try:
        assert app.analyzer is not None
        assert app.engine is not None

        # Engine-state indicator reflects the connected engine.
        assert _wait_for(lambda: "FakeStockfish" in app.window.panel.engine_label.text())
        assert "#37C871" in app.window.panel.engine_dot.styleSheet()

        # The watcher polls the (fake) screen and should lock onto the board.
        assert _wait_for(lambda: app.tracker.has_position)
        assert app.tracker.board.board_fen() == chess.Board().board_fen()

        # The analysis manager should produce MultiPV results for the GUI.
        assert _wait_for(lambda: app.window._analysis is not None
                         and not app.window._analysis.searching)
        result = app.window._analysis
        assert result.error is None
        assert len(result.moves) == 3
        assert result.moves[0].san
        assert 0.0 <= result.moves[0].win_probability <= 1.0
        assert app.window.panel.table.rowCount() == 3

        # Pause/resume round trip.
        app.window.set_paused(True)
        assert app.watcher.paused and app.analyzer.paused
        app.window.set_paused(False)
        assert not app.watcher.paused and not app.analyzer.paused
    finally:
        app.shutdown()
        app.window.close()


def test_controller_reports_unavailable_engine(qapp, monkeypatch, tmp_path):
    from chess_ai.engine.uci import UciEngineError

    config = AppConfig()
    config.vision.poll_interval_ms = 500
    monkeypatch.setattr(
        controller_module, "ensure_stockfish",
        lambda path="": (_ for _ in ()).throw(UciEngineError("not found")),
    )
    monkeypatch.setattr(controller_module, "TEMPLATE_PATH", tmp_path / "templates.npz")

    class FakeCapture:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def grab(self, region=None):
            return render_board(board_to_grid(chess.Board()), square_px=60)

    app = controller_module.AppController(config, capture=FakeCapture())
    app.start()
    try:
        QtWidgets.QApplication.processEvents()
        assert app.analyzer is None
        assert "#E0574F" in app.window.panel.engine_dot.styleSheet()
        assert "unavailable" in app.window.panel.engine_label.text()
        # The failure is surfaced as a dismissible banner, not a crash.
        assert not app.window.panel.error_row.isHidden()
    finally:
        app.shutdown()
        app.window.close()
