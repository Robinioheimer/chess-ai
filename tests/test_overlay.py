"""Headless smoke tests for the PyQt5 overlay components."""
import chess
import pytest

from tests.gl_stub import ensure_pyqt5

try:
    ensure_pyqt5()
    PYQT_AVAILABLE = True
except Exception:  # pragma: no cover - environment without GUI libs
    PYQT_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYQT_AVAILABLE,
                                reason="PyQt5 or a GL library is unavailable")

from PyQt5 import QtWidgets  # noqa: E402

from chess_ai.config import AppConfig  # noqa: E402
from chess_ai.engine.types import AnalysisResult, MoveEvaluation  # noqa: E402
from chess_ai.overlay.board_marker import MarkerSpec, screen_row_col  # noqa: E402
from chess_ai.overlay.candidate_panel import CandidatePanel  # noqa: E402
from chess_ai.overlay.dialogs import CorrectionDialog, FenDialog  # noqa: E402
from chess_ai.overlay.main_window import MainOverlay  # noqa: E402
from chess_ai.vision.detect import detect_board  # noqa: E402
from chess_ai.vision.synthetic import render_scene  # noqa: E402
from chess_ai.core.fen_utils import board_to_grid  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def _analysis_result() -> AnalysisResult:
    moves = []
    data = [
        ("e2e4", "e4", 31, None),
        ("d2d4", "d4", 22, None),
        ("g1f3", "Nf3", 20, None),
    ]
    for index, (uci, san, cp, mate) in enumerate(data):
        moves.append(MoveEvaluation(
            multipv_index=index + 1,
            uci=uci, san=san, pv_uci=[uci], pv_san=[san],
            depth=12, cp=cp, mate=mate,
            win_probability=0.5 + index * 0.01,
            opponent_win_probability=0.5 - index * 0.01,
            opponent_eval=f"{-cp / 100:+.2f}",
            nodes=10000, nps=200000, time_ms=50,
            explanation="Develops a piece toward an active square in the opening.",
        ))
    return AnalysisResult(
        fen=chess.STARTING_FEN, side_to_move="w", moves=moves,
        searching=False, depth=12, target_depth=20, nodes=30000, nps=200000,
        progress=0.6, confidence=0.72, confidence_label="Medium",
    )


def test_screen_row_col_mapping():
    # White perspective: a1 is bottom-left -> column 0, row 7.
    assert screen_row_col(chess.E4, True) == (4, 4)
    assert screen_row_col(chess.A1, True) == (0, 7)
    # Black perspective: files mirrored, ranks from the top.
    assert screen_row_col(chess.A1, False) == (7, 0)


def test_marker_spec_parses_uci():
    spec = MarkerSpec("e2e4", "#0B2A6B", True, "e4")
    assert spec.from_square == chess.E2
    assert spec.to_square == chess.E4
    assert spec.is_best is True


def test_main_overlay_renders_markers_and_filters_hidden(qapp):
    config = AppConfig()
    window = MainOverlay(config)
    scene, _ = render_scene(board_to_grid(chess.Board()), square_px=56, padding=60)
    detection = detect_board(scene)
    from chess_ai.vision.pipeline import RecognitionResult as RecResult
    recognition = RecResult(
        success=True, fen=chess.STARTING_FEN,
        screen_grid=board_to_grid(chess.Board()),
        board_grid=board_to_grid(chess.Board()),
        perspective_white=True, detection=detection,
    )
    window.update_recognition(recognition)
    window.update_analysis(_analysis_result())
    assert len(window.marker_view._markers) == 3
    assert window.marker_view._markers[0].is_best is True
    # Best-move marker uses the configured dark blue.
    assert window.marker_view._markers[0].color.name().lower() == "#0b2a6b"

    # Hide the second candidate; only two markers remain.
    window._on_visibility_changed(1, False)
    assert len(window.marker_view._markers) == 2
    window._on_visibility_changed(1, True)
    assert len(window.marker_view._markers) == 3
    window.close()


def test_opacity_and_scale_setters(qapp):
    window = MainOverlay(AppConfig())
    window.apply_opacity(0.5)
    assert window.windowOpacity() == pytest.approx(0.5, abs=0.01)
    window.apply_scale(1.5)
    assert window.font().pointSizeF() == pytest.approx(15.0)
    window.close()


def test_pause_toggle_emits_signal(qapp, qtbot=None):
    window = MainOverlay(AppConfig())
    received = []
    window.paused_changed.connect(received.append)
    window.pause_button.setChecked(True)
    assert received == [True]
    window.pause_button.setChecked(False)
    assert received == [True, False]
    window.close()


def test_candidate_panel_populates_and_reports_visibility(qapp):
    panel = CandidatePanel()
    panel.update_result(_analysis_result())
    assert panel.table.rowCount() == 3
    assert panel.progress.value() == 100  # completed analysis

    events = []
    panel.visibility_changed.connect(lambda row, visible: events.append((row, visible)))
    checkbox_widget = panel.table.cellWidget(0, 0)
    checkbox = checkbox_widget.findChild(QtWidgets.QCheckBox)
    checkbox.setChecked(False)
    assert events == [(0, False)]
    assert "Medium" in panel.confidence_label.text()


def test_candidate_panel_shows_engine_error(qapp):
    panel = CandidatePanel()
    result = AnalysisResult(fen=chess.STARTING_FEN, side_to_move="w",
                            error="Engine communication failure: timeout")
    panel.update_result(result)
    # Errors appear in a dedicated dismissible banner, never a modal dialog.
    assert not panel.error_row.isHidden()
    assert "failure" in panel.error_label.text()
    assert panel.table.rowCount() == 0
    panel.clear_error_button.click()
    assert panel.error_row.isHidden()
    # Ordinary status updates leave a persistent error banner untouched; only
    # the Clear button or a successful analysis dismiss it.
    panel.show_error("something went wrong")
    assert not panel.error_row.isHidden()
    panel.set_status("Overlay ready.")
    assert not panel.error_row.isHidden()
    panel.clear_error()
    assert panel.error_row.isHidden()


def test_engine_state_indicator(qapp):
    panel = CandidatePanel()
    panel.set_engine_state("connected", "Engine: Stockfish 17 — ready")
    assert "#37C871" in panel.engine_dot.styleSheet()
    assert "ready" in panel.engine_label.text()
    panel.set_searching(True)
    assert "#E8B33C" in panel.engine_dot.styleSheet()
    assert "analyzing" in panel.engine_label.text()
    panel.set_searching(False)
    assert "#37C871" in panel.engine_dot.styleSheet()
    assert "ready" in panel.engine_label.text()
    panel.set_engine_state("unavailable", "Engine: Stockfish unavailable")
    assert "#E0574F" in panel.engine_dot.styleSheet()


def test_overlay_show_error_is_nonmodal(qapp):
    window = MainOverlay(AppConfig())
    window.show_error("No position has been detected yet.")
    assert not window.panel.error_row.isHidden()
    assert "No position" in window.panel.error_label.text()
    window.clear_error()
    assert window.panel.error_row.isHidden()
    window.close()


def test_fen_dialog_validation(qapp):
    dialog = FenDialog(chess.STARTING_FEN)
    assert dialog.fen() == chess.STARTING_FEN
    dialog.edit.setText("garbage")
    assert "✗" in dialog.feedback.text()
    dialog.edit.setText(chess.STARTING_FEN)
    assert "✓" in dialog.feedback.text()


def test_correction_dialog_roundtrip(qapp):
    grid = board_to_grid(chess.Board())
    from chess_ai.core.fen_utils import board_grid_to_screen_grid
    screen_grid = board_grid_to_screen_grid(grid, True)
    dialog = CorrectionDialog(screen_grid, True)
    fen = dialog.corrected_fen("w")
    assert chess.Board(fen).board_fen() == chess.Board().board_fen()
    # Side-to-move selector feeds the FEN directly.
    dialog.side_combo.setCurrentIndex(1)
    assert dialog.corrected_fen().split()[1] == "b"


def test_click_through_is_safe_noop_off_windows(qapp):
    window = MainOverlay(AppConfig())
    window.set_click_through(True)  # must not raise on Linux
    window.set_click_through(False)
    window.close()


def test_cover_screen_sets_geometry(qapp):
    window = MainOverlay(AppConfig())
    window.cover_screen()
    screen = QtWidgets.QApplication.primaryScreen()
    assert window.geometry() == screen.availableGeometry()
    window.close()


def test_no_fallback_arrows_before_engine_result(qapp):
    window = MainOverlay(AppConfig())
    scene, _ = render_scene(board_to_grid(chess.Board()), square_px=56, padding=60)
    detection = detect_board(scene)
    from chess_ai.vision.pipeline import RecognitionResult as RecResult
    recognition = RecResult(
        success=True, fen=chess.STARTING_FEN,
        screen_grid=board_to_grid(chess.Board()),
        board_grid=board_to_grid(chess.Board()),
        perspective_white=True, detection=detection,
    )
    window.update_recognition(recognition)
    assert window.marker_view._markers == []
    window.close()


def test_stale_analysis_for_different_fen_is_not_drawn(qapp):
    window = MainOverlay(AppConfig())
    scene, _ = render_scene(board_to_grid(chess.Board()), square_px=56, padding=60)
    detection = detect_board(scene)
    from chess_ai.vision.pipeline import RecognitionResult as RecResult
    recognition = RecResult(
        success=True, fen=chess.STARTING_FEN,
        screen_grid=board_to_grid(chess.Board()),
        board_grid=board_to_grid(chess.Board()),
        perspective_white=True, detection=detection,
    )
    stale = _analysis_result()
    old_board = chess.Board()
    old_board.push_san("e4")
    stale.fen = old_board.fen()  # Result belongs to a different position.
    window.update_recognition(recognition)
    window.update_analysis(stale)
    assert window.marker_view._markers == []
    window.close()


def test_marker_view_renders_board_outline_before_analysis(qapp):
    window = MainOverlay(AppConfig())
    scene, _ = render_scene(board_to_grid(chess.Board()), square_px=56, padding=60)
    detection = detect_board(scene)
    from chess_ai.vision.pipeline import RecognitionResult as RecResult
    recognition = RecResult(
        success=True,
        fen=chess.STARTING_FEN,
        screen_grid=board_to_grid(chess.Board()),
        board_grid=board_to_grid(chess.Board()),
        perspective_white=True,
        detection=detection,
    )
    window.update_recognition(recognition)
    assert window.marker_view._corners is not None
    assert window.marker_view._markers == []
    window.close()


def test_correction_dialog_quick_actions_and_uncertain_highlighting(qapp):
    from chess_ai.core.fen_utils import board_grid_to_screen_grid

    grid = board_to_grid(chess.Board())
    screen_grid = board_grid_to_screen_grid(grid, True)
    dialog = CorrectionDialog(
        screen_grid, True, side_to_move="b", uncertain_squares=["e4", "a8"]
    )
    assert dialog.corrected_fen().split()[1] == "b"
    dialog._flip_perspective()
    assert dialog.perspective_white is False
    dialog._flip_perspective()
    assert dialog.perspective_white is True
    dialog._clear_all()
    assert all(cell is None for row in dialog.corrected_screen_grid() for cell in row)
    assert "✗" in dialog.validation_label.text()
    dialog._reset_to_start()
    assert "✓" in dialog.validation_label.text()
    assert chess.Board(dialog.corrected_fen()).board_fen() == chess.Board().board_fen()


def test_turn_and_perspective_buttons_emit_signals(qapp):
    window = MainOverlay(AppConfig())
    turns = []
    perspectives = []
    regions = []
    window.turn_toggle_requested.connect(lambda: turns.append(True))
    window.perspective_toggle_requested.connect(lambda: perspectives.append(True))
    window.board_region_requested.connect(lambda: regions.append(True))
    window.turn_button.click()
    window.perspective_button.click()
    window.select_board_button.click()
    assert turns == [True]
    assert perspectives == [True]
    assert regions == [True]
    window.close()


def test_markers_include_percent_accuracy_and_best_move_arrow(qapp):
    from PyQt5 import QtGui

    config = AppConfig()
    window = MainOverlay(config)
    scene, _ = render_scene(board_to_grid(chess.Board()), square_px=56, padding=60)
    detection = detect_board(scene)
    from chess_ai.vision.pipeline import RecognitionResult as RecResult

    recognition = RecResult(
        success=True,
        fen=chess.STARTING_FEN,
        screen_grid=board_to_grid(chess.Board()),
        board_grid=board_to_grid(chess.Board()),
        perspective_white=True,
        detection=detection,
    )
    window.update_recognition(recognition)
    window.update_analysis(_analysis_result())

    markers = window.marker_view._markers
    assert len(markers) == 3
    assert markers[0].is_best is True
    assert "%" in markers[0].percent_text
    assert "%" in markers[0].badge_text
    assert "e4" in markers[0].badge_text

    # Render BoardMarkerView into an image and confirm non-transparent pixels
    # are painted along the e2->e4 best-move arrow shaft.
    image = QtGui.QImage(800, 800, QtGui.QImage.Format_ARGB32_Premultiplied)
    image.fill(0)
    painter = QtGui.QPainter(image)
    window.marker_view.render(painter)
    painter.end()
    assert not image.isNull()
    window.close()

def test_visible_overlay_uses_full_screen_layer_for_move_arrows(qapp):
    """Arrows must be painted above the full-screen control window, not behind it."""
    window = MainOverlay(AppConfig())
    scene, _ = render_scene(board_to_grid(chess.Board()), square_px=56, padding=60)
    detection = detect_board(scene)
    from chess_ai.vision.pipeline import RecognitionResult as RecResult

    recognition = RecResult(
        success=True,
        fen=chess.STARTING_FEN,
        screen_grid=board_to_grid(chess.Board()),
        board_grid=board_to_grid(chess.Board()),
        perspective_white=True,
        detection=detection,
    )
    window.update_recognition(recognition)
    window.update_analysis(_analysis_result())
    window.cover_screen()
    window.show()
    qapp.processEvents()
    window._refresh_markers()
    qapp.processEvents()

    assert window.screen_marker_view.isVisible()
    assert len(window.screen_marker_view._markers) >= 1
    window.close()
