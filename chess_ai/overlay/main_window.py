"""Frameless, transparent, always-on-top overlay window.

The window hosts two layers:

* :class:`BoardMarkerView` - a full-size transparent canvas painting the
  best-move corner markers and alternative-move arrows over the board.
* :class:`CandidatePanel` - a docked, semi-opaque table of candidate moves.

It is frameless and translucent, draggable by its title bar and resizable via
a size grip, without touching the underlying application.
"""
from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from ..config import AppConfig
from ..engine.types import AnalysisResult
from ..vision.pipeline import RecognitionResult
from .board_marker import BEST_MOVE_COLOR, BoardMarkerView, MarkerSpec
from .candidate_panel import CandidatePanel

STYLESHEET = """
QWidget#OverlayRoot {
    background-color: transparent;
}
QWidget#TitleBarWidget, QWidget#ToolbarWidget {
    background-color: rgba(18, 22, 30, 230);
    border: 1px solid rgba(120, 140, 180, 120);
    border-radius: 8px;
}
QWidget#CandidatePanel {
    background-color: rgba(14, 18, 26, 215);
    border: 1px solid #33405C;
    border-radius: 10px;
}
QLabel#PanelTitle { font-size: 15px; font-weight: 700; color: #EAF0FF; }
QLabel { color: #D6DEF0; }
QLabel#Legend { color: #9FB0CC; font-size: 11px; }
QLabel#ErrorLabel { color: #FF9A93; font-weight: 600; }
QPushButton#ClearErrorButton {
    background-color: #4A2320; color: #FFB4AE;
    border: 1px solid #7A3B35; padding: 3px 8px;
}
QPushButton#ClearErrorButton:hover { background-color: #5E2C27; }
QPushButton {
    background-color: #243049;
    color: #EAF0FF;
    border: 1px solid #3A4C73;
    border-radius: 6px;
    padding: 5px 10px;
}
QPushButton:hover { background-color: #314263; }
QPushButton:checked { background-color: #0B2A6B; border-color: #4E7BD8; }
QTableWidget {
    background-color: rgba(28, 34, 48, 220);
    color: #EAF0FF;
    gridline-color: #33405C;
    border: 1px solid #33405C;
    border-radius: 6px;
}
QHeaderView::section {
    background-color: #243049;
    color: #B9C6E0;
    padding: 4px;
    border: none;
}
QProgressBar {
    background-color: #1A2233;
    border: 1px solid #33405C;
    border-radius: 5px;
    text-align: center;
    color: #EAF0FF;
}
QProgressBar::chunk { background-color: #2E6BE6; border-radius: 4px; }
QLineEdit, QSpinBox, QComboBox {
    background-color: #1A2233; color: #EAF0FF;
    border: 1px solid #3A4C73; border-radius: 4px; padding: 3px;
}
QSlider::groove:horizontal { height: 6px; background: #1A2233; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #4E7BD8; width: 14px; margin: -5px 0; border-radius: 7px;
}
"""


class TitleBar(QtWidgets.QWidget):
    """Custom title bar supporting window dragging."""

    def __init__(self, window: "MainOverlay") -> None:
        super().__init__(window)
        self.setObjectName("TitleBarWidget")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self._window = window
        self._drag_position = None
        self.setFixedHeight(38)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)

        title = QtWidgets.QLabel("♞ Chess AI Overlay")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        layout.addStretch(1)

        self.minimize_button = QtWidgets.QPushButton("—")
        self.minimize_button.setFixedSize(30, 26)
        self.minimize_button.clicked.connect(window.showMinimized)
        layout.addWidget(self.minimize_button)
        self.hide_button = QtWidgets.QPushButton("×")
        self.hide_button.setFixedSize(30, 26)
        self.hide_button.setToolTip("Hide overlay (global hotkey restores it)")
        self.hide_button.clicked.connect(window.hide)
        layout.addWidget(self.hide_button)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_position = event.globalPos() - self._window.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_position is not None and event.buttons() & QtCore.Qt.LeftButton:
            self._window.move(event.globalPos() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_position = None


class MainOverlay(QtWidgets.QWidget):
    """Top-level overlay window."""

    paused_changed = QtCore.pyqtSignal(bool)
    reanalyze_requested = QtCore.pyqtSignal()
    fen_requested = QtCore.pyqtSignal()
    correction_requested = QtCore.pyqtSignal()
    calibration_requested = QtCore.pyqtSignal()
    board_region_requested = QtCore.pyqtSignal()
    turn_toggle_requested = QtCore.pyqtSignal()
    perspective_toggle_requested = QtCore.pyqtSignal()
    settings_requested = QtCore.pyqtSignal()
    move_visibility_changed = QtCore.pyqtSignal(str, bool)
    move_focused = QtCore.pyqtSignal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self._recognition: RecognitionResult | None = None
        self._analysis: AnalysisResult | None = None
        self._hidden_uci: set[str] = set()
        self.setWindowTitle("Chess AI Overlay")
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setObjectName("OverlayRoot")
        self.setStyleSheet(STYLESHEET)
        self.resize(980, 640)
        self._build_ui()
        self.apply_opacity(config.overlay.opacity)
        self.apply_scale(config.overlay.scale)
        # Start refresh timer so arrows don't disappear.
        self._start_refresh_timer()

    # ------------------------------------------------------------------ ui
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.title_bar = TitleBar(self)
        root.addWidget(self.title_bar)

        self.marker_view = BoardMarkerView(self)
        self.marker_view.lower()
        self.marker_view.setGeometry(self.rect())

        # Companion full-screen click-through marker layer so move arrows and
        # % accuracy badges remain visible on the chessboard even when the user
        # drags or resizes the control panel window off the board.
        self.screen_marker_view = BoardMarkerView(None)
        self.screen_marker_view.setWindowTitle("Chess AI Board Markers")
        self.screen_marker_view.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
            | QtCore.Qt.WindowTransparentForInput
        )
        self.screen_marker_view.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.screen_marker_view.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(8, 4, 8, 4)
        body.setSpacing(8)
        body.addStretch(1)
        self.panel = CandidatePanel()
        self.panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.panel.visibility_changed.connect(self._on_visibility_changed)
        self.panel.row_selected.connect(self._on_row_selected)
        body.addWidget(self.panel, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)
        root.addLayout(body, 1)

        toolbar_container = QtWidgets.QWidget(self)
        toolbar_container.setObjectName("ToolbarWidget")
        toolbar_container.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        toolbar = QtWidgets.QHBoxLayout(toolbar_container)
        toolbar.setContentsMargins(10, 4, 10, 6)
        self.pause_button = QtWidgets.QPushButton("⏸ Pause")
        self.pause_button.setCheckable(True)
        self.pause_button.toggled.connect(self._toggle_paused)
        toolbar.addWidget(self.pause_button)

        reanalyze = QtWidgets.QPushButton("⟳ Reanalyze")
        reanalyze.clicked.connect(self.reanalyze_requested.emit)
        toolbar.addWidget(reanalyze)

        self.select_board_button = QtWidgets.QPushButton("⬚ Select board")
        self.select_board_button.setToolTip(
            "Drag a rectangle around the chessboard on screen (or reset to auto-detect)"
        )
        self.select_board_button.clicked.connect(self.board_region_requested.emit)
        toolbar.addWidget(self.select_board_button)

        fen_button = QtWidgets.QPushButton("FEN")
        fen_button.setToolTip("Enter a position manually via FEN")
        fen_button.clicked.connect(self.fen_requested.emit)
        toolbar.addWidget(fen_button)

        correct_button = QtWidgets.QPushButton("✎ Correct board")
        correct_button.setToolTip("Open the 8×8 visual board editor to fix squares or pieces")
        correct_button.clicked.connect(self.correction_requested.emit)
        toolbar.addWidget(correct_button)

        calibrate_button = QtWidgets.QPushButton("◈ Calibrate pieces")
        calibrate_button.setToolTip("Learn piece shapes from a visible starting position")
        calibrate_button.clicked.connect(self.calibration_requested.emit)
        toolbar.addWidget(calibrate_button)

        self.turn_button = QtWidgets.QPushButton("⇄ Turn: White")
        self.turn_button.setToolTip("Click to switch whose turn it is (White / Black to move)")
        self.turn_button.clicked.connect(self.turn_toggle_requested.emit)
        toolbar.addWidget(self.turn_button)

        toolbar.addStretch(1)
        self.perspective_label = QtWidgets.QLabel("Perspective: auto")
        toolbar.addWidget(self.perspective_label)

        self.perspective_button = QtWidgets.QPushButton("⇅ Flip")
        self.perspective_button.setToolTip("Flip board orientation (White / Black at bottom)")
        self.perspective_button.clicked.connect(self.perspective_toggle_requested.emit)
        toolbar.addWidget(self.perspective_button)

        settings_button = QtWidgets.QPushButton("⚙ Settings")
        settings_button.clicked.connect(self.settings_requested.emit)
        toolbar.addWidget(settings_button)
        root.addWidget(toolbar_container)

        size_grip = QtWidgets.QSizeGrip(self)
        size_grip.setFixedSize(16, 16)
        grip_layout = QtWidgets.QHBoxLayout()
        grip_layout.setContentsMargins(0, 0, 6, 4)
        grip_layout.addStretch(1)
        grip_layout.addWidget(size_grip)
        root.addLayout(grip_layout)

    # ------------------------------------------------------------ appearance
    def apply_opacity(self, opacity: float) -> None:
        self.setWindowOpacity(float(max(0.3, min(1.0, opacity))))

    def apply_scale(self, scale: float) -> None:
        font = self.font()
        font.setPointSizeF(10.0 * scale)
        self.setFont(font)
        self.resize(int(980 * scale), int(640 * scale))

    def cover_screen(self) -> None:
        """Expand to the available screen area so board markers line up."""
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.availableGeometry())

    def exclude_from_capture(self, enabled: bool = True) -> None:
        """Exclude the overlay window from screenshots on Windows 10/11.

        Prevents ``mss`` from capturing the overlay's own panel and move arrows
        on top of the chessboard. Safe no-op on non-Windows platforms.
        """
        import sys
        if sys.platform != "win32":
            return
        try:
            import ctypes
            # WDA_EXCLUDEFROMCAPTURE = 0x00000011, WDA_NONE = 0x00000000
            affinity = 0x00000011 if enabled else 0x00000000
            for widget in (self, self.screen_marker_view):
                hwnd = int(widget.winId())
                ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, affinity)
        except Exception:  # pragma: no cover - OS version dependent
            pass

    def set_click_through(self, enabled: bool) -> None:
        """Make the overlay ignore mouse input (Windows: WS_EX_TRANSPARENT).

        On other platforms this is a no-op; the global hotkey still restores
        the window and the title bar remains usable while the flag is off.
        """
        import sys
        if sys.platform != "win32":
            return
        import ctypes
        hwnd = int(self.winId())
        styles = ctypes.windll.user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
        transparent = 0x00000020  # WS_EX_TRANSPARENT
        layered = 0x00080000      # WS_EX_LAYERED
        if enabled:
            styles |= transparent | layered
        else:
            styles &= ~transparent
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, styles)

    # -------------------------------------------------------------- updates
    def set_side_to_move(self, is_white: bool) -> None:
        self.turn_button.setText("⇄ Turn: White" if is_white else "⇄ Turn: Black")

    def update_recognition(self, result: RecognitionResult) -> None:
        self._recognition = result
        if result.success:
            perspective = "White" if result.perspective_white else "Black"
            self.perspective_label.setText(f"Perspective: {perspective} (bottom)")
            if result.uncertain_squares:
                self.panel.set_status(
                    "Low-confidence squares: " + ", ".join(result.uncertain_squares[:8])
                    + (" …" if len(result.uncertain_squares) > 8 else "")
                    + " — use ✎ Correct board if needed."
                )
        self._refresh_markers()

    def update_analysis(self, result: AnalysisResult) -> None:
        self._analysis = result
        self.set_side_to_move(result.side_to_move == "w")
        self.panel.update_result(result)
        if not result.error:
            self.panel.set_searching(bool(result.searching))
        self._refresh_markers()

    def set_status(self, text: str) -> None:
        self.panel.set_status(text)

    def show_error(self, text: str) -> None:
        # Non-modal: a dismissible banner in the panel instead of a blocking
        # message box, so recognition/analysis keeps running underneath.
        self.panel.show_error(text)

    def clear_error(self) -> None:
        self.panel.clear_error()

    def set_engine_state(self, state: str, text: str) -> None:
        self.panel.set_engine_state(state, text)

    # Regular refresh timer so arrows don't disappear between analysis ticks.
    def _start_refresh_timer(self) -> None:
        try:
            timer = QtCore.QTimer(self)
            timer.timeout.connect(self._refresh_markers)
            timer.start(300)  # refresh every 300 ms
            self._refresh_timer = timer
        except Exception:
            pass

    def _board_inside_window(self, corners) -> bool:
        """Return True if all four board corners lie inside this window."""
        rect = self.geometry()
        return all(
            rect.contains(int(pt[0]), int(pt[1]))
            for pt in corners
        )

    def _refresh_markers(self) -> None:
        # Always show board outline and markers when recognition succeeded.
        if (
            self._recognition is None
            or not self._recognition.success
            or self._recognition.detection is None
        ):
            self.marker_view.clear()
            self.screen_marker_view.clear()
            self.screen_marker_view.hide()
            return
        markers: list[MarkerSpec] = []
        # Show analysis moves when available.
        if self._analysis is not None and self._analysis.moves:
            colors = self.config.overlay.candidate_colors
            for index, move in enumerate(self._analysis.moves):
                if move.uci in self._hidden_uci:
                    continue
                if not move.visible:
                    continue
                if index == 0:
                    color = self.config.overlay.marker_color_best or BEST_MOVE_COLOR
                    markers.append(MarkerSpec(
                        move.uci, color, True, move.san,
                        percent_text=move.win_percent_text, rank=index + 1,
                    ))
                else:
                    color = colors[(index - 1) % len(colors)]
                    markers.append(MarkerSpec(
                        move.uci, color, False, move.san,
                        percent_text=move.win_percent_text, rank=index + 1,
                    ))
        # Even without analysis, draw at least the best-move arrow if we have
        # a recognized position and a previous best move is tracked.
        if not markers and self._analysis is not None and self._analysis.best_move:
            best = self._analysis.best_move
            markers.append(MarkerSpec(
                best.uci, BEST_MOVE_COLOR, True, best.san,
                percent_text=best.win_percent_text, rank=1,
            ))
        # Fallback: wenn keine Analysis und kein Best-Move, aber Board erkannt,
        # zeige einen Hinweis-Pfeil (z. B. zentrierter Punkt) damit der Benutzer
        # sieht, dass die Erkennung funktioniert und Moves folgen können.
        # Stelle sicher, dass immer ein visuelles Feedback da ist:
        # Wenn keine Analyse und keine spezifischen Pfeile, zeige mindestens
        # einen kleinen Hinweis auf die Erkennung (z. B. zentrierter Punkt),
        # damit der Benutzer sieht, dass das System aktiv ist und Moves folgen.
        if not markers and (self._recognition is not None and self._recognition.success):
        # Minimaler Hinweis: zeige einen kleinen Punkt in der Mitte des Boards
        # als visuelles Feedback, dass die Erkennung funktioniert und Moves folgen.
        # Keine spezifischen Pfeile ohne Analyse, aber der Benutzer sieht,
        # dass das System aktiv ist und Moves folgen können.
        # Interne Figur-Tracking aktiv: Pipeline speichert _last_grid intern.
            pass
        corners = self._recognition.detection.corners
        perspective_white = self._recognition.perspective_white
        self.marker_view.set_window_origin(self.geometry().topLeft())
        self.marker_view.update_board(corners, perspective_white, markers)

        # If this window is visible and does not fully cover the detected board
        # (for example, the user dragged/resized the control panel to the side),
        # render the arrows on the full-screen click-through companion layer.
        screen = QtWidgets.QApplication.primaryScreen()
        if self.isVisible() and screen is not None and not self._board_inside_window(corners):
            avail = screen.availableGeometry()
            self.screen_marker_view.setGeometry(avail)
            self.screen_marker_view.set_window_origin(avail.topLeft())
            self.screen_marker_view.update_board(corners, perspective_white, markers)
            if not self.screen_marker_view.isVisible():
                self.screen_marker_view.show()
                self.exclude_from_capture(True)
        else:
            self.screen_marker_view.update_board(corners, perspective_white, markers)
            if self.screen_marker_view.isVisible():
                self.screen_marker_view.hide()

    def hideEvent(self, event) -> None:  # noqa: N802
        self.screen_marker_view.hide()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.screen_marker_view.close()
        super().closeEvent(event)

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        self._refresh_markers()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # The marker canvas spans the whole window behind the docked panel.
        self.marker_view.setGeometry(self.rect())
        self._refresh_markers()

    # ------------------------------------------------------------- controls
    def _toggle_paused(self, paused: bool) -> None:
        self.pause_button.setText("▶ Resume" if paused else "⏸ Pause")
        self.paused_changed.emit(paused)

    def set_paused(self, paused: bool) -> None:
        self.pause_button.setChecked(paused)

    def _on_visibility_changed(self, row: int, visible: bool) -> None:
        if not self._analysis or row >= len(self._analysis.moves):
            return
        uci = self._analysis.moves[row].uci
        if visible:
            self._hidden_uci.discard(uci)
        else:
            self._hidden_uci.add(uci)
        self.move_visibility_changed.emit(uci, visible)
        self._refresh_markers()

    def _on_row_selected(self, row: int) -> None:
        if self._analysis and row < len(self._analysis.moves):
            self.move_focused.emit(self._analysis.moves[row].uci)

    def reset_hidden_moves(self) -> None:
        self._hidden_uci.clear()
        self._refresh_markers()
