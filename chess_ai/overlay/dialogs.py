"""Settings, FEN-input, board-region selector and manual board-correction dialogs."""
from __future__ import annotations

import chess
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..config import AppConfig
from ..core.fen_utils import (
    Grid,
    board_grid_to_screen_grid,
    board_to_grid,
    build_fen,
    screen_grid_to_board_grid,
    validate_fen,
)

PIECE_CHOICES = ["", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
PIECE_DESCRIPTIONS = {
    "": "Empty square",
    "P": "White Pawn (♙)",
    "N": "White Knight (♘)",
    "B": "White Bishop (♗)",
    "R": "White Rook (♖)",
    "Q": "White Queen (♕)",
    "K": "White King (♔)",
    "p": "Black Pawn (♟)",
    "n": "Black Knight (♞)",
    "b": "Black Bishop (♝)",
    "r": "Black Rook (♜)",
    "q": "Black Queen (♛)",
    "k": "Black King (♚)",
}


class SettingsDialog(QtWidgets.QDialog):
    """Edit engine, vision and overlay settings with validation."""

    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Chess AI – Settings")
        self.setMinimumWidth(460)
        self._config = config
        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # ----- engine tab
        engine_tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(engine_tab)
        path_row = QtWidgets.QHBoxLayout()
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("Auto-detect Stockfish")
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse_engine)
        detect = QtWidgets.QPushButton("Auto-detect")
        detect.clicked.connect(self._autodetect_engine)
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse)
        path_row.addWidget(detect)
        form.addRow("Stockfish path:", path_row)

        self.threads_spin = QtWidgets.QSpinBox()
        self.threads_spin.setRange(1, 256)
        form.addRow("Threads:", self.threads_spin)

        self.hash_spin = QtWidgets.QSpinBox()
        self.hash_spin.setRange(1, 1_048_576)
        self.hash_spin.setSuffix(" MB")
        form.addRow("Hash memory:", self.hash_spin)

        self.depth_spin = QtWidgets.QSpinBox()
        self.depth_spin.setRange(1, 100)
        form.addRow("Search depth:", self.depth_spin)

        self.movetime_spin = QtWidgets.QSpinBox()
        self.movetime_spin.setRange(0, 3_600_000)
        self.movetime_spin.setSuffix(" ms (0 = by depth)")
        form.addRow("Move time cap:", self.movetime_spin)

        self.multipv_spin = QtWidgets.QSpinBox()
        self.multipv_spin.setRange(1, 500)
        form.addRow("MultiPV lines:", self.multipv_spin)

        self.all_legal_check = QtWidgets.QCheckBox(
            "Analyze every legal move when time permits"
        )
        form.addRow("", self.all_legal_check)
        tabs.addTab(engine_tab, "Engine")

        # ----- vision tab
        vision_tab = QtWidgets.QWidget()
        vform = QtWidgets.QFormLayout(vision_tab)
        self.poll_spin = QtWidgets.QSpinBox()
        self.poll_spin.setRange(200, 60_000)
        self.poll_spin.setSuffix(" ms")
        vform.addRow("Screen poll interval:", self.poll_spin)

        self.stable_spin = QtWidgets.QSpinBox()
        self.stable_spin.setRange(1, 10)
        vform.addRow("Stable readings required:", self.stable_spin)

        self.perspective_combo = QtWidgets.QComboBox()
        self.perspective_combo.addItems(["auto", "white", "black"])
        vform.addRow("Board perspective:", self.perspective_combo)
        tabs.addTab(vision_tab, "Vision")

        # ----- overlay tab
        overlay_tab = QtWidgets.QWidget()
        oform = QtWidgets.QFormLayout(overlay_tab)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(30, 100)
        oform.addRow("Opacity:", self.opacity_slider)
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setRange(50, 300)
        oform.addRow("UI scale (%):", self.scale_slider)
        self.hotkey_edit = QtWidgets.QLineEdit()
        self.hotkey_edit.setToolTip("pynput notation, e.g. <ctrl>+<alt>+s")
        oform.addRow("Show/hide hotkey:", self.hotkey_edit)
        self.click_through_check = QtWidgets.QCheckBox(
            "Click-through (mouse input passes to the app below)"
        )
        oform.addRow("", self.click_through_check)
        tabs.addTab(overlay_tab, "Overlay")

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_values(self) -> None:
        engine, vision, overlay = self._config.engine, self._config.vision, self._config.overlay
        self.path_edit.setText(engine.stockfish_path)
        self.threads_spin.setValue(engine.threads)
        self.hash_spin.setValue(engine.hash_mb)
        self.depth_spin.setValue(engine.depth)
        self.movetime_spin.setValue(engine.movetime_ms)
        self.multipv_spin.setValue(engine.multipv)
        self.all_legal_check.setChecked(engine.analyze_all_legal)
        self.poll_spin.setValue(vision.poll_interval_ms)
        self.stable_spin.setValue(vision.stable_readings)
        self.perspective_combo.setCurrentText(vision.perspective)
        self.opacity_slider.setValue(int(overlay.opacity * 100))
        self.scale_slider.setValue(int(overlay.scale * 100))
        self.hotkey_edit.setText(overlay.show_hotkey)
        self.click_through_check.setChecked(overlay.click_through)

    def _browse_engine(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Stockfish executable", "",
            "Executables (*.exe);;All files (*)"
        )
        if path:
            self.path_edit.setText(path)

    def _autodetect_engine(self) -> None:
        from ..engine.stockfish_setup import find_stockfish
        info = find_stockfish(self.path_edit.text())
        if info is None:
            QtWidgets.QMessageBox.warning(
                self, "Not found",
                "Stockfish was not found. Install it or download it automatically "
                "from the engine status panel."
            )
        else:
            self.path_edit.setText(info.path)
            QtWidgets.QMessageBox.information(
                self, "Stockfish found", f"{info.name}\n{info.path}\n(source: {info.source})"
            )

    def _save(self) -> None:
        engine, vision, overlay = self._config.engine, self._config.vision, self._config.overlay
        engine.stockfish_path = self.path_edit.text().strip()
        engine.threads = self.threads_spin.value()
        engine.hash_mb = self.hash_spin.value()
        engine.depth = self.depth_spin.value()
        engine.movetime_ms = self.movetime_spin.value()
        engine.multipv = self.multipv_spin.value()
        engine.analyze_all_legal = self.all_legal_check.isChecked()
        vision.poll_interval_ms = self.poll_spin.value()
        vision.stable_readings = self.stable_spin.value()
        vision.perspective = self.perspective_combo.currentText()
        overlay.opacity = self.opacity_slider.value() / 100.0
        overlay.scale = self.scale_slider.value() / 100.0
        overlay.show_hotkey = self.hotkey_edit.text().strip() or overlay.show_hotkey
        overlay.click_through = self.click_through_check.isChecked()
        problems = self._config.validate()
        if problems:
            QtWidgets.QMessageBox.warning(self, "Invalid settings", "\n".join(problems))
            return
        self.accept()


class FenDialog(QtWidgets.QDialog):
    """Manual FEN entry with immediate validation feedback."""

    def __init__(self, current_fen: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Enter a FEN position")
        self.setMinimumWidth(520)
        layout = QtWidgets.QVBoxLayout(self)
        self.edit = QtWidgets.QLineEdit(current_fen)
        self.edit.setPlaceholderText("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        layout.addWidget(self.edit)
        self.feedback = QtWidgets.QLabel("")
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.edit.textChanged.connect(self._validate)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._validate(current_fen)

    def _validate(self, text: str) -> None:
        ok, message = validate_fen(text)
        if ok:
            self.feedback.setText("✓ Valid FEN")
            self.feedback.setStyleSheet("color: #2E9E5B;")
        else:
            self.feedback.setText("✗ " + message)
            self.feedback.setStyleSheet("color: #C73E63;")

    def fen(self) -> str:
        return self.edit.text().strip()

    def accept(self) -> None:
        ok, message = validate_fen(self.edit.text())
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Invalid FEN", message)
            return
        super().accept()


class CorrectionDialog(QtWidgets.QDialog):
    """8x8 piece picker for correcting a misrecognized board."""

    def __init__(
        self,
        screen_grid,
        perspective_white: bool,
        parent=None,
        side_to_move: str | None = None,
        uncertain_squares: list[str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Correct the detected position")
        self.setMinimumSize(620, 620)
        self._grid = [list(row) for row in screen_grid]
        self._perspective = perspective_white
        self._uncertain = set(uncertain_squares or [])
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "Select the correct piece on any misrecognized square, then click Apply.\n"
            "Uppercase (P N B R Q K) = White pieces · Lowercase (p n b r q k) = Black pieces · "
            "Empty = empty square."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Quick action toolbar
        quick_row = QtWidgets.QHBoxLayout()
        start_btn = QtWidgets.QPushButton("↺ Starting position")
        start_btn.setToolTip("Fill the grid with the standard chess starting position")
        start_btn.clicked.connect(self._reset_to_start)
        quick_row.addWidget(start_btn)

        clear_btn = QtWidgets.QPushButton("🗑 Clear all")
        clear_btn.setToolTip("Empty all 64 squares")
        clear_btn.clicked.connect(self._clear_all)
        quick_row.addWidget(clear_btn)

        flip_btn = QtWidgets.QPushButton("⇅ Flip orientation")
        flip_btn.setToolTip("Switch between White at bottom and Black at bottom")
        flip_btn.clicked.connect(self._flip_perspective)
        quick_row.addWidget(flip_btn)
        quick_row.addStretch(1)
        layout.addLayout(quick_row)

        self._combos: list[list[QtWidgets.QComboBox]] = []
        self._file_labels: list[QtWidgets.QLabel] = []
        self._rank_labels: list[QtWidgets.QLabel] = []
        grid_widget = QtWidgets.QGridLayout()
        grid_widget.setSpacing(3)

        # Top file headers (a..h)
        for col in range(8):
            lbl = QtWidgets.QLabel("")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("font-weight: bold; color: #9FB0CC;")
            grid_widget.addWidget(lbl, 0, col + 1)
            self._file_labels.append(lbl)

        for row in range(8):
            rank_lbl = QtWidgets.QLabel("")
            rank_lbl.setAlignment(QtCore.Qt.AlignCenter)
            rank_lbl.setFixedWidth(22)
            rank_lbl.setStyleSheet("font-weight: bold; color: #9FB0CC;")
            grid_widget.addWidget(rank_lbl, row + 1, 0)
            self._rank_labels.append(rank_lbl)

            row_combos = []
            for col in range(8):
                combo = QtWidgets.QComboBox()
                combo.addItems(PIECE_CHOICES)
                for idx, sym in enumerate(PIECE_CHOICES):
                    combo.setItemData(idx, PIECE_DESCRIPTIONS[sym], QtCore.Qt.ToolTipRole)
                current = self._grid[row][col]
                combo.setCurrentText(current or "")
                combo.setFixedWidth(64)
                combo.currentTextChanged.connect(self._update_validation)
                grid_widget.addWidget(combo, row + 1, col + 1)
                row_combos.append(combo)
            self._combos.append(row_combos)
        layout.addLayout(grid_widget)

        side_row = QtWidgets.QHBoxLayout()
        side_row.addWidget(QtWidgets.QLabel("Side to move:"))
        self.side_combo = QtWidgets.QComboBox()
        self.side_combo.addItems(["white (w)", "black (b)"])
        if side_to_move in ("w", "b"):
            self.side_combo.setCurrentIndex(0 if side_to_move == "w" else 1)
        else:
            self.side_combo.setCurrentIndex(0 if perspective_white else 1)
        self.side_combo.currentIndexChanged.connect(self._update_validation)
        side_row.addWidget(self.side_combo)

        self.learn_check = QtWidgets.QCheckBox("Learn piece templates from this correction")
        self.learn_check.setChecked(True)
        self.learn_check.setToolTip(
            "Update piece recognition templates from the corrected squares so "
            "the recognizer remembers these piece shapes on subsequent moves."
        )
        side_row.addSpacing(12)
        side_row.addWidget(self.learn_check)
        side_row.addStretch(1)
        layout.addLayout(side_row)

        self.validation_label = QtWidgets.QLabel("")
        self.validation_label.setWordWrap(True)
        layout.addWidget(self.validation_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Cancel
        )
        apply_button = buttons.button(QtWidgets.QDialogButtonBox.Apply)
        if apply_button is not None:
            apply_button.clicked.connect(self.accept)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._refresh_coordinates_and_styles()
        self._update_validation()

    def _square_name(self, screen_row: int, screen_col: int) -> str:
        files = "abcdefgh"
        if self._perspective:
            return f"{files[screen_col]}{8 - screen_row}"
        return f"{files[7 - screen_col]}{screen_row + 1}"

    def _refresh_coordinates_and_styles(self) -> None:
        files = "abcdefgh" if self._perspective else "hgfedcba"
        for col, ch in enumerate(files):
            self._file_labels[col].setText(ch)
        for row in range(8):
            rank_num = (8 - row) if self._perspective else (row + 1)
            self._rank_labels[row].setText(str(rank_num))
            for col in range(8):
                sq_name = self._square_name(row, col)
                combo = self._combos[row][col]
                sym = combo.currentText()
                desc = PIECE_DESCRIPTIONS.get(sym, sym)
                is_uncertain = sq_name in self._uncertain
                combo.setToolTip(
                    f"Square {sq_name}: {desc}"
                    + (" (low confidence — please check)" if is_uncertain else "")
                )
                is_light = (row + col) % 2 == 0
                bg = "#2B3856" if is_light else "#1A2233"
                border = "2px solid #E8B33C" if is_uncertain else "1px solid #3A4C73"
                combo.setStyleSheet(
                    f"QComboBox {{ background-color: {bg}; color: #EAF0FF; "
                    f"border: {border}; border-radius: 4px; padding: 3px; font-weight: bold; }}"
                )

    def _reset_to_start(self) -> None:
        start_screen = board_grid_to_screen_grid(
            board_to_grid(chess.Board()), self._perspective
        )
        for row in range(8):
            for col in range(8):
                self._combos[row][col].setCurrentText(start_screen[row][col] or "")
        self._update_validation()

    def _clear_all(self) -> None:
        for row in range(8):
            for col in range(8):
                self._combos[row][col].setCurrentText("")
        self._update_validation()

    def _flip_perspective(self) -> None:
        self._perspective = not self._perspective
        self._refresh_coordinates_and_styles()
        self._update_validation()

    def _update_validation(self, *_args) -> None:
        self._refresh_coordinates_and_styles()
        fen = self.corrected_fen()
        ok, message = validate_fen(fen)
        if ok:
            self.validation_label.setText(f"✓ Valid position: {fen}")
            self.validation_label.setStyleSheet("color: #2E9E5B;")
        else:
            self.validation_label.setText(f"✗ {message}")
            self.validation_label.setStyleSheet("color: #FF9A93;")

    def corrected_screen_grid(self) -> Grid:
        return [
            [combo.currentText() or None for combo in row]
            for row in self._combos
        ]

    @property
    def perspective_white(self) -> bool:
        return self._perspective

    def corrected_fen(self, side_to_move: str | None = None) -> str:
        if side_to_move is None:
            side_to_move = "w" if self.side_combo.currentIndex() == 0 else "b"
        screen_grid = self.corrected_screen_grid()
        board_grid = screen_grid_to_board_grid(screen_grid, self._perspective)
        return build_fen(board_grid, side_to_move)

    def accept(self) -> None:
        ok, message = validate_fen(self.corrected_fen())
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Invalid board position", message)
            return
        super().accept()


class BoardRegionSelector(QtWidgets.QDialog):
    """Full-screen rubber-band selector for manually locking the board area."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Chessboard Area")
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setCursor(QtCore.Qt.CrossCursor)
        self._origin: QtCore.QPoint | None = None
        self._current: QtCore.QPoint | None = None
        self._corners: np.ndarray | None = None
        self._reset_auto = False

        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())
        else:
            self.resize(1280, 720)

        bar = QtWidgets.QHBoxLayout(self)
        bar.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        banner = QtWidgets.QWidget(self)
        banner.setStyleSheet(
            "background-color: rgba(18, 22, 30, 235); color: #EAF0FF; "
            "border: 1px solid #4E7BD8; border-radius: 8px; padding: 8px;"
        )
        blayout = QtWidgets.QHBoxLayout(banner)
        label = QtWidgets.QLabel(
            "Drag a box around the 8×8 chessboard with the left mouse button "
            "(Esc = cancel)."
        )
        blayout.addWidget(label)
        auto_btn = QtWidgets.QPushButton("Reset to Auto-Detect")
        auto_btn.clicked.connect(self._choose_auto)
        blayout.addWidget(auto_btn)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        blayout.addWidget(cancel_btn)
        bar.addWidget(banner)

    def _choose_auto(self) -> None:
        self._reset_auto = True
        self._corners = None
        self.accept()

    @property
    def selected_corners(self) -> np.ndarray | None:
        return self._corners

    @property
    def reset_to_auto(self) -> bool:
        return self._reset_auto

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(10, 14, 22, 110))
        if self._origin is not None and self._current is not None:
            rect = QtCore.QRect(self._origin, self._current).normalized()
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
            painter.fillRect(rect, QtCore.Qt.transparent)
            painter.setCompositionMode(QtGui.QPainter.CompositionMode_SourceOver)
            pen = QtGui.QPen(QtGui.QColor("#4E7BD8"), 2.5)
            painter.setPen(pen)
            painter.drawRect(rect)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self._origin = event.pos()
            self._current = event.pos()
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._origin is not None:
            self._current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton and self._origin is not None:
            self._current = event.pos()
            rect = QtCore.QRect(self._origin, self._current).normalized()
            if rect.width() >= 64 and rect.height() >= 64:
                win_tl = self.geometry().topLeft()
                x0 = float(win_tl.x() + rect.left())
                y0 = float(win_tl.y() + rect.top())
                x1 = float(win_tl.x() + rect.right())
                y1 = float(win_tl.y() + rect.bottom())
                self._corners = np.array(
                    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
                )
                self.accept()
            else:
                self._origin = None
                self._current = None
                self.update()
