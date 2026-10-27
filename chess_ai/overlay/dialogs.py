"""Settings, FEN-input and manual board-correction dialogs."""
from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from ..config import AppConfig
from ..core.fen_utils import build_fen, validate_fen

PIECE_CHOICES = ["", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


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

    def __init__(self, screen_grid, perspective_white: bool, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Correct the detected position")
        self.setMinimumSize(560, 560)
        self._grid = [list(row) for row in screen_grid]
        self._perspective = perspective_white
        layout = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel(
            "Fix any wrong squares, then apply. The board is shown as it appears "
            "on screen (top row first)."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._combos: list[list[QtWidgets.QComboBox]] = []
        grid_widget = QtWidgets.QGridLayout()
        grid_widget.setSpacing(2)
        for row in range(8):
            row_combos = []
            for col in range(8):
                combo = QtWidgets.QComboBox()
                combo.addItems(PIECE_CHOICES)
                current = self._grid[row][col]
                combo.setCurrentText(current or "")
                combo.setFixedWidth(64)
                grid_widget.addWidget(combo, row, col)
                row_combos.append(combo)
            self._combos.append(row_combos)
        layout.addLayout(grid_widget)

        side_row = QtWidgets.QHBoxLayout()
        side_row.addWidget(QtWidgets.QLabel("Side to move:"))
        self.side_combo = QtWidgets.QComboBox()
        self.side_combo.addItems(["white (w)", "black (b)"])
        self.side_combo.setCurrentIndex(0 if perspective_white else 1)
        side_row.addWidget(self.side_combo)
        side_row.addStretch(1)
        layout.addLayout(side_row)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Apply | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def corrected_fen(self, side_to_move: str | None = None) -> str:
        from ..core.fen_utils import screen_grid_to_board_grid
        if side_to_move is None:
            side_to_move = "w" if self.side_combo.currentIndex() == 0 else "b"
        screen_grid = [
            [combo.currentText() or None for combo in row]
            for row in self._combos
        ]
        board_grid = screen_grid_to_board_grid(screen_grid, self._perspective)
        return build_fen(board_grid, side_to_move)
