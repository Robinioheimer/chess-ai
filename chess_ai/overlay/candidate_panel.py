"""Candidate-move table for the overlay.

The panel deliberately labels the three different numbers it displays:

* **Eval** - Stockfish's engine evaluation (centipawns / mate), not a chance.
* **Win %** - a calibrated *estimate* of the expected game result.
* **Confidence** - technical reliability of the current search (depth and
  stability), shown separately from the winning probability.
"""
from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from ..engine.types import AnalysisResult, MoveEvaluation

COLUMNS = ("", "#", "Move", "Eval", "Win % (est.)", "Opp. eval", "Depth", "Why this move")


class CandidatePanel(QtWidgets.QWidget):
    """Table of candidate moves plus analysis progress and legend."""

    visibility_changed = QtCore.pyqtSignal(int, bool)
    row_selected = QtCore.pyqtSignal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("CandidatePanel")
        self._result: AnalysisResult | None = None
        self._build_ui()

    # ------------------------------------------------------------------ ui
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QtWidgets.QLabel("Stockfish analysis")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        engine_row = QtWidgets.QHBoxLayout()
        engine_row.setContentsMargins(0, 0, 0, 0)
        self.engine_dot = QtWidgets.QLabel("●")
        self.engine_dot.setObjectName("EngineDot")
        self.engine_label = QtWidgets.QLabel("Engine: not connected")
        engine_row.addWidget(self.engine_dot)
        engine_row.addWidget(self.engine_label, 1)
        layout.addLayout(engine_row)

        # Dismissible error banner: errors never open modal dialogs while the
        # screen watcher is running, because those would block the UI loop.
        self.error_row = QtWidgets.QWidget()
        error_layout = QtWidgets.QHBoxLayout(self.error_row)
        error_layout.setContentsMargins(0, 0, 0, 0)
        self.error_label = QtWidgets.QLabel()
        self.error_label.setObjectName("ErrorLabel")
        self.error_label.setWordWrap(True)
        error_layout.addWidget(self.error_label, 1)
        self.clear_error_button = QtWidgets.QPushButton("✕ Clear")
        self.clear_error_button.setObjectName("ClearErrorButton")
        self.clear_error_button.setToolTip("Dismiss this error message")
        self.clear_error_button.clicked.connect(self.clear_error)
        error_layout.addWidget(self.clear_error_button)
        self.error_row.hide()
        layout.addWidget(self.error_row)

        self.status_label = QtWidgets.QLabel("Waiting for a board…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        self.confidence_label = QtWidgets.QLabel("Search confidence: –")
        self.confidence_label.setToolTip(
            "Reliability of the current search (depth reached and best-move "
            "stability). This is NOT the probability of winning."
        )
        layout.addWidget(self.confidence_label)

        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_selection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(7, QtWidgets.QHeaderView.Stretch)
        for col in (0, 1):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setMinimumWidth(560)
        layout.addWidget(self.table, 1)

        legend = QtWidgets.QLabel(
            "Eval = Stockfish engine score (±pawns or mate-in-N).\n"
            "Win % = calibrated estimate of the expected result, not a guarantee.\n"
            "Confidence = reliability of the search itself, not a game-outcome chance."
        )
        legend.setWordWrap(True)
        legend.setObjectName("Legend")
        layout.addWidget(legend)

    # -------------------------------------------------------------- updates
    def update_result(self, result: AnalysisResult) -> None:
        self._result = result
        if result.error:
            self.show_error(result.error)
            self.progress.setValue(0)
            self.table.setRowCount(0)
            return
        self.clear_error()

        self.status_label.setText(
            ("Searching… " if result.searching else "Analysis complete. ")
            + f"depth {result.depth}/{result.target_depth} · "
            f"{result.nodes:,} nodes · {result.nps:,} nps"
        )
        self.progress.setValue(int(result.progress * 100))
        if result.searching:
            self.progress.setRange(0, 100)
        else:
            self.progress.setValue(100)
        self.confidence_label.setText(
            f"Search confidence: {result.confidence * 100:.0f}% ({result.confidence_label})"
        )
        self._populate_rows(result.moves)

    def _populate_rows(self, moves: list[MoveEvaluation]) -> None:
        self.table.setRowCount(len(moves))
        for row, move in enumerate(moves):
            checkbox = QtWidgets.QCheckBox()
            checkbox.setChecked(move.visible)
            checkbox.stateChanged.connect(
                lambda state, index=row: self.visibility_changed.emit(
                    index, state == QtCore.Qt.Checked
                )
            )
            checkbox_container = QtWidgets.QWidget()
            check_layout = QtWidgets.QHBoxLayout(checkbox_container)
            check_layout.addWidget(checkbox)
            check_layout.setAlignment(QtCore.Qt.AlignCenter)
            check_layout.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(row, 0, checkbox_container)

            values = (
                "",  # checkbox column
                str(row + 1),
                move.san,
                move.eval_text,
                move.win_percent_text,
                move.opponent_eval,
                str(move.depth),
                move.explanation,
            )
            for col, text in enumerate(values):
                if col == 0:
                    continue
                item = QtWidgets.QTableWidgetItem(text)
                if col == 2:
                    font = item.font()
                    font.setBold(row == 0)
                    item.setFont(font)
                if col == 4:
                    item.setToolTip(
                        "Calibrated winning-probability estimate (Lichess logistic "
                        "model). An estimate, never a guarantee. Opponent's "
                        f"estimated result share: {move.opponent_win_probability * 100:.1f}%."
                    )
                if col == 7:
                    item.setToolTip(text)
                self.table.setItem(row, col, item)
        self.table.resizeRowsToContents()

    def set_status(self, text: str) -> None:
        # Ordinary status updates never hide a persistent error banner; the
        # user dismisses it explicitly (Clear button) or a successful analysis
        # clears it automatically.
        self.status_label.setText(text)

    def show_error(self, text: str) -> None:
        self.error_label.setText("⚠ " + text)
        self.error_row.show()

    def clear_error(self) -> None:
        self.error_label.setText("")
        self.error_row.hide()

    # Engine-state indicator. ``state`` is one of connected, analyzing,
    # unavailable, not_connected and drives the coloured status dot.
    _ENGINE_DOT_COLORS = {
        "connected": "#37C871",
        "analyzing": "#E8B33C",
        "unavailable": "#E0574F",
        "not_connected": "#8C97AB",
    }

    def set_engine_state(self, state: str, text: str) -> None:
        color = self._ENGINE_DOT_COLORS.get(state, "#8C97AB")
        self.engine_dot.setStyleSheet(f"color: {color};")
        self.engine_label.setText(text)
        self.engine_label.setToolTip(text)
        if state in ("connected", "unavailable", "not_connected"):
            self._engine_base_text = text

    def set_searching(self, searching: bool) -> None:
        if searching:
            base = getattr(self, "_engine_base_text", "Engine")
            self.engine_dot.setStyleSheet(
                f"color: {self._ENGINE_DOT_COLORS['analyzing']};"
            )
            self.engine_label.setText(f"{base} — analyzing…")
        elif getattr(self, "_engine_base_text", None):
            self.engine_dot.setStyleSheet(
                f"color: {self._ENGINE_DOT_COLORS['connected']};"
            )
            self.engine_label.setText(self._engine_base_text)

    # ------------------------------------------------------------- signals
    def _on_selection(self) -> None:
        items = self.table.selectionModel().selectedRows()
        if items:
            self.row_selected.emit(items[0].row())

    def visible_uci_moves(self) -> list[str]:
        if not self._result:
            return []
        return [m.uci for m in self._result.moves if m.visible]
