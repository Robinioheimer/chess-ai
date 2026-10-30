"""Clean interactive training board with its own live Stockfish analysis."""
from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets
import chess

from ..engine.analyzer import AnalysisManager
from ..engine.uci import UciEngine
from ..engine.types import AnalysisResult


PIECES = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}


class AnalysisBridge(QtCore.QObject):
    result = QtCore.pyqtSignal(object)
    status = QtCore.pyqtSignal(str)


class ChessBoardWidget(QtWidgets.QWidget):
    position_changed = QtCore.pyqtSignal(str)
    move_made = QtCore.pyqtSignal(str)

    LIGHT = QtGui.QColor("#F0F0F0")
    DARK = QtGui.QColor("#252525")
    SELECT = QtGui.QColor(70, 135, 220, 105)
    LAST = QtGui.QColor(245, 205, 65, 105)
    TARGET = QtGui.QColor(30, 30, 30, 105)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.board = chess.Board()
        self.flipped = False
        self.selected = None
        self.drag_pos = None
        self.last_move = None
        self.setMinimumSize(280, 280)
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def _geometry(self):
        side = min(self.width(), self.height())
        x = (self.width() - side) / 2
        y = (self.height() - side) / 2
        return x, y, side, side / 8

    def square_at(self, point):
        x, y, side, cell = self._geometry()
        col, row = int((point.x() - x) / cell), int((point.y() - y) / cell)
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        file = 7 - col if self.flipped else col
        rank = row if self.flipped else 7 - row
        return chess.square(file, rank)

    def square_rect(self, square):
        x, y, side, cell = self._geometry()
        file, rank = chess.square_file(square), chess.square_rank(square)
        col, row = (7-file, rank) if self.flipped else (file, 7-rank)
        return QtCore.QRectF(x + col*cell, y + row*cell, cell, cell)

    def set_position(self, fen):
        self.board.set_fen(fen)
        self.selected = None
        self.drag_pos = None
        self.last_move = None
        self.update()

    def reset(self):
        self.board.reset()
        self.selected = None
        self.last_move = None
        self.update()
        self.position_changed.emit(self.board.fen())

    def flip(self):
        self.flipped = not self.flipped
        self.update()

    def _legal_from(self, square):
        return [m for m in self.board.legal_moves if m.from_square == square]

    def _play(self, move):
        san = self.board.san(move)
        self.board.push(move)
        self.last_move = move
        self.selected = None
        self.drag_pos = None
        self.update()
        self.move_made.emit(san)
        self.position_changed.emit(self.board.fen())

    def _try_move(self, source, target):
        candidates = [m for m in self._legal_from(source) if m.to_square == target]
        if not candidates:
            return False
        move = candidates[0]
        if move.promotion:
            choice, ok = QtWidgets.QInputDialog.getItem(
                self, "Promotion", "Promote pawn to:", ["Queen", "Rook", "Bishop", "Knight"], 0, False
            )
            if not ok:
                return False
            move.promotion = {"Queen": chess.QUEEN, "Rook": chess.ROOK,
                              "Bishop": chess.BISHOP, "Knight": chess.KNIGHT}[choice]
        self._play(move)
        return True

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton:
            return
        square = self.square_at(event.pos())
        if square is None:
            return
        piece = self.board.piece_at(square)
        if piece and piece.color == self.board.turn:
            self.selected = square
            self.drag_pos = event.pos()
            self.update()
        elif self.selected is not None:
            if not self._try_move(self.selected, square):
                self.selected = None
                self.drag_pos = None
                self.update()

    def mouseMoveEvent(self, event):
        if self.selected is not None and event.buttons() & QtCore.Qt.LeftButton:
            self.drag_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton or self.selected is None:
            return
        target = self.square_at(event.pos())
        source = self.selected
        if target is not None and target != source:
            if not self._try_move(source, target):
                self.selected = None
                self.drag_pos = None
                self.update()
        else:
            self.drag_pos = None
            self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        x, y, side, cell = self._geometry()
        legal_targets = {m.to_square for m in self._legal_from(self.selected)} if self.selected is not None else set()
        for square in chess.SQUARES:
            rect = self.square_rect(square)
            file, rank = chess.square_file(square), chess.square_rank(square)
            color = self.LIGHT if (file + rank) % 2 else self.DARK
            painter.fillRect(rect, color)
            if self.last_move and square in (self.last_move.from_square, self.last_move.to_square):
                painter.fillRect(rect, self.LAST)
            if square == self.selected:
                painter.fillRect(rect, self.SELECT)
            if square in legal_targets:
                painter.setBrush(self.TARGET)
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(rect.center(), cell*0.12, cell*0.12)
            piece = self.board.piece_at(square)
            if piece and not (square == self.selected and self.drag_pos is not None
                              and QtWidgets.QApplication.mouseButtons() & QtCore.Qt.LeftButton):
                self._draw_piece(painter, rect, piece, cell)
            if file == (0 if not self.flipped else 7):
                painter.setPen(QtGui.QColor("#777777" if color == self.LIGHT else "#DDDDDD"))
                painter.setFont(QtGui.QFont("Arial", max(8, int(cell*0.12)), QtGui.QFont.Bold))
                painter.drawText(rect.adjusted(3, 1, -2, -2), QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop, str(rank+1))
            if rank == (0 if not self.flipped else 7):
                painter.setPen(QtGui.QColor("#777777" if color == self.LIGHT else "#DDDDDD"))
                painter.setFont(QtGui.QFont("Arial", max(8, int(cell*0.12)), QtGui.QFont.Bold))
                painter.drawText(rect.adjusted(0, 0, -4, -2), QtCore.Qt.AlignRight | QtCore.Qt.AlignBottom,
                                 chess.FILE_NAMES[file])
        if self.selected is not None and self.drag_pos is not None and QtWidgets.QApplication.mouseButtons() & QtCore.Qt.LeftButton:
            piece = self.board.piece_at(self.selected)
            if piece:
                rect = QtCore.QRectF(self.drag_pos.x()-cell*.48, self.drag_pos.y()-cell*.48, cell*.96, cell*.96)
                self._draw_piece(painter, rect, piece, cell)

    @staticmethod
    def _draw_piece(painter, rect, piece, cell):
        symbol = PIECES[piece.symbol()]
        font = QtGui.QFont("Segoe UI Symbol", max(22, int(cell*.76)))
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        painter.setFont(font)
        # Draw a contrasting offset first so both piece colors stay readable
        # on both square colors, then draw the actual piece glyph.
        shadow = QtGui.QColor("#171717" if piece.color else "#F5F5F5")
        foreground = QtGui.QColor("#FAFAFA" if piece.color else "#151515")
        painter.setPen(shadow)
        painter.drawText(rect.translated(1.2, 1.2), QtCore.Qt.AlignCenter, symbol)
        painter.setPen(foreground)
        painter.drawText(rect, QtCore.Qt.AlignCenter, symbol)


class TrainingBoardOverlay(QtWidgets.QWidget):
    """Resizable training board with click/drag moves and live MultiPV analysis."""

    def __init__(self, engine_path=None, engine_config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chess AI · Training Board")
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        self.resize(820, 760)
        self.setMinimumSize(620, 650)
        self._engine = None
        self._analyzer = None
        self._engine_path = engine_path
        self._engine_config = engine_config
        self._bridge = AnalysisBridge(self)
        self._bridge.result.connect(self._show_analysis)
        self._last_analysis_fen = None
        self._build_ui()
        # Build widgets before connecting signals to their methods.
        self._bridge.status.connect(self.analysis_state.setText)
        if engine_path and engine_config:
            self._start_analysis(engine_path, engine_config)
        else:
            self.analysis_state.setText("Stockfish nicht verfügbar")

    def _build_ui(self):
        self.setStyleSheet("""
            QWidget { background: #171717; color: #F4F4F4; font-family: Segoe UI; font-size: 10pt; }
            QFrame#Card { background: #222222; border: 1px solid #3A3A3A; border-radius: 10px; }
            QPushButton { background: #303030; border: 1px solid #505050; border-radius: 6px; padding: 7px 12px; }
            QPushButton:hover { background: #414141; }
            QPushButton:pressed { background: #555555; }
            QSlider::groove:horizontal { height: 5px; background: #444; border-radius: 2px; }
            QSlider::handle:horizontal { background: #EEE; width: 14px; margin: -5px 0; border-radius: 7px; }
            QListWidget { background: #202020; border: 1px solid #3A3A3A; border-radius: 7px; padding: 4px; }
        """)
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)

        board_card = QtWidgets.QFrame()
        board_card.setObjectName("Card")
        board_layout = QtWidgets.QVBoxLayout(board_card)
        board_layout.setContentsMargins(12, 12, 12, 12)
        title_row = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("TRAINING BOARD")
        title.setStyleSheet("font-size: 13pt; font-weight: 700; letter-spacing: 1px;")
        title_row.addWidget(title)
        title_row.addStretch()
        self.turn_label = QtWidgets.QLabel("Weiß am Zug")
        self.turn_label.setStyleSheet("color: #BBBBBB;")
        title_row.addWidget(self.turn_label)
        board_layout.addLayout(title_row)

        self.board_view = ChessBoardWidget()
        self.board_view.position_changed.connect(self._position_changed)
        self.board_view.move_made.connect(self._move_made)
        board_layout.addWidget(self.board_view, 1)

        controls = QtWidgets.QHBoxLayout()
        for label, callback in (("Neues Spiel", self._reset), ("Zurück", self._undo),
                                ("Drehen", self.board_view.flip)):
            button = QtWidgets.QPushButton(label)
            button.clicked.connect(callback)
            controls.addWidget(button)
        board_layout.addLayout(controls)
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Brettgröße"))
        self.size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.size_slider.setRange(620, 1120)
        self.size_slider.setValue(820)
        self.size_slider.valueChanged.connect(
            lambda value: self.resize(value, max(650, int(value * 0.92)))
        )
        size_row.addWidget(self.size_slider, 1)
        board_layout.addLayout(size_row)
        root.addWidget(board_card, 3)

        side_card = QtWidgets.QFrame()
        side_card.setObjectName("Card")
        side_layout = QtWidgets.QVBoxLayout(side_card)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_title = QtWidgets.QLabel("STOCKFISH")
        side_title.setStyleSheet("font-size: 13pt; font-weight: 700; letter-spacing: 1px;")
        side_layout.addWidget(side_title)
        self.analysis_state = QtWidgets.QLabel("Warte auf Stellung …")
        self.analysis_state.setStyleSheet("color: #AAAAAA;")
        side_layout.addWidget(self.analysis_state)
        self.best_move = QtWidgets.QLabel("—")
        self.best_move.setWordWrap(True)
        self.best_move.setStyleSheet("font-size: 20pt; font-weight: 700; padding: 10px 0;")
        side_layout.addWidget(self.best_move)
        self.eval_label = QtWidgets.QLabel("Bewertung: —")
        side_layout.addWidget(self.eval_label)
        side_layout.addWidget(QtWidgets.QLabel("Beste Züge"))
        self.moves_list = QtWidgets.QListWidget()
        side_layout.addWidget(self.moves_list, 1)
        self.history_label = QtWidgets.QLabel("Zug: 1")
        self.history_label.setStyleSheet("color: #AAAAAA;")
        side_layout.addWidget(self.history_label)
        self.transparency = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.transparency.setRange(65, 100)
        self.transparency.setValue(100)
        self.transparency.valueChanged.connect(lambda value: self.setWindowOpacity(value/100))
        side_layout.addWidget(QtWidgets.QLabel("Fenster-Deckkraft"))
        side_layout.addWidget(self.transparency)
        root.addWidget(side_card, 2)

    def _start_analysis(self, engine_path, engine_config):
        try:
            self._engine = UciEngine(engine_path)
            self._analyzer = AnalysisManager(
                self._engine, engine_config,
                on_result=self._bridge.result.emit,
                on_status=self._set_analysis_status,
            )
            self._analyzer.start()
            self._position_changed(self.board_view.board.fen())
        except Exception as exc:
            self.analysis_state.setText(f"Stockfish-Fehler: {exc}")

    def _set_analysis_status(self, message):
        self._bridge.status.emit(message)

    def _position_changed(self, fen):
        self.turn_label.setText("Weiß am Zug" if self.board_view.board.turn else "Schwarz am Zug")
        self.history_label.setText(f"Zug: {self.board_view.board.fullmove_number}")
        self.best_move.setText("Berechne …")
        self.eval_label.setText("Bewertung: …")
        self.moves_list.clear()
        self._last_analysis_fen = fen
        if self._analyzer is not None:
            self._analyzer.request_analysis(fen)

    def _move_made(self, san):
        self.analysis_state.setText(f"Letzter Zug: {san}")

    def _show_analysis(self, result: AnalysisResult):
        if result.fen != self._last_analysis_fen:
            return
        if result.error:
            self.analysis_state.setText(result.error)
            return
        if not result.moves:
            self.best_move.setText("Kein legaler Zug")
            self.eval_label.setText("Bewertung: —")
            self.moves_list.clear()
            return
        best = result.moves[0]
        self.best_move.setText(f"{best.san}   ({best.uci})")
        score = f"Matt in {best.mate}" if best.mate is not None else f"{(best.cp or 0)/100:+.2f}"
        self.eval_label.setText(f"Bewertung: {score}  ·  Tiefe {result.depth}")
        self.moves_list.clear()
        for item in result.moves[:5]:
            value = f"M{item.mate}" if item.mate is not None else f"{(item.cp or 0)/100:+.2f}"
            line = "  ".join(item.pv_san[:6])
            self.moves_list.addItem(f"{item.multipv_index}.  {item.san}   {value}\n    {line}")
        self.analysis_state.setText("Analysiert" if not result.searching else "Analysiert · Suche läuft")

    def _reset(self):
        self.board_view.reset()

    def _undo(self):
        if self.board_view.board.move_stack:
            self.board_view.board.pop()
            self.board_view.last_move = self.board_view.board.peek() if self.board_view.board.move_stack else None
            self.board_view.selected = None
            self.board_view.update()
            self._position_changed(self.board_view.board.fen())

    def closeEvent(self, event):
        if self._analyzer is not None:
            self._analyzer.shutdown()
            self._analyzer = None
        event.accept()
