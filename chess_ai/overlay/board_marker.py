"""Transparent widget painting move markers over the detected board.

The best move gets dark-blue corner markers on its source and destination
squares (as required by the specification). Alternative candidates use
distinct colors, optionally with an arrow connecting the two squares. All
geometry is computed from the detected quadrilateral with bilinear
interpolation, so markers follow rotated or perspective-distorted boards.
"""
from __future__ import annotations

import chess
from PyQt5 import QtCore, QtGui, QtWidgets

from ..vision.geometry import order_points, square_center, square_polygon

BEST_MOVE_COLOR = "#0B2A6B"


def screen_row_col(square: chess.Square, perspective_white: bool) -> tuple[int, int]:
    """Map a chess square to (screen column, screen row) inside the board."""
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    if perspective_white:
        return file_index, 7 - rank_index
    return 7 - file_index, rank_index


class MarkerSpec:
    """Everything needed to draw one candidate move."""

    def __init__(self, uci: str, color: str, is_best: bool, label: str) -> None:
        move = chess.Move.from_uci(uci)
        self.from_square = move.from_square
        self.to_square = move.to_square
        self.color = QtGui.QColor(color)
        self.is_best = is_best
        self.label = label


class BoardMarkerView(QtWidgets.QWidget):
    """Full-window transparent canvas for board markers."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self._corners = None
        self._perspective_white = True
        self._markers: list[MarkerSpec] = []
        self._window_origin = QtCore.QPoint(0, 0)

    def set_window_origin(self, point: QtCore.QPoint) -> None:
        self._window_origin = point

    def update_board(
        self,
        corners,
        perspective_white: bool,
        markers: list[MarkerSpec],
    ) -> None:
        self._corners = None if corners is None else order_points(corners)
        self._perspective_white = perspective_white
        self._markers = markers
        self.update()

    def clear(self) -> None:
        self._corners = None
        self._markers = []
        self.update()

    # ---------------------------------------------------------------- painting
    def _to_widget(self, point) -> QtCore.QPointF:
        return QtCore.QPointF(
            float(point[0]) - self._window_origin.x(),
            float(point[1]) - self._window_origin.y(),
        )

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._corners is None:
            return
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing)

            # Faint board outline for visual confirmation of detection.
            outline = [self._to_widget(p) for p in self._corners]
            pen = QtGui.QPen(QtGui.QColor(78, 123, 216, 160), 2.0)
            painter.setPen(pen)
            painter.drawPolygon(QtGui.QPolygonF(outline))

            for marker in self._markers:
                self._paint_marker(painter, marker)
        except Exception:  # pragma: no cover - a paint bug must never crash the app
            import logging
            logging.getLogger(__name__).exception("Overlay painting failed")
        finally:
            painter.end()

    def _paint_marker(self, painter: QtGui.QPainter, marker: MarkerSpec) -> None:
        for square in (marker.from_square, marker.to_square):
            col, row = screen_row_col(square, self._perspective_white)
            polygon = square_polygon(self._corners, col, row)
            self._paint_corner_markers(painter, polygon, marker)
        if not marker.is_best:
            self._paint_arrow(painter, marker)
        if marker.is_best:
            self._paint_square_label(painter, marker.to_square, marker.label,
                                     marker.color)

    def _paint_corner_markers(
        self, painter: QtGui.QPainter, polygon, marker: MarkerSpec
    ) -> None:
        points = [self._to_widget(p) for p in polygon]
        tl, tr, br, bl = points
        length = 0.28  # fraction of square side
        width = 5 if marker.is_best else 3
        pen = QtGui.QPen(marker.color, width, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
        painter.setPen(pen)
        for corner, neighbors in ((tl, (tr, bl)), (tr, (tl, br)),
                                  (br, (tr, bl)), (bl, (tl, br))):
            for neighbor in neighbors:
                end = QtCore.QPointF(
                    corner.x() + (neighbor.x() - corner.x()) * length,
                    corner.y() + (neighbor.y() - corner.y()) * length,
                )
                painter.drawLine(corner, end)

    def _paint_arrow(self, painter: QtGui.QPainter, marker: MarkerSpec) -> None:
        start_col, start_row = screen_row_col(marker.from_square, self._perspective_white)
        end_col, end_row = screen_row_col(marker.to_square, self._perspective_white)
        start = self._to_widget(square_center(self._corners, start_col, start_row))
        end = self._to_widget(square_center(self._corners, end_col, end_row))
        color = QtGui.QColor(marker.color)
        color.setAlpha(180)
        pen = QtGui.QPen(color, 3, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(start, end)
        # Arrow head: two short bars pointing backwards from the destination.
        travel = QtCore.QLineF(start, end)
        if travel.length() > 1:
            head = 12.0
            direction = QtGui.QVector2D(end - start).normalized()
            normal = QtGui.QVector2D(-direction.y(), direction.x())
            back = QtGui.QVector2D(-direction.x(), -direction.y())
            for sign in (-1.0, 1.0):
                tip = (QtGui.QVector2D(end)
                       + back * head * 0.7
                       + normal * head * 0.7 * sign)
                painter.drawLine(end, tip.toPointF())

    def _paint_square_label(
        self, painter: QtGui.QPainter, square: chess.Square, label: str, color: QtGui.QColor
    ) -> None:
        col, row = screen_row_col(square, self._perspective_white)
        center = self._to_widget(square_center(self._corners, col, row))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(11)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 1))
        painter.drawText(
            QtCore.QRectF(center.x() - 14, center.y() - 10, 28, 20),
            QtCore.Qt.AlignCenter,
            label,
        )
