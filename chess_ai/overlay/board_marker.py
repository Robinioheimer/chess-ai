"""Transparent canvas that paints move arrows and % accuracy badges over the board."""
from __future__ import annotations

import math
from typing import Optional

import chess
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..vision.geometry import square_center, square_polygon

BEST_MOVE_COLOR = "#0B2A6B"


class MarkerSpec:
    """One move marker (arrow + corner brackets + % accuracy badge) to draw."""

    def __init__(
        self,
        uci: str,
        color: str,
        is_best: bool,
        label: str,
        percent_text: str = "",
        rank: int = 1,
    ) -> None:
        move = chess.Move.from_uci(uci)
        self.uci = uci
        self.from_square = move.from_square
        self.to_square = move.to_square
        self.color = QtGui.QColor(color)
        self.is_best = is_best
        self.label = label
        self.percent_text = percent_text
        self.rank = rank

    @property
    def badge_text(self) -> str:
        """Combined SAN move and win/accuracy percentage shown on the arrow."""
        if self.percent_text:
            return f"{self.label} ({self.percent_text})"
        return self.label


def screen_row_col(square: chess.Square, perspective_white: bool) -> tuple[int, int]:
    """Convert a python-chess square to ``(col, row)`` on screen."""
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    if perspective_white:
        return file_idx, 7 - rank_idx
    return 7 - file_idx, rank_idx


class BoardMarkerView(QtWidgets.QWidget):
    """Frameless transparent widget that renders board arrows and % labels."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self._corners: Optional[np.ndarray] = None
        self._perspective_white: bool = True
        self._markers: list[MarkerSpec] = []
        self._offset = QtCore.QPoint(0, 0)

    def set_window_offset(self, offset: QtCore.QPoint) -> None:
        """Set screen-origin offset so screen coordinates map to widget space."""
        self._offset = offset
        self.update()

    def set_window_origin(self, origin: QtCore.QPoint) -> None:
        """Alias for :meth:`set_window_offset` used by ``MainOverlay``."""
        self.set_window_offset(origin)

    def set_board(
        self,
        corners: Optional[np.ndarray],
        perspective_white: bool,
    ) -> None:
        self._corners = corners
        self._perspective_white = perspective_white
        self.update()

    def set_markers(self, markers: list[MarkerSpec]) -> None:
        self._markers = markers
        self.update()

    def update_board(
        self,
        corners: Optional[np.ndarray],
        perspective_white: bool,
        markers: list[MarkerSpec],
    ) -> None:
        self._corners = corners
        self._perspective_white = perspective_white
        self._markers = list(markers)
        self.update()

    def clear(self) -> None:
        self._corners = None
        self._markers = []
        self.update()

    # -- painting ----------------------------------------------------------

    def _to_local(self, pt: tuple[float, float] | np.ndarray) -> QtCore.QPointF:
        return QtCore.QPointF(
            float(pt[0]) - self._offset.x(),
            float(pt[1]) - self._offset.y(),
        )

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        if self._corners is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing, True)

        # Subtle outline around the detected board so the user sees alignment.
        outline_pen = QtGui.QPen(QtGui.QColor(76, 139, 245, 160), 2.0)
        outline_pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(outline_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        poly = QtGui.QPolygonF([self._to_local(pt) for pt in self._corners])
        painter.drawPolygon(poly)

        # Paint alternative moves first and the #1 best move last so the best
        # move arrow and its % badge always sit on top.
        for marker in reversed(self._markers):
            self._paint_marker(painter, marker)

    def _paint_marker(self, painter: QtGui.QPainter, marker: MarkerSpec) -> None:
        for square in (marker.from_square, marker.to_square):
            col, row = screen_row_col(square, self._perspective_white)
            polygon = square_polygon(self._corners, col, row)
            self._paint_corner_markers(painter, polygon, marker)
        # Draw a directional arrow for EVERY move (including the #1 best move)
        # together with its % accuracy/win-probability badge.
        self._paint_arrow(painter, marker)
        self._paint_arrow_badge(painter, marker)

    def _paint_corner_markers(
        self,
        painter: QtGui.QPainter,
        polygon: np.ndarray,
        marker: MarkerSpec,
    ) -> None:
        color = QtGui.QColor(marker.color)
        color.setAlpha(240 if marker.is_best else 185)
        width = 4.0 if marker.is_best else 2.5
        pen = QtGui.QPen(color, width)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(pen)

        # Draw L-shaped corner brackets at each of the four square vertices.
        for idx in range(4):
            corner = polygon[idx]
            prev_pt = polygon[(idx - 1) % 4]
            next_pt = polygon[(idx + 1) % 4]
            leg_a = corner + 0.28 * (prev_pt - corner)
            leg_b = corner + 0.28 * (next_pt - corner)
            c_pt = self._to_local(corner)
            a_pt = self._to_local(leg_a)
            b_pt = self._to_local(leg_b)
            painter.drawLine(c_pt, a_pt)
            painter.drawLine(c_pt, b_pt)

        if marker.is_best:
            fill = QtGui.QColor(marker.color)
            fill.setAlpha(45)
            painter.setBrush(QtGui.QBrush(fill))
            painter.setPen(QtCore.Qt.NoPen)
            qpoly = QtGui.QPolygonF([self._to_local(p) for p in polygon])
            painter.drawPolygon(qpoly)

    def _paint_arrow(self, painter: QtGui.QPainter, marker: MarkerSpec) -> None:
        assert self._corners is not None
        c1, r1 = screen_row_col(marker.from_square, self._perspective_white)
        c2, r2 = screen_row_col(marker.to_square, self._perspective_white)
        start = self._to_local(square_center(self._corners, c1, r1))
        end = self._to_local(square_center(self._corners, c2, r2))

        dx = end.x() - start.x()
        dy = end.y() - start.y()
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return
        ux, uy = dx / length, dy / length
        px, py = -uy, ux

        head_len = 18.0 if marker.is_best else 14.0
        head_half_w = 10.5 if marker.is_best else 7.5
        shaft_width = 7.0 if marker.is_best else 4.5

        # Stop the shaft at the base of the arrowhead so the tip is crisp.
        shaft_end = QtCore.QPointF(
            end.x() - ux * (head_len * 0.75),
            end.y() - uy * (head_len * 0.75),
        )
        wing_left = QtCore.QPointF(
            end.x() - ux * head_len + px * head_half_w,
            end.y() - uy * head_len + py * head_half_w,
        )
        wing_right = QtCore.QPointF(
            end.x() - ux * head_len - px * head_half_w,
            end.y() - uy * head_len - py * head_half_w,
        )
        head_poly = QtGui.QPolygonF([end, wing_left, wing_right])

        # High-contrast outer halo so dark-blue (#0B2A6B) best-move arrows and
        # coloured candidate arrows stand out clearly on both dark and light squares.
        halo_color = (
            QtGui.QColor(255, 255, 255, 225)
            if marker.is_best
            else QtGui.QColor(15, 19, 26, 190)
        )
        halo_pen = QtGui.QPen(halo_color, shaft_width + 3.0)
        halo_pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(halo_pen)
        painter.drawLine(start, shaft_end)

        head_outline_pen = QtGui.QPen(halo_color, 2.2 if marker.is_best else 1.5)
        head_outline_pen.setJoinStyle(QtCore.Qt.RoundJoin)
        color = QtGui.QColor(marker.color)
        color.setAlpha(245 if marker.is_best else 215)

        shaft_pen = QtGui.QPen(color, shaft_width)
        shaft_pen.setCapStyle(QtCore.Qt.RoundCap)
        painter.setPen(shaft_pen)
        painter.drawLine(start, shaft_end)

        painter.setPen(head_outline_pen)
        painter.setBrush(QtGui.QBrush(color))
        painter.drawPolygon(head_poly)

    def _paint_arrow_badge(
        self,
        painter: QtGui.QPainter,
        marker: MarkerSpec,
    ) -> None:
        """Paint a pill badge showing the move and its % accuracy/win rate."""
        assert self._corners is not None
        text = marker.badge_text
        if not text:
            return
        c1, r1 = screen_row_col(marker.from_square, self._perspective_white)
        c2, r2 = screen_row_col(marker.to_square, self._perspective_white)
        start = self._to_local(square_center(self._corners, c1, r1))
        end = self._to_local(square_center(self._corners, c2, r2))

        # Place badge near the destination end of the arrow (68% along shaft)
        # so the start square, arrow direction and target square remain clear.
        t = 0.68 if marker.is_best else 0.58
        anchor = QtCore.QPointF(
            start.x() + t * (end.x() - start.x()),
            start.y() + t * (end.y() - start.y()),
        )

        font = painter.font()
        font.setBold(True)
        font.setPointSizeF(9.5 if marker.is_best else 8.5)
        painter.setFont(font)
        metrics = QtGui.QFontMetricsF(font)
        rect = metrics.boundingRect(text).adjusted(-7, -3, 7, 3)
        rect.moveCenter(anchor)

        bg = QtGui.QColor(15, 19, 26, 235 if marker.is_best else 215)
        border = QtGui.QColor(marker.color)
        border.setAlpha(255)
        if marker.is_best:
            # Brighten border slightly for the dark-blue best move pill
            border = QtGui.QColor("#4C8BF5")

        painter.setBrush(QtGui.QBrush(bg))
        painter.setPen(QtGui.QPen(border, 2.0 if marker.is_best else 1.4))
        painter.drawRoundedRect(rect, 6, 6)

        painter.setPen(QtGui.QColor("#FFFFFF"))
        painter.drawText(rect, QtCore.Qt.AlignCenter, text)
