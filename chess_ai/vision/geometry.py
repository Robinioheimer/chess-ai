"""Pure geometry helpers for board quadrilaterals (no GUI/Qt dependency)."""
from __future__ import annotations

import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    point_sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(point_sum)]   # tl: smallest x+y
    rect[2] = pts[np.argmax(point_sum)]   # br: largest x+y
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]        # tr: smallest y-x
    rect[3] = pts[np.argmax(diff)]        # bl: largest y-x
    return rect


def bilinear_point(corners: np.ndarray, u: float, v: float) -> np.ndarray:
    """Interpolate a point inside a quadrilateral.

    ``u`` runs left->right (0..1), ``v`` top->bottom (0..1) relative to the
    ordered board corners ``(tl, tr, br, bl)``.
    """
    tl, tr, br, bl = order_points(corners)
    return ((1 - u) * (1 - v) * tl
            + u * (1 - v) * tr
            + u * v * br
            + (1 - u) * v * bl)


def square_polygon(corners: np.ndarray, file: int, rank_on_screen: int) -> np.ndarray:
    """Return the four screen-corner points of one board square.

    ``rank_on_screen`` is 0 for the top row of the screen image (not the chess
    rank). This works for arbitrary perspective/rotation because it uses
    bilinear interpolation across the detected quadrilateral.
    """
    u0, u1 = file / 8.0, (file + 1) / 8.0
    v0, v1 = rank_on_screen / 8.0, (rank_on_screen + 1) / 8.0
    return np.array([
        bilinear_point(corners, u0, v0),
        bilinear_point(corners, u1, v0),
        bilinear_point(corners, u1, v1),
        bilinear_point(corners, u0, v1),
    ], dtype=np.float32)


def square_center(corners: np.ndarray, file: int, rank_on_screen: int) -> np.ndarray:
    polygon = square_polygon(corners, file, rank_on_screen)
    return polygon.mean(axis=0)


def bounding_rect(corners: np.ndarray) -> tuple[int, int, int, int]:
    """Axis-aligned bounding rectangle ``(x, y, w, h)`` of a quadrilateral."""
    pts = np.asarray(corners, dtype=np.float32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)


def quad_area(corners: np.ndarray) -> float:
    """Shoelace area of a quadrilateral."""
    pts = order_points(corners)
    return float(abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
                     - np.dot(pts[:, 1], np.roll(pts[:, 0], 1))) / 2.0)
