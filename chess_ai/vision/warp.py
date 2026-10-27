"""Perspective warping and square-cell extraction."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import order_points


def warp_board(image: np.ndarray, corners: np.ndarray, size: int = 480) -> np.ndarray:
    """Warp the detected board quadrilateral to a square top-down image."""
    src = order_points(corners)
    native_side = int(round(max(
        float(np.linalg.norm(src[1] - src[0])),
        float(np.linalg.norm(src[2] - src[1])),
        float(np.linalg.norm(src[2] - src[3])),
        float(np.linalg.norm(src[3] - src[0])),
    ))) + 1
    warp_size = min(max(size, native_side), size * 2)
    dst = np.array([
        [0, 0], [warp_size - 1, 0], [warp_size - 1, warp_size - 1], [0, warp_size - 1]
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, matrix, (warp_size, warp_size), flags=cv2.INTER_LINEAR)
    if warp_size != size:
        warped = cv2.resize(warped, (size, size), interpolation=cv2.INTER_AREA)
    return warped


def split_squares(warped: np.ndarray, inset: float = 0.03) -> np.ndarray:
    """Split a warped board into an ``(8, 8)`` array of square images.

    Index ``[row][col]`` follows screen order: row 0 is the top row. Each cell
    is inset by ``inset`` (fraction of the cell size) so slight corner
    detection errors cannot bleed neighbouring squares or the desktop
    background into edge cells.
    """
    height, width = warped.shape[:2]
    cell_h, cell_w = height // 8, width // 8
    inset_y, inset_x = int(cell_h * inset), int(cell_w * inset)
    cells = np.empty((8, 8), dtype=object)
    for row in range(8):
        for col in range(8):
            y0 = row * cell_h + inset_y
            x0 = col * cell_w + inset_x
            cells[row, col] = warped[
                y0: (row + 1) * cell_h - inset_y,
                x0: (col + 1) * cell_w - inset_x,
            ]
    return cells


def inner_cell(cell: np.ndarray, margin: float = 0.12) -> np.ndarray:
    """Crop away grid-line borders around a square."""
    h, w = cell.shape[:2]
    my, mx = int(h * margin), int(w * margin)
    return cell[my:h - my, mx:w - mx]
