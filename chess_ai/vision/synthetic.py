"""Synthetic chessboard and piece rendering.

Two purposes:

1. Generate the built-in template library used by the piece recognizer.
2. Generate deterministic test fixtures with known ground truth so board
   detection and recognition can be validated without real screenshots.

Every piece is drawn as a single connected silhouette mask that is coloured
afterwards. This guarantees clean contours (no internal gaps between stacked
shapes) for both template matching and heuristic shape analysis.
"""
from __future__ import annotations

import cv2
import numpy as np

# BGR colours.
WHITE_PIECE_FILL = (238, 238, 238)
WHITE_PIECE_EDGE = (28, 28, 28)
BLACK_PIECE_FILL = (42, 42, 42)
BLACK_PIECE_EDGE = (225, 225, 225)
DEFAULT_LIGHT_SQUARE = (181, 217, 240)   # BGR warm light
DEFAULT_DARK_SQUARE = (99, 136, 181)     # BGR warm dark


def _palette(symbol: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if symbol.isupper():
        return WHITE_PIECE_FILL, WHITE_PIECE_EDGE
    return BLACK_PIECE_FILL, BLACK_PIECE_EDGE


# --------------------------------------------------------------------- shapes
def _draw_pawn(mask: np.ndarray) -> None:
    cv2.ellipse(mask, (50, 84), (24, 8), 0, 0, 360, 255, -1, cv2.LINE_AA)       # base
    body = np.array([(41, 50), (59, 50), (64, 78), (36, 78)], np.int32)
    cv2.fillPoly(mask, [body], 255, cv2.LINE_AA)
    cv2.circle(mask, (50, 38), 14, 255, -1, cv2.LINE_AA)                        # head


def _draw_rook(mask: np.ndarray) -> None:
    cv2.ellipse(mask, (50, 86), (25, 8), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.rectangle(mask, (35, 44), (65, 84), 255, -1)                            # wide straight shaft
    # Solid crenellated slab: shallow notches keep the top one connected mass.
    cv2.rectangle(mask, (30, 28), (70, 48), 255, -1)
    for notch_x in (40, 50, 60):
        cv2.rectangle(mask, (notch_x - 2, 28), (notch_x + 2, 34), 0, -1)


def _draw_knight(mask: np.ndarray) -> None:
    cv2.ellipse(mask, (50, 85), (24, 7), 0, 0, 360, 255, -1, cv2.LINE_AA)
    head = np.array([
        (33, 84), (36, 60), (44, 44), (56, 30), (68, 20), (75, 24),
        (67, 31), (74, 40), (63, 46), (59, 60), (64, 84),
    ], np.int32)
    cv2.fillPoly(mask, [head], 255, cv2.LINE_AA)


def _draw_bishop(mask: np.ndarray) -> None:
    cv2.ellipse(mask, (50, 85), (23, 7), 0, 0, 360, 255, -1, cv2.LINE_AA)
    # Steep, narrow mitre so the upper-section width stays clearly below the
    # pawn/rook ranges even after downscaling.
    body = np.array([(37, 82), (45, 58), (50, 34), (55, 58), (63, 82)], np.int32)
    cv2.fillPoly(mask, [body], 255, cv2.LINE_AA)
    cv2.circle(mask, (50, 27), 5, 255, -1, cv2.LINE_AA)                         # mitre ball


def _draw_queen(mask: np.ndarray) -> None:
    cv2.ellipse(mask, (50, 86), (25, 8), 0, 0, 360, 255, -1, cv2.LINE_AA)
    # Single goblet polygon: flared crown, narrow waist, flared base. Every
    # section stays >=10 px wide so the silhouette never fragments.
    goblet = np.array([
        (30, 30), (70, 30), (58, 52), (56, 58), (58, 72), (66, 82),
        (34, 82), (42, 72), (44, 58), (42, 52),
    ], np.int32)
    cv2.fillPoly(mask, [goblet], 255, cv2.LINE_AA)
    for tip_x in (33, 41, 50, 59, 67):
        triangle = np.array([
            (tip_x - 3, 32), (tip_x + 3, 32), (tip_x, 18),
        ], np.int32)
        cv2.fillPoly(mask, [triangle], 255, cv2.LINE_AA)
        cv2.circle(mask, (tip_x, 16), 3, 255, -1, cv2.LINE_AA)                  # crown points


def _draw_king(mask: np.ndarray) -> None:
    cv2.ellipse(mask, (50, 86), (25, 8), 0, 0, 360, 255, -1, cv2.LINE_AA)
    body = np.array([(38, 82), (43, 50), (57, 50), (62, 82)], np.int32)
    cv2.fillPoly(mask, [body], 255, cv2.LINE_AA)
    cv2.rectangle(mask, (40, 40), (60, 52), 255, -1)                            # crown
    # Extra-thick cross with a wide horizontal bar: the bar is the king's
    # signature and must survive aggressive downscaling.
    cv2.rectangle(mask, (45, 4), (55, 42), 255, -1)                             # cross vertical
    cv2.rectangle(mask, (33, 12), (67, 22), 255, -1)                            # cross horizontal


_PAINTERS = {
    "p": _draw_pawn,
    "r": _draw_rook,
    "n": _draw_knight,
    "b": _draw_bishop,
    "q": _draw_queen,
    "k": _draw_king,
}


def render_piece(symbol: str, size: int = 64) -> np.ndarray:
    """Render one piece on a transparent square.

    Returns a BGRA image where alpha marks piece pixels.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    _PAINTERS[symbol.lower()](mask)
    # Merge anti-aliased contacts and keep the single outer silhouette.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(solid, [biggest], -1, 255, thickness=cv2.FILLED)
    # Bishop's mitre slit is an intentional cut into the merged silhouette.
    if symbol.lower() == "b":
        cv2.line(solid, (50, 38), (47, 55), 0, thickness=3)

    fill, edge = _palette(symbol)
    # Dark outer halo (drop shadow) guarantees the silhouette is detectable on
    # light squares even for dark pieces whose light outline would otherwise
    # blend into the background.
    halo = cv2.dilate(solid, np.ones((5, 5), np.uint8), iterations=1)
    canvas = np.zeros((100, 100, 4), dtype=np.uint8)
    canvas[halo > 0, :3] = (24, 24, 24)
    canvas[halo > 0, 3] = 255
    canvas[solid > 0, :3] = fill
    canvas[solid > 0, 3] = 255
    # A thick contrasting outline keeps the silhouette detectable even when
    # the piece fill is close in colour to the square underneath.
    cv2.drawContours(canvas, contours, -1, (*edge, 255), 3, cv2.LINE_AA)
    if size != 100:
        canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
        canvas[:, :, 3] = np.where(canvas[:, :, 3] > 40, 255, 0).astype(np.uint8)
    return canvas


def render_board(
    grid: list[list[str | None]],
    square_px: int = 64,
    light_bgr=DEFAULT_LIGHT_SQUARE,
    dark_bgr=DEFAULT_DARK_SQUARE,
    perspective_white: bool = True,
    noise_sigma: float = 0.0,
    blur: float = 0.0,
) -> np.ndarray:
    """Render an 8x8 grid (board coordinates) into a BGR screenshot-like image.

    ``grid`` follows the board-coordinate convention (rank 1 first). The
    perspective flag mirrors the board exactly like the FEN grid helpers.
    """
    from ..core.fen_utils import board_grid_to_screen_grid

    screen = board_grid_to_screen_grid(grid, perspective_white)
    board_img = np.zeros((square_px * 8, square_px * 8, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            y0, x0 = row * square_px, col * square_px
            color = light_bgr if (row + col) % 2 == 0 else dark_bgr
            board_img[y0:y0 + square_px, x0:x0 + square_px] = color
            symbol = screen[row][col]
            if symbol is None:
                continue
            piece_size = int(square_px * 0.9)
            piece = render_piece(symbol, piece_size)
            offset = (square_px - piece_size) // 2
            roi = board_img[
                y0 + offset:y0 + offset + piece_size,
                x0 + offset:x0 + offset + piece_size,
            ]
            alpha = piece[:, :, 3:4].astype(np.float32) / 255.0
            roi[:] = (alpha * piece[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)

    if noise_sigma:
        noise = np.random.normal(0, noise_sigma, board_img.shape)
        board_img = np.clip(board_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if blur:
        board_img = cv2.GaussianBlur(board_img, (0, 0), blur)
    return board_img


def render_scene(
    grid: list[list[str | None]],
    square_px: int = 56,
    padding: int = 48,
    background_bgr=(32, 32, 32),
    angle_deg: float = 0.0,
    perspective_white: bool = True,
    noise_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed a rendered board in a larger desktop-like scene.

    Returns ``(scene, corners)`` where ``corners`` is the ground-truth board
    quadrilateral ``(tl, tr, br, bl)`` before/after rotation.
    """
    board_img = render_board(
        grid, square_px, perspective_white=perspective_white, noise_sigma=noise_sigma
    )
    side = square_px * 8
    scene = np.full((side + padding * 2, side + padding * 2, 3),
                    background_bgr, dtype=np.uint8)
    # Add a lighter "window" rectangle so the scene is not perfectly flat.
    cv2.rectangle(scene, (8, 8), (padding * 2 - 8, padding * 2 - 8),
                  (background_bgr[0] + 25, background_bgr[1] + 25,
                   background_bgr[2] + 25), -1)
    scene[padding:padding + side, padding:padding + side] = board_img
    corners = np.array([
        [padding, padding],
        [padding + side - 1, padding],
        [padding + side - 1, padding + side - 1],
        [padding, padding + side - 1],
    ], dtype=np.float32)

    if angle_deg:
        center = (scene.shape[1] / 2, scene.shape[0] / 2)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        scene = cv2.warpAffine(scene, matrix, (scene.shape[1], scene.shape[0]),
                               borderValue=background_bgr)
        ones = np.ones((4, 1), dtype=np.float32)
        corners = (matrix @ np.hstack([corners, ones]).T).T
    return scene, corners
