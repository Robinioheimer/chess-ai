"""Tests for board detection and perspective warping using synthetic scenes."""
import chess
import numpy as np

from chess_ai.core.fen_utils import board_to_grid
from chess_ai.vision.detect import detect_board
from chess_ai.vision.geometry import order_points
from chess_ai.vision.synthetic import render_scene
from chess_ai.vision.warp import split_squares, warp_board


def test_detect_board_in_plain_synthetic_scene():
    scene, truth = render_scene(board_to_grid(chess.Board()), square_px=56)
    detection = detect_board(scene)
    assert detection is not None
    detected = order_points(detection.corners)
    # Allow a few pixels of tolerance for edge-based contour placement.
    assert np.allclose(detected, truth, atol=6)


def test_detect_board_after_rotation():
    scene, truth = render_scene(board_to_grid(chess.Board()), square_px=56, angle_deg=8)
    detection = detect_board(scene)
    assert detection is not None
    detected = order_points(detection.corners)
    assert np.allclose(detected, truth, atol=8)


def test_detect_board_returns_none_for_empty_frame():
    frame = np.full((480, 640, 3), (20, 20, 20), dtype=np.uint8)
    assert detect_board(frame) is None


def test_warp_produces_square_board_with_64_cells():
    scene, _ = render_scene(board_to_grid(chess.Board()), square_px=56, angle_deg=5)
    detection = detect_board(scene)
    warped = warp_board(scene, detection.corners, size=480)
    assert warped.shape == (480, 480, 3)
    cells = split_squares(warped)
    assert cells.shape == (8, 8)
    # Cells are inset slightly to avoid boundary bleed; base size is 60 px.
    assert 50 <= cells[0, 0].shape[0] <= 60
    assert 50 <= cells[0, 0].shape[1] <= 60


def test_min_fraction_filters_small_contours():
    scene, _ = render_scene(board_to_grid(chess.Board()), square_px=40, padding=200)
    # With a huge surrounding scene the board occupies less than 8% of area.
    assert detect_board(scene, min_fraction=0.5) is None
    assert detect_board(scene, min_fraction=0.05) is not None


def test_detect_board_nested_inside_browser_window_and_wooden_frame():
    import cv2
    from chess_ai.vision.synthetic import render_board

    grid = board_to_grid(chess.Board())
    board_img = render_board(grid, square_px=56)
    scene = np.full((900, 1400, 3), (30, 34, 42), dtype=np.uint8)
    # Outer browser window + header + sidebar + wooden frame around the 8x8 board
    cv2.rectangle(scene, (50, 40), (1350, 860), (245, 245, 245), -1)
    cv2.rectangle(scene, (50, 40), (1350, 100), (60, 65, 75), -1)
    cv2.rectangle(scene, (900, 120), (1320, 820), (220, 220, 225), -1)
    cv2.rectangle(scene, (170, 150), (170 + 448 + 60, 150 + 448 + 60), (60, 90, 140), -1)
    scene[180:180 + 448, 200:200 + 448] = board_img
    truth = np.array([
        [200, 180], [200 + 447, 180], [200 + 447, 180 + 447], [200, 180 + 447],
    ], dtype=np.float32)

    detection = detect_board(scene)
    assert detection is not None
    assert np.allclose(order_points(detection.corners), truth, atol=6)


def test_detect_borderless_board_on_light_and_dark_backgrounds():
    from chess_ai.vision.synthetic import (
        DEFAULT_DARK_SQUARE,
        DEFAULT_LIGHT_SQUARE,
        render_board,
    )

    grid = board_to_grid(chess.Board())
    board_img = render_board(grid, square_px=56)
    truth = np.array([
        [220, 110], [220 + 447, 110], [220 + 447, 110 + 447], [220, 110 + 447],
    ], dtype=np.float32)

    for bg in (DEFAULT_LIGHT_SQUARE, DEFAULT_DARK_SQUARE):
        scene = np.full((700, 1000, 3), bg, dtype=np.uint8)
        scene[110:110 + 448, 220:220 + 448] = board_img
        detection = detect_board(scene)
        assert detection is not None
        assert np.allclose(order_points(detection.corners), truth, atol=6)
