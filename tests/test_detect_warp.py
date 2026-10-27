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
