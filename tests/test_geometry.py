"""Tests for perspective geometry helpers."""
import numpy as np

from chess_ai.vision.geometry import (
    bilinear_point,
    bounding_rect,
    order_points,
    quad_area,
    square_center,
    square_polygon,
)


def test_order_points_sorts_corners_regardless_of_input_order():
    shuffled = np.array([
        [10, 100],  # bl
        [100, 100],  # br
        [10, 10],    # tl
        [100, 10],   # tr
    ], dtype=np.float32)
    tl, tr, br, bl = order_points(shuffled)
    assert tl.tolist() == [10, 10]
    assert tr.tolist() == [100, 10]
    assert br.tolist() == [100, 100]
    assert bl.tolist() == [10, 100]


def test_bilinear_corners_and_center():
    corners = np.array([[0, 0], [80, 0], [80, 80], [0, 80]], dtype=np.float32)
    assert bilinear_point(corners, 0, 0).tolist() == [0, 0]
    assert bilinear_point(corners, 1, 1).tolist() == [80, 80]
    center = bilinear_point(corners, 0.5, 0.5)
    assert center.tolist() == [40, 40]


def test_square_polygon_covers_64_cells_without_overlap_gaps():
    corners = np.array([[0, 0], [800, 0], [800, 800], [0, 800]], dtype=np.float32)
    first = square_polygon(corners, 0, 0)
    assert first[0].tolist() == [0, 0]
    assert first[2].tolist() == [100, 100]
    last = square_polygon(corners, 7, 7)
    assert last[2].tolist() == [800, 800]
    center = square_center(corners, 4, 3)
    assert center.tolist() == [450.0, 350.0]


def test_bounding_rect_and_area():
    quad = np.array([[10, 20], [110, 20], [110, 120], [10, 120]], dtype=np.float32)
    assert bounding_rect(quad) == (10, 20, 100, 100)
    assert quad_area(quad) == 10000.0
