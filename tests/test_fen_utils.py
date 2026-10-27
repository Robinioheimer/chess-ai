"""Tests for FEN construction, grid transforms and validation."""
import chess
import pytest

from chess_ai.core.fen_utils import (
    board_grid_to_screen_grid,
    board_to_grid,
    build_fen,
    count_material,
    grid_signature,
    grid_to_board,
    legal_move_count,
    screen_grid_to_board_grid,
    square_name,
    validate_fen,
)

START_FEN = chess.STARTING_FEN


def _start_grid():
    board = chess.Board()
    return board_to_grid(board)


def test_roundtrip_starting_position():
    grid = _start_grid()
    board = grid_to_board(grid)
    assert board.board_fen() == chess.Board().board_fen()
    fen = build_fen(grid, "w")
    re_board = chess.Board(fen)
    assert re_board.board_fen() == chess.Board().board_fen()
    assert re_board.turn == chess.WHITE


def test_screen_white_perspective_maps_home_rank_to_bottom():
    grid = _start_grid()
    screen = board_grid_to_screen_grid(grid, perspective_white=True)
    # Top screen row is rank 8: black rooks on a8/h8.
    assert screen[0][0] == "r"
    assert screen[0][7] == "r"
    # Bottom screen row is rank 1: white rooks.
    assert screen[7][0] == "R"
    assert screen[7][7] == "R"
    back = screen_grid_to_board_grid(screen, perspective_white=True)
    assert back == grid


def test_screen_black_perspective_is_double_flipped():
    grid = _start_grid()
    screen = board_grid_to_screen_grid(grid, perspective_white=False)
    # With Black at the bottom, the top row is rank 1 with files mirrored.
    assert screen[0][7] == "R"  # a1 rook appears on the top-right
    assert screen[7][7] == "r"  # a8 rook appears on the bottom-right
    assert screen[7][0] == "r"  # h8 rook appears on the bottom-left
    back = screen_grid_to_board_grid(screen, perspective_white=False)
    assert back == grid


def test_build_fen_side_to_move():
    grid = _start_grid()
    assert build_fen(grid, "b").split()[1] == "b"
    with pytest.raises(ValueError):
        build_fen(grid, "x")


def test_validate_fen_accepts_standard_and_rejects_bad():
    ok, _ = validate_fen(START_FEN)
    assert ok
    ok, msg = validate_fen("8/8/8/8/8/8/8/8 w - - 0 1")
    assert not ok and "king" in msg
    ok, msg = validate_fen("not-a-fen")
    assert not ok and msg


def test_square_name_and_material():
    assert square_name(0, 0) == "a1"
    assert square_name(7, 7) == "h8"
    board = chess.Board()
    assert count_material(board, chess.WHITE) == 4000  # 8P+2N+2B+2R+Q
    assert legal_move_count(board) == 20


def test_grid_signature_stable_and_distinct():
    grid = _start_grid()
    assert grid_signature(grid) == grid_signature(_start_grid())
    board = chess.Board()
    board.push_san("e4")
    other = board_to_grid(board)
    assert grid_signature(other) != grid_signature(grid)
