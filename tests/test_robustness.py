"""Recognition robustness across board sizes, colour schemes and positions."""
import chess
import pytest

from chess_ai.core.fen_utils import board_to_grid
from chess_ai.vision.pipeline import RecognitionPipeline
from chess_ai.vision.synthetic import render_board


@pytest.mark.parametrize("square_px", [48, 56, 72])
def test_recognition_across_board_sizes(square_px):
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    frame = render_board(board_to_grid(board), square_px=square_px)
    pipeline = RecognitionPipeline(stable_readings=1, manual_only=False)
    result = pipeline.process_frame(frame)
    assert result.success
    assert chess.Board(result.fen).board_fen() == board.board_fen()


@pytest.mark.parametrize("light,dark", [
    ((240, 217, 181), (181, 136, 99)),    # wooden
    ((235, 235, 235), (90, 90, 90)),      # grey
    ((200, 230, 201), (67, 160, 71)),     # green
    ((220, 208, 255), (106, 76, 147)),    # purple
])
def test_recognition_across_square_colors(light, dark):
    frame = render_board(
        board_to_grid(chess.Board()), square_px=60,
        light_bgr=light, dark_bgr=dark,
    )
    pipeline = RecognitionPipeline(stable_readings=1, manual_only=False)
    result = pipeline.process_frame(frame)
    assert result.success
    assert chess.Board(result.fen).board_fen() == chess.Board().board_fen()


def test_arbitrary_position_with_promoted_queen_and_sparse_board():
    # Tactical position: promoted queens and only a handful of pieces.
    fen = "8/2Q5/8/8/8/5q2/1k4P1/4K3 w - - 0 1"
    board = chess.Board(fen)
    frame = render_board(board_to_grid(board), square_px=64)
    pipeline = RecognitionPipeline(stable_readings=1, manual_only=False)
    result = pipeline.process_frame(frame)
    assert result.success
    assert chess.Board(result.fen).board_fen() == board.board_fen()


def test_noise_and_blur_do_not_break_starting_position():
    frame = render_board(
        board_to_grid(chess.Board()), square_px=64,
        noise_sigma=6.0, blur=0.6,
    )
    pipeline = RecognitionPipeline(stable_readings=1, manual_only=False)
    result = pipeline.process_frame(frame)
    assert result.success
    assert chess.Board(result.fen).board_fen() == chess.Board().board_fen()


def test_black_perspective_after_moves():
    board = chess.Board()
    for san in ("e4", "c5", "Nf3", "d6"):
        board.push_san(san)
    frame = render_board(board_to_grid(board), square_px=60,
                         perspective_white=False)
    pipeline = RecognitionPipeline(stable_readings=1, manual_only=False)
    pipeline.set_manual_perspective(False)
    result = pipeline.process_frame(frame)
    assert result.success
    assert result.perspective_white is False
    assert chess.Board(result.fen).board_fen() == board.board_fen()
