"""End-to-end recognition pipeline tests on rendered boards."""
import chess
import numpy as np

from chess_ai.core.fen_utils import board_to_grid
from chess_ai.vision.pipeline import RecognitionPipeline, infer_perspective
from chess_ai.vision.synthetic import render_board, render_scene


def _grid_after_moves(moves: list[str]):
    board = chess.Board()
    for san in moves:
        board.push_san(san)
    return board_to_grid(board), board


def test_starting_position_full_recognition():
    grid = board_to_grid(chess.Board())
    frame = render_board(grid, square_px=60)
    pipeline = RecognitionPipeline(stable_readings=1)
    result = pipeline.process_frame(frame)
    assert result.success
    assert result.perspective_white is True
    recognized = chess.Board(result.fen)
    assert recognized.board_fen() == chess.Board().board_fen()
    assert result.mean_confidence > 0.5


def test_midgame_position_full_recognition():
    grid, original = _grid_after_moves(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])
    frame = render_board(grid, square_px=60)
    pipeline = RecognitionPipeline(stable_readings=1)
    result = pipeline.process_frame(frame)
    assert result.success
    recognized = chess.Board(result.fen)
    assert recognized.board_fen() == original.board_fen()


def test_black_perspective_board():
    grid = board_to_grid(chess.Board())
    frame = render_board(grid, square_px=60, perspective_white=False)
    pipeline = RecognitionPipeline(stable_readings=1)
    pipeline.set_manual_perspective(False)
    result = pipeline.process_frame(frame)
    assert result.success
    recognized = chess.Board(result.fen)
    assert recognized.board_fen() == chess.Board().board_fen()
    assert result.perspective_white is False


def test_no_board_reports_clear_error():
    frame = np.full((400, 400, 3), (15, 15, 15), dtype=np.uint8)
    pipeline = RecognitionPipeline(stable_readings=1)
    result = pipeline.process_frame(frame)
    assert not result.success
    assert "No chessboard" in result.error


def test_stability_filter_requires_repeated_readings():
    grid = board_to_grid(chess.Board())
    frame = render_board(grid, square_px=60)
    pipeline = RecognitionPipeline(stable_readings=2)

    first = pipeline.process_stable(frame)
    assert first.success  # first reading is accepted immediately

    # A single noisy/different frame must not replace the accepted grid.
    noisy_grid, _ = _grid_after_moves(["e4"])
    noisy_frame = render_board(noisy_grid, square_px=60)
    second = pipeline.process_stable(noisy_frame)
    assert second.success
    assert chess.Board(second.fen).board_fen() == chess.Board().board_fen()

    # The same new position repeated enough times is accepted.
    third = pipeline.process_stable(noisy_frame)
    e4_board = chess.Board()
    e4_board.push_san("e4")
    assert chess.Board(third.fen).board_fen() == e4_board.board_fen()


def test_duplicate_frames_keep_same_fen():
    grid = board_to_grid(chess.Board())
    frame = render_board(grid, square_px=60)
    pipeline = RecognitionPipeline(stable_readings=2)
    pipeline.process_stable(frame)
    again = pipeline.process_stable(frame)
    assert again.success
    assert chess.Board(again.fen).board_fen() == chess.Board().board_fen()


def test_pipeline_detects_board_inside_scene():
    grid = board_to_grid(chess.Board())
    scene, _ = render_scene(grid, square_px=56, padding=60)
    pipeline = RecognitionPipeline(stable_readings=1)
    result = pipeline.process_frame(scene)
    assert result.success
    assert result.detection is not None
    recognized = chess.Board(result.fen)
    assert recognized.board_fen() == chess.Board().board_fen()


def test_infer_perspective_starting_position():
    assert infer_perspective(board_to_grid(chess.Board())) is True


def test_infer_perspective_manual_override():
    grid = board_to_grid(chess.Board())
    frame = render_board(grid, square_px=60, perspective_white=False)
    pipeline = RecognitionPipeline(stable_readings=1)
    pipeline.set_manual_perspective(False)
    result = pipeline.process_frame(frame)
    assert result.perspective_white is False
    pipeline.clear_manual_perspective()
    assert pipeline.manual_perspective is None


def test_calibration_from_starting_position(tmp_path):
    grid = board_to_grid(chess.Board())
    frame = render_board(grid, square_px=64)
    pipeline = RecognitionPipeline(stable_readings=1)
    count = pipeline.calibrate_from_starting_position(frame)
    assert count == 32
    assert pipeline.recognizer.calibrated
