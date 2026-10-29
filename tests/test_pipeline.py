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


def test_auto_infer_black_perspective_without_manual_override():
    grid, original = _grid_after_moves(["e4", "c5", "Nf3", "d6"])
    frame = render_board(grid, square_px=60, perspective_white=False)
    pipeline = RecognitionPipeline(stable_readings=1)
    result = pipeline.process_frame(frame)
    assert result.success
    assert result.perspective_white is False
    assert chess.Board(result.fen).board_fen() == original.board_fen()


def test_chess_constraints_repair_back_rank_pawns_and_missing_kings():
    from chess_ai.vision.features import CellFeatures

    pipeline = RecognitionPipeline(stable_readings=1)
    grid = [[None] * 8 for _ in range(8)]
    # Back-rank pawn on row 0 col 0, two queens and zero kings
    grid[0][0] = "p"
    grid[0][4] = "q"
    grid[7][0] = "P"
    grid[7][4] = "Q"
    confidences = [[0.8] * 8 for _ in range(8)]
    features_grid = [
        [CellFeatures(occupied=False) for _ in range(8)]
        for _ in range(8)
    ]
    features_grid[0][0] = CellFeatures(occupied=True, h_ratio=0.78, top_width_ratio=0.60, width_at_15=0.38)
    features_grid[0][4] = CellFeatures(occupied=True, h_ratio=0.92, center_protrusion=0.08, symmetry=0.96)
    features_grid[7][0] = CellFeatures(occupied=True, h_ratio=0.78, top_width_ratio=0.60, width_at_15=0.38)
    features_grid[7][4] = CellFeatures(occupied=True, h_ratio=0.92, center_protrusion=0.08, symmetry=0.96)

    pipeline._enforce_chess_constraints(grid, confidences, features_grid)
    assert grid[0][0] == "r"
    assert grid[7][0] == "R"
    assert grid[0][4] == "k"
    assert grid[7][4] == "K"


def test_manual_correction_persists_until_square_changes():
    from chess_ai.core.fen_utils import board_grid_to_screen_grid

    grid = board_to_grid(chess.Board())
    frame = render_board(grid, square_px=60)
    pipeline = RecognitionPipeline(stable_readings=1)
    initial = pipeline.process_stable(frame)
    assert initial.success

    # Simulate user correcting e2 -> empty and e4 -> White Pawn on the same frame
    e4_board = chess.Board()
    e4_board.push_san("e4")
    corrected_screen = board_grid_to_screen_grid(board_to_grid(e4_board), True)
    pipeline.apply_manual_correction(corrected_screen, True, learn_templates=False)

    # Next tick on the unchanged frame preserves the user's manual correction
    again = pipeline.process_stable(frame)
    assert chess.Board(again.fen).board_fen() == e4_board.board_fen()


def test_manual_corners_lock_and_clear():
    scene, truth = render_scene(board_to_grid(chess.Board()), square_px=56, padding=60)
    pipeline = RecognitionPipeline(stable_readings=1)
    pipeline.set_manual_corners(truth)
    result = pipeline.process_frame(scene)
    assert result.success
    assert result.detection.method == "manual"
    pipeline.clear_manual_corners()
    assert pipeline.manual_corners is None


def test_auto_calibration_on_early_opening_position():
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6"):
        board.push_san(san)
    scene, _ = render_scene(board_to_grid(board), square_px=60, padding=60)
    pipeline = RecognitionPipeline(stable_readings=1)
    assert not pipeline.recognizer.calibrated
    result = pipeline.process_stable(scene)
    assert result.success
    assert pipeline.recognizer.calibrated
    assert chess.Board(result.fen).board_fen() == board.board_fen()


def test_move_continuity_tracks_sequence_of_moves():
    board = chess.Board()
    pipeline = RecognitionPipeline(stable_readings=1)
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O"):
        board.push_san(san)
        scene, _ = render_scene(board_to_grid(board), square_px=60, padding=60)
        result = pipeline.process_stable(scene)
        assert result.success
        assert chess.Board(result.fen).board_fen() == board.board_fen()


def test_corner_coordinate_label_does_not_trigger_occupancy():
    import cv2
    import numpy as np
    from chess_ai.vision.features import extract_cell_features

    cell = np.full((64, 64, 3), (235, 236, 208), dtype=np.uint8)
    # Draw a dark rank/file label '8' in the top-left corner of an empty square.
    cv2.putText(cell, "8", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 110, 70), 1)
    features = extract_cell_features(cell)
    assert not features.occupied

def test_frame_recognition_does_not_overwrite_last_accepted_grid():
    """Move continuity must compare a new frame with the prior accepted position."""
    start_grid = board_to_grid(chess.Board())
    start_frame = render_board(start_grid, square_px=60)
    pipeline = RecognitionPipeline(stable_readings=2)
    first = pipeline.process_stable(start_frame)
    assert first.success
    accepted_grid = [row[:] for row in pipeline._last_grid]

    moved = chess.Board()
    moved.push_san("e4")
    moved_frame = render_board(board_to_grid(moved), square_px=60)
    raw = pipeline.process_frame(moved_frame)
    assert raw.success
    # A raw frame is only a candidate; the accepted baseline stays unchanged.
    assert pipeline._last_grid == accepted_grid

    pending = pipeline.process_stable(moved_frame)
    assert pending.success
    assert chess.Board(pending.fen).board_fen() == chess.Board().board_fen()
    accepted = pipeline.process_stable(moved_frame)
    assert accepted.success
    assert chess.Board(accepted.fen).board_fen() == moved.board_fen()

def test_auto_calibration_retries_after_non_opening_position():
    pipeline = RecognitionPipeline(stable_readings=1)
    empty_grid = [[None] * 8 for _ in range(8)]
    cells = np.empty((8, 8), dtype=object)

    assert pipeline._try_auto_calibrate_opening(empty_grid, cells, 110.0) is None
    assert pipeline._calibration_attempted is False
