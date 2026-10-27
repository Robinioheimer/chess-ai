"""Tests for position tracking, de-duplication and side-to-move handling."""
import chess

from chess_ai.core.tracker import PositionTracker


def test_initial_observation():
    tracker = PositionTracker()
    update = tracker.observe_placement_fen(chess.STARTING_FEN)
    assert update.accepted and update.changed
    assert tracker.board.turn == chess.WHITE


def test_duplicate_reading_suppressed():
    tracker = PositionTracker()
    tracker.observe_placement_fen(chess.STARTING_FEN)
    update = tracker.observe_placement_fen(chess.STARTING_FEN)
    assert update.accepted and not update.changed
    assert "same position" in update.reason.lower()


def test_move_flips_side_to_move_and_tracks_uci():
    tracker = PositionTracker()
    tracker.observe_placement_fen(chess.STARTING_FEN)
    board = chess.Board()
    board.push_san("e4")
    update = tracker.observe_placement_fen(board.fen())
    assert update.changed and update.move_uci == "e2e4"
    assert tracker.board.turn == chess.BLACK


def test_illegal_placement_rejected_and_state_kept():
    tracker = PositionTracker()
    tracker.observe_placement_fen(chess.STARTING_FEN)
    update = tracker.observe_placement_fen("8/8/8/8/8/8/8/8 w - - 0 1")
    assert not update.accepted
    assert tracker.fen is not None
    assert tracker.board.board_fen() == chess.Board().board_fen()


def test_manual_fen_override():
    tracker = PositionTracker()
    custom = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    update = tracker.set_manual_fen(custom)
    assert update.accepted and update.changed
    assert tracker.board.turn == chess.BLACK


def test_manual_side_override():
    tracker = PositionTracker()
    tracker.observe_placement_fen(chess.STARTING_FEN)
    tracker.set_side_to_move(chess.BLACK)
    assert tracker.board.turn == chess.BLACK


def test_reset_clears_state():
    tracker = PositionTracker()
    tracker.observe_placement_fen(chess.STARTING_FEN)
    tracker.reset()
    assert not tracker.has_position


def test_placement_only_fen_is_normalized():
    tracker = PositionTracker()
    update = tracker.observe_placement_fen(chess.Board().board_fen())
    assert update.accepted


def test_manual_fen_not_overwritten_by_unchanged_raw_vision_fen():
    tracker = PositionTracker()
    raw_fen = chess.STARTING_FEN
    tracker.observe_placement_fen(raw_fen)

    corrected = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    update = tracker.set_manual_fen(corrected, raw_vision_fen=raw_fen)
    assert update.accepted and update.changed
    assert tracker.board.turn == chess.BLACK

    # Subsequent tick with the same pre-correction raw vision FEN must NOT overwrite manual FEN
    dup = tracker.observe_placement_fen(raw_fen)
    assert dup.accepted and not dup.changed
    assert tracker.board.board_fen() == chess.Board(corrected).board_fen()
    assert tracker.board.turn == chess.BLACK

    # Once a genuinely new position appears on screen, normal tracking resumes
    next_board = chess.Board(corrected)
    next_board.push_san("e5")
    moved = tracker.observe_placement_fen(next_board.fen())
    assert moved.accepted and moved.changed
    assert tracker.board.board_fen() == next_board.board_fen()
    assert tracker.board.turn == chess.WHITE


def test_observe_placement_flips_turn_when_opponent_in_check():
    tracker = PositionTracker(side_to_move=chess.WHITE)
    # Position where Black's king on e8 is in check from White's queen on e2:
    # it must be Black's turn so was_into_check() is False.
    fen_black_in_check = "4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1"
    update = tracker.observe_placement_fen(fen_black_in_check)
    assert update.accepted
    assert not update.board.was_into_check()
    assert update.board.turn == chess.BLACK

