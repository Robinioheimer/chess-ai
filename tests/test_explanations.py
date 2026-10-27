"""Tests for auto-generated move explanations."""
import chess

from chess_ai.core.explanations import explain_move


def test_explains_capture():
    # Position after 1.e4 d5 - White can capture the d5 pawn.
    board = chess.Board()
    board.push_san("e4")
    board.push_san("d5")
    text = explain_move(board.fen(), "e4d5", 120, None, 20, None)
    assert "pawn" in text.lower()


def test_explains_castling():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    text = explain_move(board.fen(), "e1g1", 30, None, 10, None)
    assert "castle" in text.lower()


def test_explains_promotion():
    board = chess.Board("8/P7/8/8/8/8/8/4k2K w - - 0 1")
    text = explain_move(board.fen(), "a7a8q", 900, None, 100, None)
    assert "promot" in text.lower()


def test_explains_mate():
    board = chess.Board()
    for san in ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"):
        board.push_san(san)
    text = explain_move(board.fen(), "h5f7", None, 1, None, None)
    assert "mate" in text.lower()


def test_explains_tactical_gap():
    # Top move much better than second: should mention the gap.
    board = chess.Board()
    text = explain_move(board.fen(), "e2e4", 200, None, second_cp=0)
    assert text.endswith(".")
    assert "better" in text.lower() or "advantage" in text.lower()


def test_illegal_move_handled_gracefully():
    text = explain_move(chess.STARTING_FEN, "e2e5", 0, None)
    assert "unavailable" in text.lower()
