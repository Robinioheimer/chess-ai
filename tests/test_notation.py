"""Tests for SAN generation and evaluation formatting."""
import chess
import pytest

from chess_ai.core.notation import (
    describe_score_balance,
    format_eval,
    format_opponent_eval,
    format_win_percent,
    san_for_uci,
)


def test_san_basic_moves():
    fen = chess.STARTING_FEN
    assert san_for_uci(fen, "e2e4") == "e4"
    assert san_for_uci(fen, "g1f3") == "Nf3"


def test_san_capture_and_check():
    # Scholar's mate-ish position where Qxf7 is check.
    board = chess.Board()
    for san in ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"):
        board.push_san(san)
    fen = board.fen()
    assert san_for_uci(fen, "h5f7") == "Qxf7#"


def test_san_castling_and_promotion():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert san_for_uci(board.fen(), "e1g1") == "O-O"
    assert san_for_uci(board.fen(), "e1c1") == "O-O-O"
    promo = chess.Board("8/P7/8/8/8/8/8/4k2K w - - 0 1")
    assert san_for_uci(promo.fen(), "a7a8q") == "a8=Q"


def test_san_illegal_raises():
    with pytest.raises(ValueError):
        san_for_uci(chess.STARTING_FEN, "e2e5")


def test_format_eval_variants():
    assert format_eval(123, None) == "+1.23"
    assert format_eval(-40, None) == "-0.40"
    assert format_eval(None, 5) == "M5"
    assert format_eval(None, -3) == "-M3"
    assert format_eval(None, None) == "--"


def test_opponent_perspective_formatting():
    assert format_opponent_eval(120, None) == "-1.20"
    assert format_opponent_eval(None, 4) == "-M4"


def test_win_percent_string_and_balance_labels():
    assert format_win_percent(0, None) == "50.0%"
    text = format_win_percent(None, 1)
    assert text.endswith("%") and float(text.rstrip("%")) > 99
    assert describe_score_balance(None, 2) == "forced mate for us"
    assert describe_score_balance(150, None) == "clearly better"
    assert describe_score_balance(-500, None) == "losing"
