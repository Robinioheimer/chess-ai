"""Move notation and evaluation formatting."""
from __future__ import annotations

import chess

from .probabilities import opponent_perspective, score_to_win_probability


def san_for_uci(fen: str, uci_move: str) -> str:
    """Return SAN notation (e.g. ``Nf3``, ``Qxd5+``) for a UCI move in a FEN.

    Raises :class:`ValueError` if the move is illegal or malformed in the
    position.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move {uci_move!r} for position {fen!r}")
    return board.san(move)


def san_for_move(board: chess.Board, move: chess.Move) -> str:
    """SAN for a move without mutating the caller's board."""
    return board.san(move)


def format_eval(cp: float | None, mate: int | None) -> str:
    """Format an engine score the way chess UIs do.

    * Centipawns are shown in pawns with a sign, e.g. ``+1.23`` / ``-0.40``.
    * Mates are shown as ``M5`` (mate in 5) or ``-M3``.
    * A missing score renders as ``--``.
    """
    if mate is not None:
        return f"M{mate}" if mate > 0 else f"-M{abs(mate)}"
    if cp is None:
        return "--"
    pawns = cp / 100.0
    return f"{pawns:+.2f}"


def format_opponent_eval(cp: float | None, mate: int | None) -> str:
    """Evaluation string from the opponent's perspective."""
    opp_cp, opp_mate = opponent_perspective(cp, mate)
    return format_eval(opp_cp, opp_mate)


def format_win_percent(cp: float | None, mate: int | None) -> str:
    """Winning-probability estimate as a labelled percentage string."""
    probability = score_to_win_probability(cp, mate)
    return f"{probability * 100:.1f}%"


def describe_score_balance(cp: float | None, mate: int | None) -> str:
    """Short qualitative label of who is better (side to move perspective)."""
    if mate is not None:
        return "forced mate for us" if mate > 0 else "we get mated"
    if cp is None:
        return "no evaluation yet"
    if cp >= 300:
        return "winning"
    if cp >= 100:
        return "clearly better"
    if cp >= 40:
        return "slightly better"
    if cp > -40:
        return "roughly equal"
    if cp > -100:
        return "slightly worse"
    if cp > -300:
        return "clearly worse"
    return "losing"
