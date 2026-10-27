"""Rule-based, auto-generated move explanations.

The engine only returns scores and principal variations; it does not explain
*why* a move is good. This module derives short, honest explanations from
chess facts (captures, checks, castling, promotion, evaluation gaps). The UI
labels these as auto-generated heuristics - they are tutoring hints, never
guarantees about the game outcome.
"""
from __future__ import annotations

import chess

from .fen_utils import PIECE_VALUES

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def _material_phrase(cp_delta: float) -> str:
    pawns = abs(cp_delta) / 100.0
    if pawns >= 9:
        return "about a queen"
    if pawns >= 5:
        return "about a rook"
    if pawns >= 3:
        return "about a minor piece"
    if pawns >= 1:
        return f"{pawns:.1f} pawns"
    return "a small amount of material"


def explain_move(
    fen: str,
    uci_move: str,
    cp: float | None,
    mate: int | None,
    second_cp: float | None = None,
    second_mate: int | None = None,
) -> str:
    """Produce a short explanation for a candidate move.

    Parameters
    ----------
    fen:
        Position before the move.
    uci_move:
        UCI notation of the candidate move.
    cp / mate:
        Engine score after the move, side-to-move perspective.
    second_cp / second_mate:
        Score of the second-best candidate, used to quantify how much stronger
        the top move is.
    """
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)
    if move not in board.legal_moves:
        return "Legal evaluation unavailable for this move."

    san = board.san(move)
    moving_piece = board.piece_at(move.from_square)
    captured = board.piece_at(move.to_square)
    # En-passant captures land on an empty square.
    is_en_passant = board.is_en_passant(move)
    if is_en_passant:
        captured = chess.Piece(chess.PAWN, not board.turn)
    in_check_before = board.is_check()

    board.push(move)
    gives_check = board.is_check()
    is_checkmate = board.is_checkmate()
    ply = board.ply()
    board.pop()

    reasons: list[str] = []

    if is_checkmate or (mate is not None and 0 < mate <= 1):
        reasons.append("Delivers checkmate" if is_checkmate else "Forces checkmate")
    elif mate is not None and mate > 0:
        reasons.append(f"Sets up a forced mate in {mate}")
    elif mate is not None and mate < 0:
        reasons.append("Best defence in a mating attack")

    if captured is not None and "checkmate" not in " ".join(reasons):
        value = PIECE_VALUES[captured.piece_type]
        if is_en_passant:
            reasons.append("Wins a pawn by en passant")
        elif value >= 500:
            reasons.append(f"Captures the enemy {PIECE_NAMES[captured.piece_type]}")
        else:
            reasons.append(f"Wins the {PIECE_NAMES[captured.piece_type]}")

    if move.promotion is not None:
        promo_name = PIECE_NAMES[move.promotion]
        reasons.append(f"Promotes to a {promo_name}")

    if san in ("O-O", "O-O-O") and not is_checkmate:
        reasons.append("Castles to improve king safety and connect the rooks")

    if gives_check and not is_checkmate:
        reasons.append("Gives check to force the opponent's hand")

    if in_check_before and not gives_check and not is_checkmate:
        reasons.append("Meets the check while keeping the position solid")

    # Evaluation gap versus the second-best line.
    if mate is None and cp is not None and second_mate is None and second_cp is not None:
        gap = cp - second_cp
        if gap >= 150 and not reasons:
            reasons.append(f"Tactically strongest: about {_material_phrase(gap)} better than the alternative")
        elif gap >= 80 and not reasons:
            reasons.append("Creates a clear practical advantage over the alternatives")

    # Positional / development fallback reasoning.
    if not reasons:
        if moving_piece is not None and moving_piece.piece_type in (chess.KNIGHT, chess.BISHOP) and ply <= 20:
            reasons.append("Develops a piece toward an active square in the opening")
        elif moving_piece is not None and moving_piece.piece_type == chess.PAWN:
            reasons.append("Improves the pawn structure and opens space for the pieces")
        elif moving_piece is not None and moving_piece.piece_type in (chess.ROOK, chess.QUEEN):
            reasons.append("Activates a major piece on an open line")
        elif cp is not None and abs(cp) < 40:
            reasons.append("Keeps the position balanced with no tactical concessions")
        else:
            reasons.append("Holds the engine's preferred evaluation in this position")

    # Cap length; join at most two clauses.
    text = "; ".join(reasons[:2])
    return text[0].upper() + text[1:] + "."
