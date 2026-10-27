"""FEN construction, parsing and validation helpers.

A *grid* is an 8x8 list-of-lists indexed ``grid[rank][file]`` where
``rank = 0`` is chess rank 1 (White's home rank) and ``file = 0`` is the
a-file. Each cell is either ``None`` (empty) or a single FEN piece symbol,
e.g. ``"P"`` (white pawn) or ``"n"`` (black knight).

Computer vision produces grids in screen order (top row first). The
:func:`screen_grid_to_board_grid` helper converts them into the board
coordinate system used here.
"""
from __future__ import annotations

from typing import Iterable, Optional

import chess

PieceSymbol = str  # one of P N B R Q K p n b r q k
Grid = list[list[Optional[PieceSymbol]]]

FILES = "abcdefgh"
# Standard piece values (centipawns) used for material heuristics.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def screen_grid_to_board_grid(screen_grid: Grid, perspective_white: bool) -> Grid:
    """Convert a screen-order grid (top row first) to board coordinates.

    ``perspective_white=True`` means White's home rank is at the bottom of
    the screen (the usual White view). When Black is on bottom, rows and
    columns are flipped.
    """
    if len(screen_grid) != 8 or any(len(row) != 8 for row in screen_grid):
        raise ValueError("Grid must be 8x8.")
    if perspective_white:
        # White at bottom: screen row 0 (top) is rank 8, files are not mirrored.
        return [list(row) for row in reversed(screen_grid)]
    # Black at bottom: screen row 0 (top) is rank 1 but the a-file is on the
    # viewer's right, so only columns are mirrored.
    return [list(reversed(row)) for row in screen_grid]


def board_grid_to_screen_grid(board_grid: Grid, perspective_white: bool) -> Grid:
    """Inverse of :func:`screen_grid_to_board_grid`."""
    if perspective_white:
        return [list(row) for row in reversed(board_grid)]
    return [list(reversed(row)) for row in board_grid]


def grid_to_board(grid: Grid) -> chess.Board:
    """Build a :class:`chess.Board` from a board-coordinate grid."""
    board = chess.Board(fen=None)  # completely empty board
    for rank_index, row in enumerate(grid):
        for file_index, symbol in enumerate(row):
            if symbol is None:
                continue
            square = chess.square(file_index, rank_index)
            piece = chess.Piece.from_symbol(symbol)
            board.set_piece_at(square, piece)
    return board


def board_to_grid(board: chess.Board) -> Grid:
    """Serialize a board to a board-coordinate grid."""
    grid: Grid = [[None] * 8 for _ in range(8)]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            grid[chess.square_rank(square)][chess.square_file(square)] = piece.symbol()
    return grid


def build_fen(grid: Grid, side_to_move: str = "w") -> str:
    """Build a legal *position* FEN from a board-coordinate grid.

    Castling rights and en-passant squares cannot be inferred reliably from a
    single screenshot, so they are emitted as ``-``. The halfmove/fullmove
    counters use safe defaults. The FEN is validated with python-chess.
    """
    if side_to_move not in ("w", "b"):
        raise ValueError("side_to_move must be 'w' or 'b'")
    board = grid_to_board(grid)
    board.turn = chess.WHITE if side_to_move == "w" else chess.BLACK
    board.castling_rights = 0
    board.ep_square = None
    return board.fen()


def validate_fen(fen: str) -> tuple[bool, str]:
    """Validate a FEN string.

    Returns ``(ok, message)``. ``message`` is empty when validation succeeds.
    """
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return False, str(exc)
    white_kings = len(board.pieces(chess.KING, chess.WHITE))
    black_kings = len(board.pieces(chess.KING, chess.BLACK))
    if white_kings != 1 or black_kings != 1:
        return False, f"Position must contain exactly one king per side (white={white_kings}, black={black_kings})."
    if board.is_check():
        # A check is legal, but the side NOT to move giving check is impossible.
        pass
    return True, ""


def parse_fen(fen: str) -> chess.Board:
    """Strict FEN parse raising :class:`ValueError` on invalid input."""
    ok, message = validate_fen(fen)
    if not ok:
        raise ValueError(message)
    return chess.Board(fen)


def square_name(file_index: int, rank_index: int) -> str:
    """Return algebraic square name from 0-based board coordinates."""
    if not (0 <= file_index <= 7 and 0 <= rank_index <= 7):
        raise ValueError("Square coordinates must be between 0 and 7.")
    return f"{FILES[file_index]}{rank_index + 1}"


def grid_signature(grid: Grid) -> str:
    """Compact, hashable signature of piece placement (ignores side to move)."""
    return "/".join("".join(cell or "." for cell in row) for row in grid)


def count_material(board: chess.Board, color: chess.Color) -> int:
    """Total material value in centipawns for one side (kings excluded)."""
    return sum(
        PIECE_VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.color == color
    )


def legal_move_count(board: chess.Board) -> int:
    """Number of legal moves in the current position."""
    return board.legal_moves.count()


def iter_legal_moves(board: chess.Board) -> Iterable[chess.Move]:
    return iter(board.legal_moves)
