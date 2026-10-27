"""Position tracking, change detection and duplicate prevention.

The vision pipeline emits raw FEN *placements* (it cannot see who is to move).
The tracker is the single source of truth for the current game state: it
validates placements, suppresses duplicate readings, alternates the side to
move when the placement changes, and accepts manual corrections / FEN input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import chess

from .fen_utils import build_fen, parse_fen, validate_fen
from ..logging_setup import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PositionUpdate:
    """Result of feeding a new observed placement to the tracker."""

    accepted: bool
    changed: bool
    fen: str
    board: chess.Board
    reason: str
    move_uci: Optional[str] = None


class PositionTracker:
    """Stateful tracker of the currently displayed chess position."""

    def __init__(self, side_to_move: chess.Color = chess.WHITE) -> None:
        self._board: Optional[chess.Board] = None
        self._last_placement: Optional[str] = None  # placement-only FEN (first 4 fields)
        self._ignored_vision_placement: Optional[str] = None
        self.side_to_move: chess.Color = side_to_move
        self.move_counter: int = 0

    # ------------------------------------------------------------- state
    @property
    def board(self) -> Optional[chess.Board]:
        return self._board

    @property
    def fen(self) -> Optional[str]:
        return None if self._board is None else self._board.fen()

    @property
    def has_position(self) -> bool:
        return self._board is not None

    # ------------------------------------------------------------ updates
    def observe_placement_fen(self, placement_fen: str) -> PositionUpdate:
        """Process a FEN produced by computer vision.

        The FEN may carry any side-to-move flag; only the piece placement is
        trusted. A changed, legal placement flips the side to move (one move
        was played). Identical readings are de-duplicated.
        """
        ok, message = validate_fen(self._normalize(placement_fen))
        if not ok:
            LOGGER.debug("Rejected illegal observed position: %s", message)
            return self._rejected(placement_fen, f"Illegal position: {message}")

        observed = chess.Board(self._normalize(placement_fen))
        placement_key = self._placement_key(observed)

        if self._board is None:
            observed.turn = self.side_to_move
            observed.castling_rights = 0
            observed.ep_square = None
            self._board = observed
            self._last_placement = placement_key
            LOGGER.info("Initial position accepted (%s)", observed.fen())
            return PositionUpdate(True, True, observed.fen(), observed, "Initial position detected")

        if placement_key == self._last_placement:
            # A valid duplicate reading: nothing changed, but not an error.
            return PositionUpdate(True, False, self._board.fen(), self._board,
                                  "Same position as last reading")

        if (
            self._ignored_vision_placement is not None
            and placement_key == self._ignored_vision_placement
        ):
            # Screen still shows the pre-correction vision reading; keep manual override.
            return PositionUpdate(True, False, self._board.fen(), self._board,
                                  "Same position as last reading (manual override active)")
        self._ignored_vision_placement = None

        move = self._guess_move(self._board, observed)
        # A move happened: flip side to move and adopt the new placement.
        observed.turn = not self._board.turn
        observed.castling_rights = 0
        observed.ep_square = None
        self._board = observed
        self._last_placement = placement_key
        self.move_counter += 1
        reason = f"Move detected: {move}" if move else "Position changed"
        LOGGER.info("New position accepted (%s)", reason)
        return PositionUpdate(True, True, observed.fen(), observed, reason, move)

    def observe_grid(self, grid, side_to_move: str = "w") -> PositionUpdate:
        """Convenience wrapper accepting a board-coordinate piece grid."""
        return self.observe_placement_fen(build_fen(grid, side_to_move))

    def set_manual_fen(
        self, fen: str, raw_vision_fen: Optional[str] = None
    ) -> PositionUpdate:
        """Apply manually entered or corrected FEN (full FEN is trusted)."""
        ok, message = validate_fen(fen)
        if not ok:
            return self._rejected(fen, f"Invalid FEN: {message}")
        board = parse_fen(fen)
        new_key = self._placement_key(board)
        if raw_vision_fen:
            try:
                raw_board = chess.Board(self._normalize(raw_vision_fen))
                raw_key = self._placement_key(raw_board)
                if raw_key != new_key:
                    self._ignored_vision_placement = raw_key
            except ValueError:
                pass
        elif self._last_placement is not None and self._last_placement != new_key:
            self._ignored_vision_placement = self._last_placement
        changed = self._board is None or new_key != self._last_placement or board.turn != self.side_to_move
        self._board = board
        self._last_placement = new_key
        self.side_to_move = board.turn
        if changed:
            self.move_counter += 1
        LOGGER.info("Manual position set: %s", fen)
        return PositionUpdate(True, changed, board.fen(), board, "Manual FEN applied")

    def set_side_to_move(self, color: chess.Color) -> None:
        """Manual override of the side to move (also fixes the stored FEN)."""
        self.side_to_move = color
        if self._board is not None:
            self._board.turn = color

    def reset(self, side_to_move: chess.Color = chess.WHITE) -> None:
        """Forget the current position."""
        self._board = None
        self._last_placement = None
        self._ignored_vision_placement = None
        self.side_to_move = side_to_move
        self.move_counter = 0

    # ------------------------------------------------------------ helpers
    def _rejected(self, raw_fen: str, reason: str) -> PositionUpdate:
        current = self._board.fen() if self._board else raw_fen
        board = self._board or chess.Board()
        return PositionUpdate(False, False, current, board, reason)

    @staticmethod
    def _normalize(fen: str) -> str:
        """Ensure the FEN has all six fields (vision only produces placement)."""
        parts = fen.split()
        if len(parts) == 1:
            return f"{fen} w - - 0 1"
        while len(parts) < 6:
            parts.append("-")
        return " ".join(parts)

    @staticmethod
    def _placement_key(board: chess.Board) -> str:
        """Hashable key covering piece placement only."""
        return board.board_fen()

    @staticmethod
    def _guess_move(previous: chess.Board, current: chess.Board) -> Optional[str]:
        """Best-effort UCI move describing the transition, if a single move fits."""
        prev_map = previous.piece_map()
        cur_map = current.piece_map()
        from_squares = [sq for sq in prev_map if sq not in cur_map or cur_map[sq] != prev_map[sq]]
        to_squares = [sq for sq in cur_map if sq not in prev_map or prev_map.get(sq) != cur_map[sq]]
        if len(from_squares) == 1 and len(to_squares) >= 1:
            # Prefer the destination carrying the moving piece's color.
            origin = from_squares[0]
            piece = prev_map[origin]
            for target in to_squares:
                if cur_map.get(target) == piece:
                    return chess.Move(origin, target).uci()
            return chess.Move(origin, to_squares[-1]).uci()
        return None
