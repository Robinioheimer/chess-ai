"""End-to-end board recognition pipeline.

Combines detection, perspective warping, square splitting, piece recognition
and orientation inference into one testable component. Stability filtering
(identical readings required across several frames) prevents flicker and
duplicate position updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import chess
import cv2
import numpy as np

from ..core.fen_utils import (
    Grid,
    board_grid_to_screen_grid,
    board_to_grid,
    build_fen,
    screen_grid_to_board_grid,
)
from ..logging_setup import get_logger
from .detect import BoardDetection, detect_board
from .features import CellFeatures, extract_cell_features
from .geometry import order_points, quad_area
from .recognizer import PieceRecognizer, classify_non_pawn, king_likeness
from .warp import split_squares, warp_board

LOGGER = get_logger(__name__)


@dataclass
class RecognitionResult:
    """Output of one pipeline run."""

    success: bool
    fen: Optional[str] = None
    screen_grid: Optional[Grid] = None
    board_grid: Optional[Grid] = None
    perspective_white: bool = True
    detection: Optional[BoardDetection] = None
    cell_confidences: list[list[float]] = field(default_factory=list)
    uncertain_squares: list[str] = field(default_factory=list)
    mean_confidence: float = 0.0
    error: Optional[str] = None


def infer_perspective(board_grid: Grid) -> bool:
    """Guess whether White or Black is at the bottom of the screen.

    ``board_grid`` is passed assuming White-at-bottom indexing (``row 0`` is
    the bottom row on screen, ``row 7`` is the top row on screen). We compare
    how many White vs. Black pieces sit in the bottom camp (rows 0..2) and top
    camp (rows 5..7), falling back to overall material balance and finally to
    White-by-default. The user can always override the orientation manually.
    """
    bottom_white = sum(
        1 for row in board_grid[:3] for cell in row
        if cell is not None and cell.isupper()
    )
    bottom_black = sum(
        1 for row in board_grid[:3] for cell in row
        if cell is not None and cell.islower()
    )
    top_white = sum(
        1 for row in board_grid[5:] for cell in row
        if cell is not None and cell.isupper()
    )
    top_black = sum(
        1 for row in board_grid[5:] for cell in row
        if cell is not None and cell.islower()
    )
    camp_score = (bottom_white - bottom_black) + (top_black - top_white)
    if camp_score >= 4:
        return True
    if camp_score <= -4:
        return False

    board_white = sum(
        1 for row in board_grid for cell in row
        if cell is not None and cell.isupper()
    )
    board_black = sum(
        1 for row in board_grid for cell in row
        if cell is not None and cell.islower()
    )
    if board_white == 16 and board_black == 16:
        return True
    if board_white >= board_black + 4:
        return True
    if board_black >= board_white + 4:
        return False
    return True


class RecognitionPipeline:
    """Stateful pipeline with stability filtering and manual overrides."""

    def __init__(
        self,
        recognizer: Optional[PieceRecognizer] = None,
        warp_size: int = 480,
        stable_readings: int = 2,
        min_board_fraction: float = 0.02,
    ) -> None:
        self.recognizer = recognizer or PieceRecognizer()
        self.warp_size = warp_size
        self.stable_readings = max(1, stable_readings)
        self.min_board_fraction = min_board_fraction
        self._last_grid: Optional[Grid] = None
        self._last_raw_grid: Optional[Grid] = None
        self._last_cells: Optional[np.ndarray] = None
        self._candidate_grid: Optional[Grid] = None
        self._candidate_count = 0
        self._last_detection: Optional[BoardDetection] = None
        self.manual_perspective: Optional[bool] = None  # None = automatic
        self.manual_corners: Optional[np.ndarray] = None  # None = automatic
        # Map (screen_row, screen_col) -> (corrected_symbol, raw_symbol_when_set)
        self._square_overrides: dict[tuple[int, int], tuple[Optional[str], Optional[str]]] = {}

    # ----------------------------------------------------------- detection
    def _pad_frame(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """Add a thin background-coloured border.

        Boards touching the screenshot edge otherwise produce a clipped warp;
        the padding ring gives the contour detector background on all sides.
        """
        border = max(8, int(min(frame.shape[:2]) * 0.04))
        median_color = np.median(
            np.concatenate([
                frame[:8].reshape(-1, 3),
                frame[-8:].reshape(-1, 3),
                frame[:, :8].reshape(-1, 3),
                frame[:, -8:].reshape(-1, 3),
            ]),
            axis=0,
        ).astype(np.uint8).tolist()
        padded = cv2.copyMakeBorder(
            frame, border, border, border, border,
            cv2.BORDER_CONSTANT, value=median_color,
        )
        return padded, border

    def _detect(self, frame: np.ndarray) -> tuple[Optional[BoardDetection], np.ndarray, int]:
        """Return detection (in original-frame coordinates), padded frame, padding."""
        padded, padding = self._pad_frame(frame)
        if self.manual_corners is not None:
            corners = order_points(self.manual_corners.copy())
            height, width = frame.shape[:2]
            corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
            corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
            detection = BoardDetection(corners, quad_area(corners), 1.0, "manual")
            self._last_detection = detection
            return detection, padded, padding

        detection = detect_board(padded, min_fraction=self.min_board_fraction)
        if detection is None:
            return None, padded, padding
        if padding:
            # Keep an in-padding copy for warping (no negative source
            # coordinates) and expose original-frame coordinates to the overlay.
            detection.corners = detection.corners - np.array(
                [padding, padding], dtype=np.float32)
            # Clip to the original frame so a contour that slightly bites into
            # the padding ring cannot overshoot the board during warping.
            height, width = frame.shape[:2]
            detection.corners[:, 0] = np.clip(detection.corners[:, 0], 0, width - 1)
            detection.corners[:, 1] = np.clip(detection.corners[:, 1], 0, height - 1)
        self._last_detection = detection
        return detection, padded, padding

    def _warp_from_detection(self, frame: np.ndarray,
                             detection: BoardDetection,
                             padded: np.ndarray, padding: int) -> np.ndarray:
        if not padding:
            return warp_board(frame, detection.corners, self.warp_size)
        padded_corners = detection.corners + np.array(
            [padding, padding], dtype=np.float32)
        return warp_board(padded, padded_corners, self.warp_size)

    # ------------------------------------------------------------- override
    def set_manual_perspective(self, perspective_white: bool) -> None:
        self.manual_perspective = perspective_white

    def clear_manual_perspective(self) -> None:
        self.manual_perspective = None

    def set_manual_corners(self, corners: np.ndarray) -> None:
        """Lock detection onto a user-selected quadrilateral/rectangle."""
        self.manual_corners = order_points(np.asarray(corners, dtype=np.float32))
        self._candidate_grid = None
        self._candidate_count = 0

    def clear_manual_corners(self) -> None:
        """Return to automatic full-screen board detection."""
        self.manual_corners = None

    def apply_manual_correction(
        self,
        corrected_screen_grid: Grid,
        perspective_white: Optional[bool] = None,
        learn_templates: bool = True,
    ) -> None:
        """Persist a user-corrected screen grid and optionally learn templates.

        Each modified square remains pinned to the user's correction until the
        underlying raw vision reading on that square changes (i.e. a move is
        played on that square).
        """
        clean_grid: Grid = [[cell for cell in row] for row in corrected_screen_grid]
        raw_base = self._last_raw_grid or self._last_grid
        if raw_base is not None:
            for row in range(8):
                for col in range(8):
                    raw_sym = raw_base[row][col]
                    corr_sym = clean_grid[row][col]
                    if corr_sym != raw_sym:
                        self._square_overrides[(row, col)] = (corr_sym, raw_sym)
                    else:
                        self._square_overrides.pop((row, col), None)
        self._last_grid = clean_grid
        self._candidate_grid = None
        self._candidate_count = 0
        if perspective_white is not None:
            self.manual_perspective = perspective_white
        if learn_templates and self._last_cells is not None:
            try:
                self.recognizer.calibrate_from_cells(
                    self._last_cells, clean_grid, replace=True
                )
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.debug("Could not learn templates from correction: %s", exc)

    def clear_manual_corrections(self) -> None:
        self._square_overrides.clear()

    def reset_stability(self) -> None:
        self._last_grid = None
        self._last_raw_grid = None
        self._candidate_grid = None
        self._candidate_count = 0
        self._last_detection = None
        self._square_overrides.clear()

    # ------------------------------------------------------------------ run
    def process_frame(self, frame: np.ndarray) -> RecognitionResult:
        """Detect and recognize a board in one screenshot."""
        detection, padded, padding = self._detect(frame)
        if detection is None:
            return RecognitionResult(False, error="No chessboard detected on screen.")

        try:
            warped = self._warp_from_detection(frame, detection, padded, padding)
            cells = split_squares(warped)
            self._last_cells = cells
            screen_grid, confidences = self._recognize_cells(cells)
        except (cv2.error, ValueError) as exc:  # pragma: no cover - defensive
            LOGGER.exception("Recognition failed")
            return RecognitionResult(False, detection=detection,
                                     error=f"Recognition error: {exc}")

        perspective_white = (
            self.manual_perspective
            if self.manual_perspective is not None
            else infer_perspective(screen_grid_to_board_grid(screen_grid, True))
        )
        # Re-align starting-position king/queen columns when manual_perspective is False.
        if (
            self.manual_perspective is False
            and self._is_starting_occupancy(screen_grid)
        ):
            screen_grid = self._starting_screen_grid(perspective_white=False)
        board_grid = screen_grid_to_board_grid(screen_grid, perspective_white)
        fen = build_fen(board_grid)

        flat = [value for row in confidences for value in row]
        mean_confidence = float(np.mean(flat)) if flat else 0.0
        uncertain = self._uncertain_squares(confidences, perspective_white)
        return RecognitionResult(
            success=True,
            fen=fen,
            screen_grid=screen_grid,
            board_grid=board_grid,
            perspective_white=perspective_white,
            detection=detection,
            cell_confidences=confidences,
            uncertain_squares=uncertain,
            mean_confidence=mean_confidence,
        )

    def process_stable(self, frame: np.ndarray) -> RecognitionResult:
        """Like :meth:`process_frame` but only accepts a changed grid once it
        has repeated for ``stable_readings`` consecutive frames.

        The first successful reading is always accepted. Unchanged readings
        return ``success=True`` with the same FEN, letting callers distinguish
        "no board" from "board unchanged".
        """
        result = self.process_frame(frame)
        if not result.success or result.screen_grid is None:
            return result
        signature = tuple(tuple(row) for row in result.screen_grid)
        if self._last_grid is not None and signature == tuple(
            tuple(row) for row in self._last_grid
        ):
            self._candidate_grid = None
            self._candidate_count = 0
            return result
        if signature == self._candidate_signature():
            self._candidate_count += 1
        else:
            self._candidate_grid = result.screen_grid
            self._candidate_count = 1
        if self._candidate_count < self.stable_readings and self._last_grid is not None:
            # Wait for confirmation; report the previously accepted position.
            result.error = None
            LOGGER.debug("Awaiting stable reading %d/%d",
                         self._candidate_count, self.stable_readings)
            stable = RecognitionResult(
                success=True,
                fen=build_fen(screen_grid_to_board_grid(
                    self._last_grid, result.perspective_white)),
                screen_grid=self._last_grid,
                board_grid=screen_grid_to_board_grid(
                    self._last_grid, result.perspective_white),
                perspective_white=result.perspective_white,
                detection=result.detection,
                cell_confidences=result.cell_confidences,
                uncertain_squares=result.uncertain_squares,
                mean_confidence=result.mean_confidence,
            )
            return stable
        self._last_grid = result.screen_grid
        self._candidate_grid = None
        self._candidate_count = 0
        return result

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _adaptive_color_threshold(
        features_grid: list[list[CellFeatures]], default_threshold: float
    ) -> float:
        """Compute a board-wide White/Black piece luminance split if bimodal."""
        lums = [
            f.piece_luminance
            for row in features_grid
            for f in row
            if f.occupied
        ]
        if len(lums) < 4:
            return default_threshold
        lo, hi = float(min(lums)), float(max(lums))
        if hi - lo < 45.0:
            return default_threshold
        threshold = 0.5 * (lo + hi)
        arr = np.array(lums, dtype=np.float32)
        for _ in range(4):
            dark = arr[arr < threshold]
            light = arr[arr >= threshold]
            if dark.size == 0 or light.size == 0:
                return default_threshold
            threshold = 0.5 * (float(dark.mean()) + float(light.mean()))
        return float(threshold)

    @staticmethod
    def _is_starting_occupancy(grid: Grid) -> bool:
        """Check whether rows 0,1 and 6,7 are full (opposite colours) and 2..5 empty."""
        for r in (0, 1, 6, 7):
            if any(grid[r][c] is None for c in range(8)):
                return False
        for r in (2, 3, 4, 5):
            if any(grid[r][c] is not None for c in range(8)):
                return False
        top_white = sum(1 for r in (0, 1) for c in range(8) if grid[r][c] and grid[r][c].isupper())
        bot_white = sum(1 for r in (6, 7) for c in range(8) if grid[r][c] and grid[r][c].isupper())
        return (top_white <= 2 and bot_white >= 14) or (top_white >= 14 and bot_white <= 2)

    @staticmethod
    def _starting_screen_grid(perspective_white: bool) -> Grid:
        start_board_grid = board_to_grid(chess.Board())
        return board_grid_to_screen_grid(start_board_grid, perspective_white)

    def _enforce_chess_constraints(
        self,
        grid: Grid,
        confidences: list[list[float]],
        features_grid: list[list[CellFeatures]],
    ) -> None:
        """Guarantee a legal chess placement (no back-rank pawns, 1 king/side)."""
        # 1. Exact starting-position layout (16 top, 32 empty middle, 16 bottom).
        if self._is_starting_occupancy(grid) and not self.recognizer.calibrated:
            top_white = sum(
                1 for r in (0, 1) for c in range(8)
                if grid[r][c] is not None and grid[r][c].isupper()
            )
            persp_white = top_white < 8 if self.manual_perspective is None else self.manual_perspective
            expected = self._starting_screen_grid(persp_white)
            for r in range(8):
                for c in range(8):
                    grid[r][c] = expected[r][c]
            return

        # 2. Pawns cannot exist on rank 1 or rank 8 (screen rows 0 and 7).
        for row in (0, 7):
            for col in range(8):
                sym = grid[row][col]
                if sym in ("P", "p"):
                    new_type = classify_non_pawn(features_grid[row][col])
                    new_sym = chess. PIECE_SYMBOLS[new_type]
                    grid[row][col] = new_sym.upper() if sym.isupper() else new_sym

        # 3. Ensure exactly one king per colour whenever that colour has pieces on the board.
        for is_white in (True, False):
            king_sym = "K" if is_white else "k"
            queen_sym = "Q" if is_white else "q"
            rook_sym = "R" if is_white else "r"
            piece_coords = [
                (r, c)
                for r in range(8)
                for c in range(8)
                if grid[r][c] is not None and (grid[r][c].isupper() == is_white)
            ]
            if not piece_coords:
                continue
            king_coords = [(r, c) for (r, c) in piece_coords if grid[r][c] == king_sym]
            if len(king_coords) > 1:
                king_coords.sort(
                    key=lambda rc: king_likeness(features_grid[rc[0]][rc[1]]),
                    reverse=True,
                )
                for r, c in king_coords[1:]:
                    f = features_grid[r][c]
                    grid[r][c] = queen_sym if f.h_ratio >= 0.84 else rook_sym
                    confidences[r][c] = min(confidences[r][c], 0.60)
            elif len(king_coords) == 0:
                best_r, best_c = max(
                    piece_coords,
                    key=lambda rc: king_likeness(features_grid[rc[0]][rc[1]]),
                )
                grid[best_r][best_c] = king_sym
                confidences[best_r][best_c] = min(confidences[best_r][best_c], 0.58)

    def _recognize_cells(
        self, cells: np.ndarray
    ) -> tuple[Grid, list[list[float]]]:
        features_grid: list[list[CellFeatures]] = [
            [extract_cell_features(cells[row, col]) for col in range(8)]
            for row in range(8)
        ]
        threshold = self._adaptive_color_threshold(
            features_grid, self.recognizer.color_threshold
        )

        grid: Grid = [[None] * 8 for _ in range(8)]
        confidences = [[0.0] * 8 for _ in range(8)]
        for row in range(8):
            for col in range(8):
                piece, confidence, _features = self.recognizer.recognize_cell(
                    cells[row, col], color_threshold=threshold
                )
                confidences[row][col] = confidence
                if piece is not None:
                    grid[row][col] = piece.symbol()

        self._enforce_chess_constraints(grid, confidences, features_grid)
        self._last_raw_grid = [[cell for cell in row] for row in grid]

        # Apply any active user square corrections whose underlying square has not changed.
        if self._square_overrides:
            for (row, col), (corrected_sym, raw_sym) in list(self._square_overrides.items()):
                if grid[row][col] == raw_sym:
                    grid[row][col] = corrected_sym
                    confidences[row][col] = 0.99
                else:
                    self._square_overrides.pop((row, col), None)
            # Re-verify king uniqueness after overrides.
            self._enforce_chess_constraints(grid, confidences, features_grid)

        return grid, confidences

    def _candidate_signature(self) -> Optional[tuple]:
        if self._candidate_grid is None:
            return None
        return tuple(tuple(row) for row in self._candidate_grid)

    @staticmethod
    def _uncertain_squares(
        confidences: list[list[float]], perspective_white: bool,
        threshold: float = 0.75,
    ) -> list[str]:
        """Return chess-square names with low recognition confidence."""
        uncertain: list[str] = []
        files = "abcdefgh"
        for screen_row in range(8):
            for screen_col in range(8):
                if confidences[screen_row][screen_col] >= threshold:
                    continue
                if perspective_white:
                    file_index = screen_col
                    rank_index = 7 - screen_row
                else:
                    file_index = 7 - screen_col
                    rank_index = screen_row
                uncertain.append(f"{files[file_index]}{rank_index + 1}")
        return uncertain

    def calibrate_from_starting_position(self, frame: np.ndarray) -> int:
        """One-click calibration: learn templates from a visible start board."""
        detection, padded, padding = self._detect(frame)
        if detection is None:
            raise ValueError("No chessboard detected for calibration.")
        warped = self._warp_from_detection(frame, detection, padded, padding)
        cells = split_squares(warped)
        self._last_cells = cells
        if self.manual_perspective is not None:
            perspective_white = self.manual_perspective
        else:
            top_lums = [
                extract_cell_features(cells[r, c]).piece_luminance
                for r in (0, 1) for c in range(8)
            ]
            bot_lums = [
                extract_cell_features(cells[r, c]).piece_luminance
                for r in (6, 7) for c in range(8)
            ]
            perspective_white = float(np.mean(bot_lums)) >= float(np.mean(top_lums))
        start_grid = board_to_grid(chess.Board())
        symbols = board_grid_to_screen_grid(start_grid, perspective_white)
        self.recognizer.calibrate_from_cells(cells, symbols, replace=True)
        return len([s for row in symbols for s in row if s is not None])

    @property
    def last_detection(self) -> Optional[BoardDetection]:
        return getattr(self, "_last_detection", None)

    @staticmethod
    def ordered_corners(detection: BoardDetection) -> np.ndarray:
        return order_points(detection.corners)
