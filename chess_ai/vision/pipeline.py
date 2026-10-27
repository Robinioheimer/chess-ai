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
    board_to_grid,
    build_fen,
    screen_grid_to_board_grid,
)
from ..logging_setup import get_logger
from .detect import BoardDetection, detect_board
from .geometry import order_points
from .recognizer import PieceRecognizer
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

    A standard starting position is unambiguous. For arbitrary positions we
    fall back to material balance (the stronger side is assumed to sit at the
    bottom) and finally to White-by-default. The user can always override the
    orientation manually in the UI.
    """
    board_white = sum(
        1 for row in board_grid for cell in row
        if cell is not None and cell.isupper()
    )
    board_black = sum(
        1 for row in board_grid for cell in row
        if cell is not None and cell.islower()
    )
    if board_white == 16 and board_black == 16:
        # A complete starting set: White at the bottom is the standard view.
        return True
    # Strong imbalance suggests the bottom player's camp is fully populated.
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
        min_board_fraction: float = 0.08,
    ) -> None:
        self.recognizer = recognizer or PieceRecognizer()
        self.warp_size = warp_size
        self.stable_readings = max(1, stable_readings)
        self.min_board_fraction = min_board_fraction
        self._last_grid: Optional[Grid] = None
        self._candidate_grid: Optional[Grid] = None
        self._candidate_count = 0
        self._last_detection: Optional[BoardDetection] = None
        self.manual_perspective: Optional[bool] = None  # None = automatic

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

    def reset_stability(self) -> None:
        self._last_grid = None
        self._candidate_grid = None
        self._candidate_count = 0
        self._last_detection = None

    # ------------------------------------------------------------------ run
    def process_frame(self, frame: np.ndarray) -> RecognitionResult:
        """Detect and recognize a board in one screenshot."""
        detection, padded, padding = self._detect(frame)
        if detection is None:
            return RecognitionResult(False, error="No chessboard detected on screen.")

        try:
            warped = self._warp_from_detection(frame, detection, padded, padding)
            cells = split_squares(warped)
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
    def _recognize_cells(
        self, cells: np.ndarray
    ) -> tuple[Grid, list[list[float]]]:
        grid: Grid = [[None] * 8 for _ in range(8)]
        confidences = [[0.0] * 8 for _ in range(8)]
        for row in range(8):
            for col in range(8):
                piece, confidence, _features = self.recognizer.recognize_cell(cells[row, col])
                confidences[row][col] = confidence
                if piece is not None:
                    grid[row][col] = piece.symbol()
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
        start_grid = board_to_grid(chess.Board())
        from ..core.fen_utils import board_grid_to_screen_grid
        symbols = board_grid_to_screen_grid(start_grid, True)
        self.recognizer.calibrate_from_cells(cells, symbols)
        return len([s for row in symbols for s in row if s is not None])

    @property
    def last_detection(self) -> Optional[BoardDetection]:
        return getattr(self, "_last_detection", None)

    @staticmethod
    def ordered_corners(detection: BoardDetection) -> np.ndarray:
        return order_points(detection.corners)
