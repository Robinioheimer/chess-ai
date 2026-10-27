"""Piece recognition combining shape heuristics with calibrated templates.

Two complementary classifiers:

* :meth:`classify_heuristic` - rule based, needs no calibration and works on
  any piece set whose silhouettes follow conventional chess designs.
* :meth:`classify_template` - nearest-neighbour matching on Hu moments and
  shape features against templates captured from the user's actual board
  (one click on a known starting position, or learned from manual corrections).
  This handles custom piece sets.

The recognizer always returns a confidence value so the pipeline can flag
uncertain squares for manual correction.
"""
from __future__ import annotations

from pathlib import Path

import chess
import cv2
import numpy as np

from ..logging_setup import get_logger
from .features import CellFeatures, extract_cell_features

LOGGER = get_logger(__name__)

PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
# Features compared during template matching, in normalized order.
FEATURE_KEYS = (
    "h_ratio", "w_ratio", "aspect", "extent", "solidity", "top_width_ratio",
    "width_at_5", "width_at_15", "width_at_35", "width_at_55", "waist_ratio",
    "upper_mass_ratio", "symmetry", "hole_count", "top_transitions",
    "center_protrusion", "crossbar_ratio", "top_offset", "centroid_x", "centroid_y",
)
# Roughly calibrated acceptance threshold (lower is better for the score).
TEMPLATE_ACCEPTANCE = 2.2
COLOR_LUMINANCE_THRESHOLD = 110.0


def classify_heuristic(features: CellFeatures) -> chess.PieceType:
    """Rule-based piece-type classification from shape descriptors."""
    f = features
    # 1. Knight: its snout/mane make it the only strongly asymmetric piece.
    #    Catches both off-centre-top knights (top_offset >= 0.14) and
    #    centred-ear / left-facing knights (0.40 <= symmetry < 0.885).
    if (f.top_offset >= 0.14 and f.symmetry < 0.94) or (0.40 <= f.symmetry < 0.885):
        return chess.KNIGHT
    # 2. King: tallest symmetric silhouette with a cross that stays narrower
    #    than the queen's goblet crown (queen crowns span >= 0.74 of width).
    if f.h_ratio >= 0.90 and f.top_width_ratio <= 0.73:
        return chess.KING
    # 3. Bishop: sharp mitre gives a very narrow upper section.
    if f.width_at_15 <= 0.23:
        return chess.BISHOP
    # 4. Queen: tall piece with a wide goblet crown. Rooks never reach this
    #    height (their slab-to-base span stays below 0.85 of the bounding box).
    if f.h_ratio >= 0.85 and f.top_width_ratio >= 0.73:
        return chess.QUEEN
    # 5. Rook: broad flat slab top.
    if f.top_width_ratio >= 0.65 or f.width_at_5 >= 0.32:
        return chess.ROOK
    # 6. Pawn: compact symmetric silhouette with a small rounded head.
    return chess.PAWN


def classify_non_pawn(features: CellFeatures) -> chess.PieceType:
    """Classify an occupied square that cannot legally be a pawn (rank 1/8)."""
    f = features
    if (f.top_offset >= 0.10 and f.symmetry < 0.95) or (0.40 <= f.symmetry < 0.895):
        return chess.KNIGHT
    if f.h_ratio >= 0.82:
        return chess.KING if f.top_width_ratio <= 0.73 else chess.QUEEN
    if f.top_width_ratio >= 0.45 or f.width_at_5 >= 0.22:
        return chess.ROOK
    if 0.0 < f.width_at_15 <= 0.28:
        return chess.BISHOP
    return chess.ROOK


def king_likeness(features: CellFeatures) -> float:
    """Continuous score for how strongly a silhouette resembles a king."""
    f = features
    crown_penalty = max(0.0, f.top_width_ratio - 0.73) * 1.2
    knight_penalty = max(0.0, f.top_offset - 0.12) * 1.5
    return float(
        2.2 * f.h_ratio
        + 1.6 * f.center_protrusion
        + 0.8 * f.crossbar_ratio
        + 0.5 * f.symmetry
        - crown_penalty
        - knight_penalty
    )


class PieceRecognizer:
    """Template-aware piece recognizer with a heuristic fallback."""

    def __init__(self, color_threshold: float = COLOR_LUMINANCE_THRESHOLD) -> None:
        self.color_threshold = color_threshold
        self._symbols: list[str] = []
        self._masks: list[np.ndarray] = []
        self._vectors: list[np.ndarray] = []

    # -------------------------------------------------------- calibration
    def clear_templates(self) -> None:
        """Remove all learned piece templates."""
        self._symbols.clear()
        self._masks.clear()
        self._vectors.clear()

    def calibrate_from_cells(
        self,
        cells: np.ndarray,
        symbols: list[list[str | None]],
        replace: bool = False,
    ) -> None:
        """Learn templates from a board whose pieces are known (e.g. start pos).

        ``cells`` is an 8x8 object array of square images in screen order and
        ``symbols`` the matching FEN symbols.
        """
        if replace:
            self.clear_templates()
        for row in range(8):
            for col in range(8):
                symbol = symbols[row][col]
                if symbol is None:
                    continue
                features = extract_cell_features(cells[row, col])
                if not features.occupied or features.mask is None:
                    LOGGER.warning("Calibration square %d/%d expected %s but looked empty",
                                   row, col, symbol)
                    continue
                self._symbols.append(symbol)
                self._masks.append(features.mask)
                self._vectors.append(self._feature_vector(features))
        LOGGER.info("Calibrated %d piece templates (%d distinct symbols)",
                    len(self._symbols), len(set(self._symbols)))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._symbols:
            raise ValueError("No templates to save.")
        np.savez(
            path,
            symbols=np.array(self._symbols),
            masks=np.stack(self._masks).astype(np.uint8),
            vectors=np.stack(self._vectors).astype(np.float32),
        )
        return path

    def load(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self._symbols = list(data["symbols"].astype(str))
        self._masks = list(data["masks"])
        self._vectors = list(data["vectors"])
        LOGGER.info("Loaded %d piece templates from %s", len(self._symbols), path)

    @property
    def calibrated(self) -> bool:
        return bool(self._symbols)

    # -------------------------------------------------------- recognition
    def recognize_cell(
        self,
        cell: np.ndarray,
        color_threshold: float | None = None,
    ) -> tuple[chess.Piece | None, float, CellFeatures]:
        """Classify one square.

        Returns ``(piece, confidence, features)``; ``piece`` is ``None`` for
        empty squares.
        """
        features = extract_cell_features(cell)
        if not features.occupied:
            return None, 0.95, features
        threshold = self.color_threshold if color_threshold is None else color_threshold
        color = chess.WHITE if features.piece_luminance >= threshold else chess.BLACK

        if self.calibrated:
            symbol, score = self.classify_template(features, color)
            if symbol is not None and score <= TEMPLATE_ACCEPTANCE:
                piece_type = chess.PIECE_SYMBOLS.index(symbol.lower())
                confidence = float(np.clip(1.0 - score / (TEMPLATE_ACCEPTANCE * 1.6), 0.45, 0.99))
                return chess.Piece(piece_type, color), confidence, features
            LOGGER.debug("Template match weak (%.2f), using heuristic fallback", score)

        piece_type = classify_heuristic(features)
        return chess.Piece(piece_type, color), 0.62, features

    def classify_template(self, features: CellFeatures,
                          color: chess.Color) -> tuple[str | None, float]:
        """Nearest template of the given color; returns symbol and match score."""
        if features.mask is None or not self._symbols:
            return None, float("inf")
        wanted_case = "upper" if color == chess.WHITE else "lower"
        best_symbol, best_score = None, float("inf")
        vector = self._feature_vector(features)
        for symbol, mask, template_vector in zip(self._symbols, self._masks, self._vectors):
            if (symbol.isupper() and wanted_case != "upper") or \
               (symbol.islower() and wanted_case != "lower"):
                continue
            hu_distance = cv2.matchShapes(features.mask, mask, cv2.CONTOURS_MATCH_I1, 0.0)
            feature_distance = float(np.linalg.norm(vector - template_vector))
            score = hu_distance * 8.0 + feature_distance
            if score < best_score:
                best_score, best_symbol = score, symbol
        return best_symbol, best_score

    @staticmethod
    def _feature_vector(features: CellFeatures) -> np.ndarray:
        return np.array([getattr(features, key) for key in FEATURE_KEYS], dtype=np.float32)
