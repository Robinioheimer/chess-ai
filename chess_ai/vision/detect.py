"""Chessboard quadrilateral detection in screenshots.

The detector gathers candidate quadrilaterals from the strongest contours and
validates them with an internal *grid score*: a real chessboard shows eight
regularly spaced rows and columns of alternating colour, which produces nine
edge lines with near-constant spacing in both projection axes. This rejects
internal checkerboard contours that approximate to distorted quadrilaterals.

An axis-aligned bounding-rect candidate is always evaluated as well, keeping
the system reliable for perfectly rectangular on-screen boards and boards
touching the screenshot edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..logging_setup import get_logger
from .geometry import bounding_rect, order_points, quad_area
from .warp import warp_board

LOGGER = get_logger(__name__)


@dataclass
class BoardDetection:
    corners: np.ndarray          # ordered (tl, tr, br, bl), screen coordinates
    area: float
    confidence: float            # 0..1, 1.0 = proper quadrilateral
    method: str

    @property
    def bounding_rect(self) -> tuple[int, int, int, int]:
        return bounding_rect(self.corners)


# --------------------------------------------------------------- grid scoring
def _projection_peaks(profile: np.ndarray, min_distance: int) -> list[int]:
    """Strong, regularly spaced peaks in a 1-D projection profile."""
    if profile.size < 3:
        return []
    threshold = float(profile.mean()) + 0.6 * float(profile.std())
    peaks: list[int] = []
    for index in range(1, profile.size - 1):
        if profile[index] >= threshold and \
                profile[index] >= profile[index - 1] and \
                profile[index] >= profile[index + 1]:
            if not peaks or index - peaks[-1] >= min_distance:
                peaks.append(index)
            elif profile[index] > profile[peaks[-1]]:
                peaks[-1] = index
    return peaks


def grid_score(image: np.ndarray, size: int = 240) -> float:
    """Score how strongly an image looks like a warped 8x8 chessboard.

    Returns a value in ``[0, 1]`` combining the number of detected grid lines
    (ideally 9 per axis) with their spacing regularity.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size))
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    min_distance = max(4, size // 12)

    scores = []
    for gradient, axis in ((grad_x, 0), (grad_y, 1)):
        projection = np.abs(gradient).sum(axis=axis)
        peaks = _projection_peaks(projection, min_distance)
        if len(peaks) < 4:
            scores.append(0.0)
            continue
        count_score = min(1.0, len(peaks) / 9.0)
        gaps = np.diff(peaks)
        if gaps.size and gaps.mean() > 0:
            regularity = float(1.0 - np.clip(gaps.std() / gaps.mean(), 0, 1))
        else:
            regularity = 0.0
        scores.append(0.6 * count_score + 0.4 * regularity)
    return float(0.5 * scores[0] + 0.5 * scores[1])


# --------------------------------------------------------------- candidates
def _quad_candidate(contour: np.ndarray, image_area: int,
                    min_fraction: float) -> BoardDetection | None:
    area = cv2.contourArea(contour)
    if area < image_area * min_fraction:
        return None
    peri = cv2.arcLength(contour, True)
    for epsilon in (0.02, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(contour, epsilon * peri, True)
        if len(approx) != 4:
            continue
        corners = approx.reshape(4, 2).astype(np.float32)
        if not cv2.isContourConvex(corners):
            continue
        x, y, w, h = bounding_rect(corners)
        if w == 0 or h == 0 or not 0.7 <= w / float(h) <= 1.4:
            continue
        fill_ratio = area / float(w * h)
        if fill_ratio < 0.65:
            continue
        ordered = order_points(corners)
        confidence = float(np.clip(0.7 + 0.2 * fill_ratio, 0.5, 1.0))
        return BoardDetection(ordered, float(area), confidence, "quadrilateral")
    return None


def _rect_candidate(contour: np.ndarray, image_area: int,
                    min_fraction: float) -> BoardDetection | None:
    area = cv2.contourArea(contour)
    if area < image_area * min_fraction:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if not 0.7 <= w / max(1, h) <= 1.4:
        return None
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                       dtype=np.float32)
    return BoardDetection(order_points(corners), quad_area(corners), 0.5,
                          "bounding-rect")


# -------------------------------------------------------------------- public
def detect_board(image: np.ndarray, min_fraction: float = 0.08) -> BoardDetection | None:
    """Detect the most plausible chessboard quadrilateral in ``image``."""
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    image_area = gray.shape[0] * gray.shape[1]

    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    if not contours:
        LOGGER.info("No contours found in screenshot")
        return None

    candidates: list[BoardDetection] = []
    for contour in contours:
        quad = _quad_candidate(contour, image_area, min_fraction)
        if quad is not None:
            candidates.append(quad)
        rect = _rect_candidate(contour, image_area, min_fraction)
        if rect is not None:
            candidates.append(rect)
    # Deduplicate near-identical candidates.
    unique: list[BoardDetection] = []
    for candidate in candidates:
        if all(np.linalg.norm(candidate.corners - other.corners, axis=1).mean() > 6
               for other in unique):
            unique.append(candidate)

    if not unique:
        LOGGER.info("No board-sized quadrilateral detected")
        return None

    scored = []
    for candidate in unique:
        try:
            warped = warp_board(image, candidate.corners, size=240)
            score = grid_score(warped)
        except cv2.error:  # pragma: no cover - defensive
            score = 0.0
        scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)

    best_score, best = scored[0]
    if best_score < 0.35:
        # No convincing grid: fall back to the largest plausible candidate.
        best = max(unique, key=lambda c: c.area)
        LOGGER.debug("Low grid score (%.2f); falling back to largest contour", best_score)
    else:
        LOGGER.debug("Board detected via %s (grid score %.2f, area %.0f)",
                     best.method, best_score, best.area)
    return best
