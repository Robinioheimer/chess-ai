"""Chessboard quadrilateral detection in screenshots.

The detector combines two complementary strategies:

1. **Multi-pass contour candidates** (both ``RETR_EXTERNAL`` and ``RETR_LIST``
   across Canny and adaptive-threshold edge maps) so boards nested inside
   browser windows, desktop applications or split panels are found reliably,
   including rotated and perspective-skewed boards.
2. **7x7 internal X-corner (saddle) lattice detection** that locates the 49
   internal square intersections where two light squares on one diagonal meet
   two dark squares on the other diagonal, then extrapolates by one square
   width on all four sides. This detects boards with *no outer border* (where
   edge squares blend into a light or dark page background) and strips away
   decorative outer frames around the playable 8x8 grid.

Every candidate is validated with :func:`grid_score`.
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

    Returns a value in ``[0, 1]`` combining:
    * regularity of detected projection peaks,
    * alignment of gradient energy with the 7 internal grid lines ``k * size/8``,
    * 7x7 internal X-corner (checkerboard saddle) alternation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size))
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    min_distance = max(4, size // 12)
    step = size // 8

    peak_scores = []
    align_scores = []
    for gradient, axis in ((grad_x, 0), (grad_y, 1)):
        projection = np.abs(gradient).sum(axis=axis)
        peaks = _projection_peaks(projection, min_distance)
        if len(peaks) < 4:
            peak_scores.append(0.0)
        else:
            count_score = min(1.0, len(peaks) / 9.0)
            gaps = np.diff(peaks)
            if gaps.size and gaps.mean() > 0:
                regularity = float(1.0 - np.clip(gaps.std() / gaps.mean(), 0, 1))
            else:
                regularity = 0.0
            peak_scores.append(0.6 * count_score + 0.4 * regularity)

        mean_proj = float(projection.mean()) + 1e-5
        energies = []
        for k in range(1, 8):
            pos = k * step
            band = projection[max(0, pos - 3): min(size, pos + 4)]
            energies.append(float(band.max()) / mean_proj)
        active = sum(1 for e in energies if e > 1.3) / 7.0
        strength = float(np.clip((np.mean(energies) - 1.0) / 2.5, 0.0, 1.0))
        align_scores.append(0.5 * active + 0.5 * strength)

    blur = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    d = max(3, step // 6)
    saddles = []
    signs = []
    for r in range(1, 8):
        for c in range(1, 8):
            y, x = r * step, c * step
            tl = float(blur[y - d, x - d])
            br = float(blur[y + d, x + d])
            tr = float(blur[y - d, x + d])
            bl = float(blur[y + d, x - d])
            diff = (tl + br) - (tr + bl)
            penalty = abs(tl - br) + abs(tr - bl)
            saddles.append(max(0.0, abs(diff) - 1.2 * penalty))
            signs.append(np.sign(diff) * ((-1) ** (r + c)))
    saddle_active = float(np.mean([s > 8.0 for s in saddles]))
    sign_consistency = abs(float(np.mean(signs)))
    saddle_score = saddle_active * (0.4 + 0.6 * sign_consistency)

    base = 0.5 * peak_scores[0] + 0.5 * peak_scores[1]
    aligned = 0.5 * align_scores[0] + 0.5 * align_scores[1]
    return float(np.clip(0.30 * base + 0.35 * aligned + 0.35 * saddle_score, 0.0, 1.0))


# --------------------------------------------------------------- candidates
def _quad_candidate(contour: np.ndarray, image_area: int,
                    min_fraction: float) -> BoardDetection | None:
    area = cv2.contourArea(contour)
    if area < image_area * min_fraction or area > image_area * 0.995:
        return None
    peri = cv2.arcLength(contour, True)
    for epsilon in (0.015, 0.02, 0.04, 0.06, 0.08):
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
    if area < image_area * min_fraction or area > image_area * 0.995:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if not 0.7 <= w / max(1, h) <= 1.4:
        return None
    if (w * h) > image_area * 0.995:
        return None
    corners = np.array([[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
                       dtype=np.float32)
    return BoardDetection(order_points(corners), quad_area(corners), 0.5,
                          "bounding-rect")


def _refine_7_lines(profile: np.ndarray, p1: int, step: int,
                    limit: int, window: int = 4) -> tuple[float, float]:
    """Fit a least-squares line to 7 internal grid peaks and extrapolate to 0..8."""
    ks = np.arange(1, 8, dtype=np.float32)
    positions: list[float] = []
    for k_idx in range(7):
        center = p1 + k_idx * step
        lo = max(0, center - window)
        hi = min(profile.size, center + window + 1)
        seg = profile[lo:hi]
        total = float(seg.sum())
        if total > 0:
            coords = np.arange(lo, hi, dtype=np.float32)
            positions.append(float((coords * seg).sum() / total))
        else:
            positions.append(float(center))
    pts = np.array(positions, dtype=np.float32)
    mean_p = float(pts.mean())
    fitted_step = float(((ks - 4.0) * pts).sum() / 28.0)
    if fitted_step <= 4.0:
        fitted_step = float(step)
    p0 = float(np.clip(mean_p - 4.0 * fitted_step, 0.0, limit - 1.0))
    p8 = float(np.clip(mean_p + 4.0 * fitted_step - 1.0, 0.0, limit - 1.0))
    return p0, p8


def _saddle_lattice_candidate(image: np.ndarray,
                              min_fraction: float) -> BoardDetection | None:
    """Detect an 8x8 board via its 49 internal X-corners (7x7 saddle lattice)."""
    h, w = image.shape[:2]
    image_area = h * w
    min_board_side = max(64, int(np.sqrt(image_area * min_fraction)))
    max_board_side = min(h, w)
    if min_board_side > max_board_side:
        return None

    # Downscale large screenshots for fast lattice search, then refine in full res.
    scale = 1.0
    max_dim = max(h, w)
    if max_dim > 960:
        scale = 960.0 / float(max_dim)
        work_img = cv2.resize(
            image, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        work_img = image

    wh, ww = work_img.shape[:2]
    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    d = max(3, min(wh, ww) // 120)
    if wh <= 2 * d + 16 or ww <= 2 * d + 16:
        return None

    tl = blur[:-2 * d, :-2 * d]
    br = blur[2 * d:, 2 * d:]
    tr = blur[:-2 * d, 2 * d:]
    bl = blur[2 * d:, :-2 * d]
    diag_diff = (tl + br) - (tr + bl)
    diag_penalty = np.abs(tl - br) + np.abs(tr - bl)
    raw_saddle = np.maximum(0.0, np.abs(diag_diff) - 1.35 * diag_penalty)

    saddle = np.zeros((wh, ww), dtype=np.float32)
    signed = np.zeros((wh, ww), dtype=np.float32)
    saddle[d:wh - d, d:ww - d] = cv2.GaussianBlur(raw_saddle, (5, 5), 0)
    signed[d:wh - d, d:ww - d] = cv2.GaussianBlur(diag_diff, (5, 5), 0)

    if float(saddle.max()) < 10.0:
        return None

    proj_y = saddle.sum(axis=1)
    min_s = max(6, int(round((min_board_side * scale) / 8.0)))
    max_s = max(min_s, ww // 8, wh // 8)
    checker = ((-1) ** (np.arange(7)[:, None] + np.arange(7)[None, :])).astype(np.float32)

    best_score = -1.0
    best_params: tuple[int, int, int] | None = None

    for s in range(min_s, max_s + 1):
        span = 6 * s
        valid_h = wh - 2 * s - span
        valid_w = ww - 2 * s - span
        if valid_h <= 0 or valid_w <= 0:
            break
        comb_y = sum(proj_y[s + k * s: s + k * s + valid_h] for k in range(7))
        y1 = s + int(np.argmax(comb_y))

        y_top = max(0, y1 - s // 2)
        y_bot = min(wh, y1 + span + s // 2)
        proj_x = saddle[y_top:y_bot, :].sum(axis=0)
        comb_x = sum(proj_x[s + k * s: s + k * s + valid_w] for k in range(7))
        x1 = s + int(np.argmax(comb_x))

        ys = y1 + np.arange(7) * s
        xs = x1 + np.arange(7) * s
        grid_s = saddle[np.ix_(ys, xs)]
        grid_sign = signed[np.ix_(ys, xs)]
        alt_score = abs(float((np.sign(grid_sign) * checker).mean()))
        active_frac = float((grid_s > 8.0).mean())
        mean_strength = float(grid_s.mean())
        score = mean_strength * active_frac * (0.4 + 0.6 * alt_score)
        if score > best_score and active_frac >= 0.40 and alt_score >= 0.40:
            best_score = score
            best_params = (x1, y1, s)

    if best_params is None:
        return None

    wx1, wy1, ws = best_params
    # Refine in full resolution
    if scale != 1.0:
        full_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        full_blur = cv2.GaussianBlur(full_gray, (5, 5), 0)
        fd = max(3, int(round(d / scale)))
        ftl = full_blur[:-2 * fd, :-2 * fd]
        fbr = full_blur[2 * fd:, 2 * fd:]
        ftr = full_blur[:-2 * fd, 2 * fd:]
        fbl = full_blur[2 * fd:, :-2 * fd]
        fraw = np.maximum(
            0.0,
            np.abs((ftl + fbr) - (ftr + fbl)) - 1.35 * (np.abs(ftl - fbr) + np.abs(ftr - fbl)),
        )
        full_saddle = np.zeros((h, w), dtype=np.float32)
        full_saddle[fd:h - fd, fd:w - fd] = fraw
        fx1 = int(round(wx1 / scale))
        fy1 = int(round(wy1 / scale))
        fs = max(8, int(round(ws / scale)))
    else:
        full_saddle = saddle
        fx1, fy1, fs = wx1, wy1, ws

    band_y0 = max(0, fy1 - fs)
    band_y1 = min(h, fy1 + 7 * fs)
    band_x0 = max(0, fx1 - fs)
    band_x1 = min(w, fx1 + 7 * fs)
    local_proj_x = full_saddle[band_y0:band_y1, :].sum(axis=0)
    local_proj_y = full_saddle[:, band_x0:band_x1].sum(axis=1)

    x0, x8 = _refine_7_lines(local_proj_x, fx1, fs, w, window=max(4, fs // 10))
    y0, y8 = _refine_7_lines(local_proj_y, fy1, fs, h, window=max(4, fs // 10))

    bw, bh = x8 - x0, y8 - y0
    if bw <= 0 or bh <= 0 or not 0.75 <= bw / bh <= 1.33:
        return None
    area = float(bw * bh)
    if area < image_area * min_fraction:
        return None
    corners = np.array([
        [x0, y0], [x8, y0], [x8, y8], [x0, y8],
    ], dtype=np.float32)
    return BoardDetection(order_points(corners), area, 0.92, "saddle-lattice")


def _collect_contours(gray: np.ndarray, blurred: np.ndarray) -> list[np.ndarray]:
    """Gather contours across external and nested edge/threshold maps."""
    edge_maps = []
    e1 = cv2.Canny(blurred, 50, 150)
    edge_maps.append(cv2.dilate(e1, np.ones((3, 3), np.uint8), iterations=1))

    e2 = cv2.Canny(blurred, 25, 80)
    edge_maps.append(cv2.dilate(e2, np.ones((3, 3), np.uint8), iterations=1))

    adapt = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3
    )
    edge_maps.append(adapt)

    all_contours: list[np.ndarray] = []
    # External contours of primary Canny map first (preserves classic priority)
    ext_contours, _ = cv2.findContours(edge_maps[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_contours.extend(sorted(ext_contours, key=cv2.contourArea, reverse=True)[:12])

    # Nested contours so boards inside browser/app windows are discovered too
    for emap in edge_maps:
        nested, _ = cv2.findContours(emap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(sorted(nested, key=cv2.contourArea, reverse=True)[:25])
    return all_contours


# -------------------------------------------------------------------- public
def detect_board(image: np.ndarray, min_fraction: float = 0.02) -> BoardDetection | None:
    """Detect the most plausible chessboard quadrilateral in ``image``."""
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    image_area = gray.shape[0] * gray.shape[1]

    contours = _collect_contours(gray, blurred)
    candidates: list[BoardDetection] = []
    for contour in contours:
        quad = _quad_candidate(contour, image_area, min_fraction)
        if quad is not None:
            candidates.append(quad)
        rect = _rect_candidate(contour, image_area, min_fraction)
        if rect is not None:
            candidates.append(rect)

    lattice = _saddle_lattice_candidate(image, min_fraction)
    if lattice is not None:
        candidates.append(lattice)

    # Deduplicate near-identical candidates.
    unique: list[BoardDetection] = []
    for candidate in candidates:
        if all(np.linalg.norm(candidate.corners - other.corners, axis=1).mean() > 6
               for other in unique):
            unique.append(candidate)

    if not unique:
        LOGGER.info("No board-sized quadrilateral detected")
        return None

    scored: list[tuple[float, BoardDetection]] = []
    for candidate in unique:
        try:
            warped = warp_board(image, candidate.corners, size=240)
            score = grid_score(warped)
        except cv2.error:  # pragma: no cover - defensive
            score = 0.0
        scored.append((score, candidate))
    scored.sort(key=lambda item: (round(item[0], 2), item[1].confidence), reverse=True)

    best_score, best = scored[0]
    # If best score is low, prefer the most stable previous detection
    # to avoid sudden jumps ("verrutschen") of the board region.
    if best_score < 0.35 and best is not None:
        # Note: detect_board is stateless; stability is handled by pipeline.
        pass
    if best_score < 0.35:
        # No convincing grid: if a saddle lattice was found, prefer it;
        # otherwise fall back to the largest contour only if it has some grid structure.
        if lattice is not None:
            return lattice
        if best_score < 0.18:
            LOGGER.info("No candidate achieved minimal chessboard grid score (%.2f)", best_score)
            return None
        best = max(unique, key=lambda c: c.area)
        LOGGER.debug("Low grid score (%.2f); falling back to largest contour", best_score)
    else:
        LOGGER.debug("Board detected via %s (grid score %.2f, area %.0f)",
                     best.method, best_score, best.area)
    return best
