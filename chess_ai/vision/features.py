"""Per-square feature extraction for piece classification."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MASK_SIZE = 64
OCCUPANCY_THRESHOLD = 0.05
# A real piece silhouette covers a substantial share of its square; smaller
# surviving blobs are sensor/compression noise rather than pieces.
MIN_COMPONENT_FRACTION = 0.06
# Minimum Otsu separation (grey levels) between square and piece outline.
OTSU_NOISE_FLOOR = 15.0


@dataclass
class CellFeatures:
    """Shape and colour descriptors of one square."""

    occupied: bool
    occupancy_ratio: float = 0.0
    piece_luminance: float = 0.0
    background_luminance: float = 0.0
    mask: np.ndarray | None = None          # normalized filled mask (64x64)
    h_ratio: float = 0.0
    w_ratio: float = 0.0
    aspect: float = 0.0
    extent: float = 0.0
    solidity: float = 0.0
    top_width_ratio: float = 0.0
    width_at_5: float = 0.0
    width_at_15: float = 0.0
    width_at_35: float = 0.0
    width_at_55: float = 0.0
    waist_ratio: float = 0.0
    upper_mass_ratio: float = 0.0
    symmetry: float = 0.0
    hole_count: int = 0
    top_transitions: int = 0
    center_protrusion: float = 0.0
    crossbar_ratio: float = 0.0
    top_offset: float = 0.0
    centroid_x: float = 0.0
    centroid_y: float = 0.0
    confidence: float = 0.5


def _background_color(cell: np.ndarray) -> np.ndarray:
    h, w = cell.shape[:2]
    patch = max(2, h // 7)
    corners = np.concatenate([
        cell[:patch, :patch].reshape(-1, 3),
        cell[:patch, -patch:].reshape(-1, 3),
        cell[-patch:, :patch].reshape(-1, 3),
        cell[-patch:, -patch:].reshape(-1, 3),
    ])
    return np.median(corners, axis=0)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the largest central connected component and its area fraction.

    Requiring overlap with the central [0.22..0.78] region of the square
    rejects rank/file coordinate labels ('1'-'8', 'a'-'h') printed in square
    corners on Chess.com and Lichess.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask, 0.0
    h, w = mask.shape[:2]
    cy0, cy1 = int(h * 0.22), max(int(h * 0.22) + 1, int(h * 0.78))
    cx0, cx1 = int(w * 0.22), max(int(w * 0.22) + 1, int(w * 0.78))
    central_labels = set(np.unique(labels[cy0:cy1, cx0:cx1])) - {0}
    if not central_labels:
        return np.zeros_like(mask), 0.0
    largest = max(central_labels, key=lambda lbl: int(stats[int(lbl), cv2.CC_STAT_AREA]))
    area = float(stats[int(largest), cv2.CC_STAT_AREA])
    component = (labels == largest).astype(np.uint8) * 255
    return component, area / float(mask.size)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _count_holes(raw_mask: np.ndarray) -> int:
    """Count enclosed background components (border components are exterior)."""
    inverted = cv2.bitwise_not(raw_mask)
    count, labels = cv2.connectedComponents(inverted, 8)
    border_labels = set(labels[0, :]) | set(labels[-1, :]) | \
        set(labels[:, 0]) | set(labels[:, -1])
    return sum(1 for label in range(1, count) if label not in border_labels)


def _column_tops(mask: np.ndarray) -> np.ndarray:
    """Topmost occupied row per column, MASK_SIZE when the column is empty."""
    tops = np.full(MASK_SIZE, MASK_SIZE, dtype=np.int32)
    for col in range(MASK_SIZE):
        rows = np.where(mask[:, col] > 0)[0]
        if rows.size:
            tops[col] = rows[0]
    return tops


def _row_width(mask: np.ndarray, row_fraction: float, y0: int, y1: int) -> float:
    """Occupied width at a relative height inside the bounding box."""
    row = y0 + int(row_fraction * max(1, y1 - y0))
    row = min(MASK_SIZE - 1, max(0, row))
    return float((mask[row] > 0).sum()) / MASK_SIZE


def extract_cell_features(cell: np.ndarray) -> CellFeatures:
    """Extract descriptors from a single square image."""
    h, w = cell.shape[:2]
    margin_y, margin_x = int(h * 0.10), int(w * 0.10)
    inner = cell[margin_y:h - margin_y, margin_x:w - margin_x]
    background = _background_color(inner)

    diff = cv2.absdiff(inner, np.full_like(inner, background.astype(np.uint8)))
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    otsu_value, mask = cv2.threshold(
        diff_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Noise-only squares produce a near-zero Otsu level (a few grey levels of
    # sensor/compression noise); real piece outlines always separate much more
    # strongly. This floor reliably rejects false pieces on noisy empty squares.
    if otsu_value < OTSU_NOISE_FLOOR:
        return CellFeatures(
            occupied=False,
            occupancy_ratio=float(mask.mean() / 255.0),
            background_luminance=float(cv2.cvtColor(
                np.full((1, 1, 3), background.astype(np.uint8), dtype=np.uint8),
                cv2.COLOR_BGR2HSV)[0, 0, 2]),
        )
    # Median blur removes salt-and-pepper noise without eroding thin parts
    # (mitres, ears, crosses) the way morphological opening would.
    mask = cv2.medianBlur(mask, 3)
    # A strong close bridges narrow gaps so outline fragments (cross stems,
    # crown tips) stay attached to the main silhouette.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    occupancy = float(mask.mean() / 255.0)
    bg_hsv = cv2.cvtColor(
        np.full((1, 1, 3), background.astype(np.uint8), dtype=np.uint8),
        cv2.COLOR_BGR2HSV)[0, 0, 2]
    if occupancy < OCCUPANCY_THRESHOLD:
        return CellFeatures(occupied=False, occupancy_ratio=occupancy,
                            background_luminance=float(bg_hsv))

    # Keep the largest component and measure its REAL area (before hole
    # filling) so noise blobs cannot inflate themselves into fake pieces.
    component, component_fraction = _largest_component(mask)
    if component_fraction < MIN_COMPONENT_FRACTION:
        return CellFeatures(occupied=False, occupancy_ratio=component_fraction,
                            background_luminance=float(bg_hsv))
    # Fill the interior of outline-only silhouettes for shape descriptors.
    filled = _fill_holes(component)
    holes = _count_holes(filled)

    # Sample colour around the silhouette's centroid: even when low contrast
    # reduces the mask to an outline ring, the centroid always sits inside the
    # piece's fill area, avoiding the contrasting outline colour.
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    center_moments = cv2.moments(filled)
    if center_moments["m00"]:
        cx = int(center_moments["m10"] / center_moments["m00"])
        cy = int(center_moments["m01"] / center_moments["m00"])
        radius = max(2, filled.shape[0] // 12)
        x0, x1 = max(0, cx - radius), min(filled.shape[1], cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(filled.shape[0], cy + radius + 1)
        color_pixels = value[y0:y1, x0:x1][filled[y0:y1, x0:x1] > 0]
    else:
        color_pixels = np.array([], dtype=np.uint8)
    if color_pixels.size < 8:  # fallback: deep interior via distance transform
        distance = cv2.distanceTransform(filled, cv2.DIST_L2, 3)
        color_pixels = value[distance >= 2.0]
    if color_pixels.size < 8:  # fallback: eroded mask
        eroded = cv2.erode(filled, np.ones((3, 3), np.uint8), iterations=1)
        color_pixels = value[eroded > 0]
    if color_pixels.size < 8:
        color_pixels = value[filled > 0]
    luminance = float(np.median(color_pixels)) if color_pixels.size else 0.0
    # Relative cross-check: a piece whose fill is close to its square colour
    # and not clearly lighter is treated as dark (handles dark pieces on
    # low-contrast dark squares where only the light outline survives masking).
    if luminance >= bg_hsv and abs(luminance - bg_hsv) < 35:
        luminance = bg_hsv - 40

    normalized = cv2.resize(filled, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_NEAREST)
    # Strip isolated 1-2px halo spikes at the very top of the normalized mask
    # so they cannot skew y0, top_width_ratio or top_offset.
    for _ in range(6):
        occ_rows = np.where((normalized > 0).any(axis=1))[0]
        if occ_rows.size < 4:
            break
        r0 = int(occ_rows[0])
        w0 = int((normalized[r0] > 0).sum())
        w2 = int((normalized[min(MASK_SIZE - 1, r0 + 2)] > 0).sum())
        if w0 <= 2 and w2 <= 3:
            normalized[r0, :] = 0
        else:
            break
    ys, xs = np.where(normalized > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
    area = float((normalized > 0).sum())

    contours, _ = cv2.findContours(normalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(contour))))
    solidity = float(np.clip(area / hull_area, 0, 1))

    top_band = normalized[y0:y0 + max(1, bh // 8), x0:x1 + 1]
    top_width_ratio = float((top_band.sum(axis=0) > 0).mean())
    upper_mass_ratio = float(
        (normalized[: MASK_SIZE // 3] > 0).sum() / max(1.0, area))
    flipped = cv2.flip(normalized, 1)
    symmetry = 1.0 - float(np.count_nonzero(normalized != flipped)) / float(MASK_SIZE ** 2)

    tops = _column_tops(normalized)
    active = tops[tops < MASK_SIZE]
    top_transitions = int(np.count_nonzero(np.diff(active) > 2)) if active.size > 2 else 0

    # King cross: centre columns extend above their immediate neighbours and a
    # horizontal bar widens the top region.
    center_cols = slice(MASK_SIZE // 2 - 4, MASK_SIZE // 2 + 4)
    neighbor_cols = np.r_[
        MASK_SIZE // 2 - 9: MASK_SIZE // 2 - 4,
        MASK_SIZE // 2 + 4: MASK_SIZE // 2 + 9,
    ]
    center_top = float(tops[center_cols].min())
    neighbor_top = float(np.median(tops[neighbor_cols]))
    center_protrusion = max(0.0, neighbor_top - center_top) / MASK_SIZE
    top_region = normalized[: MASK_SIZE // 4, :]
    row_widths = (top_region > 0).sum(axis=1)
    crossbar_ratio = float(row_widths.max()) / MASK_SIZE if row_widths.size else 0.0

    # Horizontal offset of the topmost pixels: an off-centre top identifies
    # the knight's snout even when its outline is partially lost. Averaging
    # over the top few occupied rows avoids single-pixel anti-aliasing jitter.
    top_rows = np.where((normalized > 0).any(axis=1))[0]
    if top_rows.size:
        _, top_cols = np.where(normalized[top_rows[:4]] > 0)
        top_center_x = float(top_cols.mean()) if top_cols.size else MASK_SIZE / 2.0
    else:
        top_center_x = MASK_SIZE / 2.0
    top_offset = abs(top_center_x - MASK_SIZE / 2.0) / MASK_SIZE

    # Narrowest row in the middle band: queen/pawn bodies thin out, rook
    # shafts stay wide.
    middle = normalized[int(MASK_SIZE * 0.42): int(MASK_SIZE * 0.78), :]
    row_occupancy = (middle > 0).sum(axis=1)
    waist_ratio = float(row_occupancy.min()) / MASK_SIZE if row_occupancy.size else 0.0

    moments = cv2.moments(normalized)
    centroid_x = float(moments["m10"] / moments["m00"]) / MASK_SIZE if moments["m00"] else 0.5
    centroid_y = float(moments["m01"] / moments["m00"]) / MASK_SIZE if moments["m00"] else 0.5

    return CellFeatures(
        occupied=True,
        occupancy_ratio=occupancy,
        piece_luminance=luminance,
        background_luminance=float(bg_hsv),
        mask=normalized,
        h_ratio=bh / MASK_SIZE,
        w_ratio=bw / MASK_SIZE,
        aspect=bh / float(bw),
        extent=area / float(bw * bh),
        solidity=solidity,
        top_width_ratio=top_width_ratio,
        width_at_5=_row_width(normalized, 0.05, y0, y1),
        width_at_15=_row_width(normalized, 0.15, y0, y1),
        width_at_35=_row_width(normalized, 0.35, y0, y1),
        width_at_55=_row_width(normalized, 0.55, y0, y1),
        waist_ratio=waist_ratio,
        upper_mass_ratio=upper_mass_ratio,
        symmetry=float(np.clip(symmetry, 0, 1)),
        hole_count=holes,
        top_transitions=top_transitions,
        center_protrusion=center_protrusion,
        crossbar_ratio=crossbar_ratio,
        top_offset=top_offset,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
    )
