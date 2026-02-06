"""Classical CV heuristic orientation detector (metadata-agnostic)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ..constants import CARDINAL_ANGLES, WHITE_RGB


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256, dtype=np.float64), hist)

    sum_background = 0.0
    weight_background = 0.0
    max_between = -1.0
    threshold = 127

    for idx in range(256):
        weight_background += hist[idx]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += idx * hist[idx]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if between > max_between:
            max_between = between
            threshold = idx

    return int(threshold)


def _ink_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    threshold = _otsu_threshold(gray)
    mask = gray < threshold
    if mask.mean() < 0.005:
        # Keep sparse pages from collapsing to all-empty by relaxing threshold.
        mask = gray < min(threshold + 15, 245)
    return mask.astype(np.float32)


def _projection_axis_score(mask: np.ndarray) -> float:
    row_profile = mask.sum(axis=1)
    col_profile = mask.sum(axis=0)
    row_var = float(np.var(row_profile))
    col_var = float(np.var(col_profile))
    denom = row_var + col_var + 1e-6
    return (row_var - col_var) / denom


def _line_centroid_score(mask: np.ndarray) -> float:
    height, _ = mask.shape
    row_profile = mask.sum(axis=1)

    smooth_window = max(5, height // 180)
    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
    smoothed = np.convolve(row_profile, kernel, mode="same")

    threshold = max(np.percentile(smoothed, 70) * 0.2, 1.0)
    active = smoothed > threshold

    min_band_height = max(6, height // 220)
    weighted_offsets: list[tuple[float, float]] = []

    start = None
    for idx, is_active in enumerate(active):
        if is_active and start is None:
            start = idx
        elif not is_active and start is not None:
            end = idx
            if end - start >= min_band_height:
                line = mask[start:end, :]
                mass = float(line.sum())
                if mass > 0:
                    y_coords = np.arange(line.shape[0], dtype=np.float32)[:, None]
                    centroid = float((y_coords * line).sum() / mass)
                    normalized = centroid / max(line.shape[0] - 1, 1)
                    weighted_offsets.append((0.5 - normalized, mass))
            start = None

    if start is not None:
        end = len(active)
        if end - start >= min_band_height:
            line = mask[start:end, :]
            mass = float(line.sum())
            if mass > 0:
                y_coords = np.arange(line.shape[0], dtype=np.float32)[:, None]
                centroid = float((y_coords * line).sum() / mass)
                normalized = centroid / max(line.shape[0] - 1, 1)
                weighted_offsets.append((0.5 - normalized, mass))

    if not weighted_offsets:
        return 0.0

    total_mass = sum(weight for _, weight in weighted_offsets)
    return sum(offset * weight for offset, weight in weighted_offsets) / max(total_mass, 1e-6)


def _global_upright_score(mask: np.ndarray) -> float:
    row_profile = mask.sum(axis=1)
    mass = float(row_profile.sum())
    if mass <= 0:
        return 0.0
    y_coords = np.arange(mask.shape[0], dtype=np.float32)
    centroid = float((y_coords * row_profile).sum() / mass)
    normalized = centroid / max(mask.shape[0] - 1, 1)
    return 0.5 - normalized


def _candidate_score(image: Image.Image, candidate_rotation: int) -> tuple[float, dict[str, float]]:
    # candidate_rotation means "page is rotated clockwise by this amount".
    corrected = image.rotate(
        candidate_rotation,
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor=WHITE_RGB,
    )

    mask = _ink_mask(corrected)
    axis_score = _projection_axis_score(mask)
    line_score = _line_centroid_score(mask)
    global_score = _global_upright_score(mask)
    upright_score = 0.75 * line_score + 0.25 * global_score
    total = axis_score + 0.85 * upright_score
    details = {
        "axis_score": axis_score,
        "line_centroid_score": line_score,
        "global_upright_score": global_score,
        "upright_score": upright_score,
        "total_score": total,
    }
    return total, details


def predict_rotation_heuristic(image: Image.Image) -> dict:
    """Predict page clockwise orientation among 0/90/180/270."""
    scores: dict[int, float] = {}
    breakdown: dict[int, dict[str, float]] = {}

    for angle in CARDINAL_ANGLES:
        score, details = _candidate_score(image, angle)
        scores[angle] = score
        breakdown[angle] = details

    ordered = np.array([scores[a] for a in CARDINAL_ANGLES], dtype=np.float64)
    temperature = 0.25
    logits = ordered / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum() + 1e-12

    best_idx = int(np.argmax(probs))
    predicted = CARDINAL_ANGLES[best_idx]

    return {
        "predicted_rotation_deg": predicted,
        "confidence": float(probs[best_idx]),
        "probabilities": {str(angle): float(probs[idx]) for idx, angle in enumerate(CARDINAL_ANGLES)},
        "scores": {str(angle): float(scores[angle]) for angle in CARDINAL_ANGLES},
        "details": {str(angle): breakdown[angle] for angle in CARDINAL_ANGLES},
    }
