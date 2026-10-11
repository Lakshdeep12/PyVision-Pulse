"""
metrics.py — Multi-Metric Image Comparison & Degradation Analysis
=================================================================
Computes per-frame quality metrics against a user-chosen reference image.

Metrics computed
----------------
1. **Resolution**       — (W × H) dimensions
2. **Brightness**       — Mean pixel intensity of the grayscale image.
                          Formula: μ = (1/N) Σ I(x,y)
3. **Sharpness**        — Variance of the Laplacian.
                          High variance → sharp edges; low → blurry.
                          L = ∇²I,  sharpness = Var(L)
4. **Edge Density**     — Ratio of Canny edge pixels to total pixels.
                          Captures how "detailed" or texture-rich a frame is.
5. **SSIM**             — Structural Similarity Index (Wang et al., 2004).
                          Ranges [−1, 1]; 1 = identical structure.
6. **MSE**              — Mean Squared Error, pixel-level deviation.
                          MSE = (1/N) Σ (I_ref − I_cmp)²
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# scikit-image SSIM — gracefully degrade if unavailable
try:
    from skimage.metrics import structural_similarity as _ssim_fn
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FrameMetrics:
    """Quality metrics for a single frame."""
    frame_label: str               # e.g. "T=0s"
    resolution: str                # "1280×720"
    brightness: float              # Mean grayscale intensity [0–255]
    sharpness: float               # Variance of Laplacian
    edge_density: float            # Ratio of edge pixels (0.0–1.0)
    ssim: Optional[float] = None   # Structural similarity vs reference
    mse: Optional[float] = None    # Mean squared error vs reference
    brightness_delta: float = 0.0  # Change from reference
    sharpness_delta: float = 0.0   # Change from reference


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity of a single-channel (grayscale) image.

    .. math::
        \\mu = \\frac{1}{H \\times W} \\sum_{x,y} I(x,y)
    """
    return float(np.mean(gray))


def compute_sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian — a simple, widely-used focus measure.

    The Laplacian operator approximates the 2nd spatial derivative:
        L(x,y) = ∂²I/∂x² + ∂²I/∂y²

    A well-focused image has many high-frequency components, producing a
    *high* Laplacian variance.  A blurry image has mostly low-frequency
    content, yielding a *low* variance.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_edge_density(
    gray: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> float:
    """Fraction of pixels detected as edges by the Canny detector.

    Canny performs:
      1. Gaussian smoothing
      2. Gradient magnitude via Sobel
      3. Non-maximum suppression
      4. Hysteresis thresholding with *low_threshold* / *high_threshold*

    Returns a ratio ∈ [0, 1].
    """
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_pixels = np.count_nonzero(edges)
    return round(edge_pixels / total_pixels, 6)


def compute_ssim(
    gray_ref: np.ndarray,
    gray_cmp: np.ndarray,
) -> Optional[float]:
    """Structural Similarity Index between two grayscale images.

    SSIM is defined over local windows as:
        SSIM(x, y) = [f(luminance) · f(contrast) · f(structure)]

    This implementation delegates to `skimage.metrics.structural_similarity`,
    which handles the windowed computation, dynamic range, and Gaussian
    weighting.

    Returns ``None`` if scikit-image is not installed or the images cannot
    be compared (e.g. mismatched dimensions).
    """
    if not _HAS_SKIMAGE:
        return None

    try:
        # Ensure both images share the same shape
        if gray_ref.shape != gray_cmp.shape:
            gray_cmp = cv2.resize(
                gray_cmp, (gray_ref.shape[1], gray_ref.shape[0])
            )

        # Determine a valid win_size: must be odd and ≤ smallest dimension
        min_dim = min(gray_ref.shape[0], gray_ref.shape[1])
        win_size = min(7, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3

        score = _ssim_fn(
            gray_ref,
            gray_cmp,
            win_size=win_size,
            data_range=gray_ref.max() - gray_ref.min(),
        )
        return round(float(score), 6)
    except Exception:
        return None


def compute_mse(ref: np.ndarray, cmp: np.ndarray) -> Optional[float]:
    """Mean Squared Error between two images (per-pixel).

    .. math::
        MSE = \\frac{1}{H \\times W} \\sum_{x,y} \\bigl(I_{ref}(x,y)
              - I_{cmp}(x,y)\\bigr)^2

    Returns ``None`` on dimension mismatch that cannot be reconciled.
    """
    try:
        if ref.shape != cmp.shape:
            cmp = cv2.resize(cmp, (ref.shape[1], ref.shape[0]))
        err = np.mean((ref.astype(np.float64) - cmp.astype(np.float64)) ** 2)
        return round(float(err), 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Batch analysis
# ---------------------------------------------------------------------------

def analyse_frame(
    frame_bgr: np.ndarray,
    label: str,
    ref_gray: Optional[np.ndarray] = None,
    ref_brightness: float = 0.0,
    ref_sharpness: float = 0.0,
) -> FrameMetrics:
    """Compute every metric for a single BGR frame.

    Parameters
    ----------
    frame_bgr : np.ndarray
        The source frame in BGR colour space.
    label : str
        Human-readable label, e.g. ``"T=4s"``.
    ref_gray : np.ndarray, optional
        Grayscale version of the reference image for SSIM / MSE.
    ref_brightness, ref_sharpness : float
        Reference values for computing deltas.
    """
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    brightness = compute_brightness(gray)
    sharpness = compute_sharpness(gray)
    edge_density = compute_edge_density(gray)

    ssim_val = compute_ssim(ref_gray, gray) if ref_gray is not None else None
    mse_val = compute_mse(ref_gray, gray) if ref_gray is not None else None

    return FrameMetrics(
        frame_label=label,
        resolution=f"{w}×{h}",
        brightness=round(brightness, 2),
        sharpness=round(sharpness, 2),
        edge_density=edge_density,
        ssim=ssim_val,
        mse=mse_val,
        brightness_delta=round(brightness - ref_brightness, 2),
        sharpness_delta=round(sharpness - ref_sharpness, 2),
    )


def analyse_all_frames(
    frames: list,
    reference_index: int = 0,
) -> Tuple[List[FrameMetrics], Dict]:
    """Run full analysis on every captured frame against a chosen reference.

    Parameters
    ----------
    frames : list[CapturedFrame]
        Ordered list of keyframes from the capture module.
    reference_index : int
        Index into *frames* that the user selected as the reference.

    Returns
    -------
    (metrics_list, summary_dict)
        *metrics_list* — one ``FrameMetrics`` per frame.
        *summary_dict* — aggregated statistics for the session.
    """
    if not frames:
        return [], {}

    ref_frame = frames[reference_index]
    ref_gray = cv2.cvtColor(ref_frame.image, cv2.COLOR_BGR2GRAY)
    ref_brightness = compute_brightness(ref_gray)
    ref_sharpness = compute_sharpness(ref_gray)

    metrics: List[FrameMetrics] = []
    for idx, f in enumerate(frames):
        label = f"T={f.timestamp}s"
        m = analyse_frame(
            f.image,
            label=label,
            ref_gray=ref_gray,
            ref_brightness=ref_brightness,
            ref_sharpness=ref_sharpness,
        )
        metrics.append(m)

    # Session-level summary
    ssim_values = [m.ssim for m in metrics if m.ssim is not None]
    mse_values = [m.mse for m in metrics if m.mse is not None]

    summary = {
        "total_frames": len(metrics),
        "reference": metrics[reference_index].frame_label,
        "avg_brightness": round(np.mean([m.brightness for m in metrics]), 2),
        "avg_sharpness": round(np.mean([m.sharpness for m in metrics]), 2),
        "avg_ssim": round(np.mean(ssim_values), 4) if ssim_values else None,
        "avg_mse": round(np.mean(mse_values), 4) if mse_values else None,
        "min_ssim": round(min(ssim_values), 4) if ssim_values else None,
        "max_mse": round(max(mse_values), 4) if mse_values else None,
    }

    return metrics, summary
