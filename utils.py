"""
utils.py — Shared Utilities
============================
Colour-space conversion, image formatting helpers, and visualisation aids
used across the capture and UI layers.
"""

from __future__ import annotations

import cv2
import numpy as np


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR OpenCV image to RGB for Streamlit / Matplotlib display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_for_display(
    image: np.ndarray,
    max_width: int = 640,
) -> np.ndarray:
    """Down-scale *image* so its width does not exceed *max_width*,
    preserving aspect ratio.  Returns the original if already small enough.
    """
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def overlay_timestamp_badge(
    image: np.ndarray,
    text: str,
    position: str = "top-left",
) -> np.ndarray:
    """Burn a translucent timestamp badge onto *image* (in-place).

    Parameters
    ----------
    image : np.ndarray
        BGR or RGB image.
    text : str
        Label text, e.g. ``"T=4s"``.
    position : str
        One of ``"top-left"``, ``"top-right"``, ``"bottom-left"``,
        ``"bottom-right"``.

    Returns
    -------
    np.ndarray
        The same image array (modified in-place for efficiency).
    """
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    padding = 10

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    bw, bh = tw + 2 * padding, th + 2 * padding + baseline

    h, w = img.shape[:2]
    if position == "top-left":
        x, y = 8, 8
    elif position == "top-right":
        x, y = w - bw - 8, 8
    elif position == "bottom-left":
        x, y = 8, h - bh - 8
    else:
        x, y = w - bw - 8, h - bh - 8

    # Semi-transparent dark background
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    # White text
    text_x = x + padding
    text_y = y + padding + th
    cv2.putText(img, text, (text_x, text_y), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

    return img


def generate_diff_heatmap(
    ref_bgr: np.ndarray,
    cmp_bgr: np.ndarray,
) -> np.ndarray:
    """Create a colour-mapped absolute-difference heatmap between two frames.

    Bright (hot) regions indicate large pixel differences; cool regions
    are structurally similar.

    Both images are resized to the reference dimensions if they differ.
    """
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    cmp_gray = cv2.cvtColor(cmp_bgr, cv2.COLOR_BGR2GRAY)

    if ref_gray.shape != cmp_gray.shape:
        cmp_gray = cv2.resize(cmp_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    diff = cv2.absdiff(ref_gray, cmp_gray)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    return heatmap


def generate_edge_overlay(gray: np.ndarray) -> np.ndarray:
    """Return a green-on-black Canny edge visualisation."""
    edges = cv2.Canny(gray, 50, 150)
    canvas = np.zeros((*gray.shape, 3), dtype=np.uint8)
    canvas[edges > 0] = (0, 255, 0)  # bright green edges
    return canvas
