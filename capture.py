"""
capture.py — Video Capture & Frame Sampling Engine
===================================================
Handles webcam initialisation, FPS auto-detection, timed video recording,
and intelligent keyframe extraction at configurable intervals.

Key design decisions
--------------------
* **Dynamic FPS detection** via `cv2.CAP_PROP_FPS` with a 30 FPS fallback so
  the sampling math stays correct on any camera.
* **Frame-skip sampling** — instead of grabbing every frame and discarding,
  we compute the exact frame indices that land on the 2-second boundaries
  and only decode those, keeping CPU/memory usage minimal.
* All frames are returned as BGR `np.ndarray` objects so downstream code
  can freely use OpenCV *or* convert to RGB for Streamlit display.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CapturedFrame:
    """A single keyframe together with its capture metadata."""
    image: np.ndarray            # BGR image (H × W × 3)
    timestamp: float             # Seconds since start of recording
    index: int                   # Original frame index in the video stream
    resolution: Tuple[int, int]  # (width, height)


@dataclass
class CaptureResult:
    """Aggregated output of a single recording session."""
    frames: List[CapturedFrame] = field(default_factory=list)
    fps: float = 30.0
    total_frames_read: int = 0
    duration_seconds: float = 0.0
    success: bool = False
    error_message: str = ""


# ---------------------------------------------------------------------------
# Webcam helpers
# ---------------------------------------------------------------------------

def detect_fps(cap: cv2.VideoCapture, fallback: float = 30.0) -> float:
    """Return the camera's reported FPS, falling back to *fallback*.

    Many consumer webcams report 0 or an unreasonably low number for
    `CAP_PROP_FPS`, so we clamp to the fallback in those cases.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        return fallback
    return fps


def open_webcam(device_index: int = 0) -> Tuple[Optional[cv2.VideoCapture], str]:
    """Attempt to open the webcam at *device_index*.

    Returns
    -------
    (cap, error_msg)
        *cap* is a valid, opened `VideoCapture` or ``None`` on failure.
        *error_msg* is an empty string on success.
    """
    try:
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None, (
                f"Could not open webcam at index {device_index}. "
                "Ensure the camera is connected and not used by another app."
            )
        return cap, ""
    except Exception as exc:
        return None, f"Webcam initialisation error: {exc}"


# ---------------------------------------------------------------------------
# Core capture & sampling
# ---------------------------------------------------------------------------

def capture_and_sample(
    duration: float = 10.0,
    sample_interval: float = 2.0,
    device_index: int = 0,
    progress_callback=None,
) -> CaptureResult:
    """Record from the webcam for *duration* seconds, extracting one
    keyframe every *sample_interval* seconds.

    Parameters
    ----------
    duration : float
        Total recording time in seconds.
    sample_interval : float
        Seconds between successive keyframe extractions.
    device_index : int
        OpenCV camera device index.
    progress_callback : callable, optional
        ``callback(fraction: float)`` called periodically so the UI can
        show a progress bar.  *fraction* ∈ [0, 1].

    Returns
    -------
    CaptureResult
        Collected keyframes plus session metadata.
    """
    result = CaptureResult()

    cap, err = open_webcam(device_index)
    if cap is None:
        result.error_message = err
        return result

    try:
        fps = detect_fps(cap)
        result.fps = fps

        # Frame indices that correspond to the desired sample times
        # e.g. at 30 FPS, interval=2 → frames 0, 60, 120, 180, 240
        total_frames_target = int(duration * fps)
        frames_per_interval = int(sample_interval * fps)

        sample_indices = set(
            range(0, total_frames_target, max(frames_per_interval, 1))
        )

        frame_idx = 0
        start_time = time.time()

        while frame_idx < total_frames_target:
            ret, frame = cap.read()
            if not ret:
                # Camera stopped delivering frames early
                break

            if frame_idx in sample_indices:
                h, w = frame.shape[:2]
                timestamp = frame_idx / fps
                result.frames.append(
                    CapturedFrame(
                        image=frame.copy(),
                        timestamp=round(timestamp, 2),
                        index=frame_idx,
                        resolution=(w, h),
                    )
                )

            frame_idx += 1
            result.total_frames_read = frame_idx

            # Progress feedback
            if progress_callback and frame_idx % 10 == 0:
                progress_callback(min(frame_idx / total_frames_target, 1.0))

        result.duration_seconds = round(time.time() - start_time, 2)
        result.success = len(result.frames) > 0

        if not result.success:
            result.error_message = (
                "Recording completed but no frames were captured. "
                "The camera may be blocked or returning blank frames."
            )

    except Exception as exc:
        result.error_message = f"Capture error: {exc}"
    finally:
        cap.release()

    return result


# ---------------------------------------------------------------------------
# Video-file sampling (for offline / demo use)
# ---------------------------------------------------------------------------

def sample_from_video_file(
    path: str,
    sample_interval: float = 2.0,
    max_duration: Optional[float] = None,
    progress_callback=None,
) -> CaptureResult:
    """Extract keyframes from an existing video file on disk.

    This mirrors the webcam pipeline but reads from *path* instead of a
    live camera, making it useful for demos and unit-testing.
    """
    result = CaptureResult()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        result.error_message = f"Cannot open video file: {path}"
        return result

    try:
        fps = detect_fps(cap)
        result.fps = fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_duration:
            total_frames = min(total_frames, int(max_duration * fps))

        frames_per_interval = max(int(sample_interval * fps), 1)
        sample_indices = set(range(0, total_frames, frames_per_interval))

        frame_idx = 0
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in sample_indices:
                h, w = frame.shape[:2]
                timestamp = frame_idx / fps
                result.frames.append(
                    CapturedFrame(
                        image=frame.copy(),
                        timestamp=round(timestamp, 2),
                        index=frame_idx,
                        resolution=(w, h),
                    )
                )

            frame_idx += 1
            result.total_frames_read = frame_idx

            if progress_callback and frame_idx % 30 == 0:
                progress_callback(min(frame_idx / total_frames, 1.0))

        result.duration_seconds = round(total_frames / fps, 2)
        result.success = len(result.frames) > 0

        if not result.success:
            result.error_message = "No frames could be decoded from the video."

    except Exception as exc:
        result.error_message = f"Video read error: {exc}"
    finally:
        cap.release()

    return result
