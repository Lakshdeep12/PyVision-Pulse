"""
app.py — FrameForensics: Production Video Degradation Analyzer
==============================================================
Automated video capture, keyframe sampling, interactive reference selection,
and multi-metric structural degradation analysis (SSIM, MSE, sharpness,
brightness, edge density).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from capture import (
    CaptureResult,
    capture_and_sample,
    sample_from_video_file,
)
from metrics import FrameMetrics, analyse_all_frames
from utils import (
    bgr_to_rgb,
    generate_diff_heatmap,
    generate_edge_overlay,
    overlay_timestamp_badge,
    resize_for_display,
)


# ─── Page Configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="FrameForensics — Video Degradation Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Design Tokens & Theme Constants ────────────────────────────────────
SAMPLE_VIDEO = os.path.join("video", "Indian_classroom_video.mp4")
SSIM_CONSISTENT = 0.95
SSIM_DEGRADED = 0.80

C = {
    "canvas": "#0A0E17",
    "panel": "#111827",
    "panel_elevated": "#1E293B",
    "ink": "#F8FAFC",
    "muted": "#94A3B8",
    "line": "#334155",
    "line_light": "rgba(255, 255, 255, 0.08)",
    "accent": "#6366F1",
    "accent_cyan": "#06B6D4",
    "good": "#10B981",
    "fair": "#F59E0B",
    "poor": "#EF4444",
}


def _streamlit_version() -> tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:
        return (0, 0)


# Support modern Streamlit width="stretch" or legacy use_container_width
STRETCH: dict = (
    {"width": "stretch"}
    if _streamlit_version() >= (1, 50)
    else {"use_container_width": True}
)


# ─── CSS Injection ──────────────────────────────────────────────────────
def _clean(markup: str) -> str:
    """Strip indentation and blank lines so Markdown never treats HTML as code blocks."""
    return "\n".join(l.strip() for l in markup.strip().splitlines() if l.strip())


def _html(markup: str) -> None:
    st.markdown(_clean(markup), unsafe_allow_html=True)


_html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {{
    --canvas: {C['canvas']};
    --panel: {C['panel']};
    --panel-elevated: {C['panel_elevated']};
    --ink: {C['ink']};
    --muted: {C['muted']};
    --line: {C['line']};
    --accent: {C['accent']};
    --accent-cyan: {C['accent_cyan']};
    --good: {C['good']};
    --fair: {C['fair']};
    --poor: {C['poor']};
}}

/* Global Base */
.stApp {{
    background: radial-gradient(circle at 50% 0%, #151C2C 0%, #0A0E17 80%) !important;
    color: var(--ink) !important;
}}

.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3,
.stMarkdown, .stCaption, [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
.stButton button, .stTabs [data-baseweb="tab"], .stDownloadButton button {{
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
}}

code, pre, .font-mono {{
    font-family: 'JetBrains Mono', monospace !important;
}}

.block-container {{
    max-width: 1360px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}}

header[data-testid="stHeader"] {{
    background: transparent !important;
}}

#MainMenu, footer {{
    visibility: hidden;
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background: #0D111D !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}}

.side-title {{
    color: #F8FAFC !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    margin: 1.2rem 0 0.5rem 0 !important;
    display: flex;
    align-items: center;
    gap: 6px;
}}

/* Hero Header */
.ff-hero {{
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}}

.ff-hero::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #6366F1, #06B6D4, #10B981);
}}

.ff-hero-title {{
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0 0 0.4rem 0;
    background: linear-gradient(135deg, #FFFFFF 30%, #C7D2FE 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: flex;
    align-items: center;
    gap: 12px;
}}

.ff-hero-sub {{
    color: #94A3B8;
    font-size: 0.98rem;
    margin: 0;
    line-height: 1.5;
}}


/* Section Headings */
.ff-section {{
    margin: 2rem 0 0.9rem 0;
    display: flex;
    flex-direction: column;
}}

.ff-section-title {{
    font-size: 1.28rem;
    font-weight: 750;
    letter-spacing: -0.02em;
    color: #F8FAFC;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
}}

.ff-section-hint {{
    margin: 0.2rem 0 0 0;
    color: #94A3B8;
    font-size: 0.88rem;
}}

/* Capture Plan Card */
.ff-plan-box {{
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}}

.ff-plan {{
    display: flex;
    flex-wrap: wrap;
    gap: 12px 36px;
    margin: 0 0 1rem 0;
}}

.ff-plan dt {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin: 0;
}}

.ff-plan dd {{
    font-size: 1.05rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 2px 0 0 0;
}}

/* Status Chips / Badges */
.chip {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    white-space: nowrap;
}}

.chip-ref {{
    background: rgba(99, 102, 241, 0.2);
    color: #A5B4FC;
    border: 1px solid rgba(99, 102, 241, 0.5);
    box-shadow: 0 0 10px rgba(99, 102, 241, 0.3);
}}

.chip-good {{
    background: rgba(16, 185, 129, 0.2);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.5);
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.25);
}}

.chip-fair {{
    background: rgba(245, 158, 11, 0.2);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.5);
    box-shadow: 0 0 10px rgba(245, 158, 11, 0.25);
}}

.chip-poor {{
    background: rgba(239, 68, 68, 0.22);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.55);
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.3);
}}

/* Frame Meta */
.ff-meta {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin: 6px 0 8px 0;
}}

.ff-meta .name {{
    font-weight: 700;
    color: #F8FAFC;
    font-size: 0.9rem;
}}

.ff-meta .time {{
    color: #94A3B8;
    font-size: 0.84rem;
    font-variant-numeric: tabular-nums;
}}

/* Empty Slots */
.ff-slots {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 12px;
    margin: 0.8rem 0;
}}

.ff-slot {{
    aspect-ratio: 16 / 9;
    border: 1.5px dashed #334155;
    border-radius: 8px;
    background: rgba(17, 24, 39, 0.4);
    display: grid;
    place-items: center;
    color: #64748B;
    font-size: 0.85rem;
    font-weight: 600;
}}

/* Verdict Banner */
.ff-verdict {{
    background: rgba(17, 24, 39, 0.8);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-left: 6px solid #6366F1;
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin: 0.8rem 0 1.2rem 0;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}}

.ff-verdict h3 {{
    font-size: 1.3rem;
    font-weight: 750;
    letter-spacing: -0.02em;
    margin: 0 0 0.3rem 0;
    color: #F8FAFC;
}}

.ff-verdict p {{
    margin: 0;
    color: #94A3B8;
    font-size: 0.93rem;
    line-height: 1.5;
}}

.ff-verdict.v-good {{
    border-left-color: #10B981;
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.12) 0%, rgba(17, 24, 39, 0.8) 100%);
}}
.ff-verdict.v-good h3 {{ color: #34D399; }}

.ff-verdict.v-fair {{
    border-left-color: #F59E0B;
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.12) 0%, rgba(17, 24, 39, 0.8) 100%);
}}
.ff-verdict.v-fair h3 {{ color: #FBBF24; }}

.ff-verdict.v-poor {{
    border-left-color: #EF4444;
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.14) 0%, rgba(17, 24, 39, 0.8) 100%);
}}
.ff-verdict.v-poor h3 {{ color: #F87171; }}

/* Streamlit Native Overrides for Clean Dark Contrast */
.stButton > button {{
    border-radius: 9px !important;
    font-weight: 650 !important;
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #F8FAFC !important;
    transition: all 0.2s ease !important;
}}

.stButton > button:hover {{
    background: rgba(51, 65, 85, 0.8) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
    box-shadow: 0 0 14px rgba(99, 102, 241, 0.2) !important;
}}

.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid #818CF8 !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
}}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
    box-shadow: 0 6px 22px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-1px);
}}

/* Active Reference Frame Button */
.stButton > button:disabled {{
    background: rgba(99, 102, 241, 0.25) !important;
    color: #A5B4FC !important;
    border: 1px solid rgba(99, 102, 241, 0.5) !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}}

/* Metric Cards */
[data-testid="stMetric"] {{
    background: rgba(17, 24, 39, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25) !important;
}}

[data-testid="stMetricLabel"] p {{
    color: #94A3B8 !important;
    font-size: 0.86rem !important;
    font-weight: 600 !important;
}}

[data-testid="stMetricValue"] {{
    color: #F8FAFC !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
}}

.stTabs [data-baseweb="tab"] {{
    padding: 10px 18px !important;
    font-weight: 650 !important;
    color: #94A3B8 !important;
    border-radius: 8px 8px 0 0 !important;
}}

.stTabs [aria-selected="true"] {{
    color: #F8FAFC !important;
    background: rgba(99, 102, 241, 0.12) !important;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background: #6366F1 !important;
    height: 3px !important;
    box-shadow: 0 0 10px #6366F1 !important;
}}

/* Image Outlines and Captions */
[data-testid="stImage"] img {{
    border-radius: 6px !important;
}}

[data-testid="stImageCaption"] {{
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}}

/* Border Wrapper Container */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: rgba(17, 24, 39, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}}

@media (max-width: 640px) {{
    .ff-hero-title {{ font-size: 1.6rem; }}
}}
</style>
""")


# ─── Session State Initialization ───────────────────────────────────────
_DEFAULTS = {
    "capture_result": None,
    "reference_idx": 0,
    "analysis_done": False,
    "metrics_list": [],
    "summary": {},
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def set_reference(idx: int) -> None:
    """Callback to switch reference frame and trigger re-analysis."""
    st.session_state.reference_idx = idx
    st.session_state.analysis_done = False


def clear_results() -> None:
    for key, value in _DEFAULTS.items():
        st.session_state[key] = value


# ─── Visual & Formatting Helpers ─────────────────────────────────────────
def status_of(ssim: float | None) -> tuple[str, str]:
    """Return (css_key, display_label) based on SSIM thresholding."""
    if ssim is None:
        return "ref", "Reference"
    if ssim >= SSIM_CONSISTENT:
        return "good", "Consistent"
    if ssim >= SSIM_DEGRADED:
        return "fair", "Drifting"
    return "poor", "Degraded"


STATUS_COLOR = {
    "ref": C["accent"],
    "good": C["good"],
    "fair": C["fair"],
    "poor": C["poor"],
    "neutral": C["panel_elevated"],
}


def chip(key: str, text: str) -> str:
    icons = {
        "ref": "★",
        "good": "●",
        "fair": "▲",
        "poor": "■",
    }
    icon = icons.get(key, "")
    return f'<span class="chip chip-{key}">{icon} {html.escape(text)}</span>'


def fmt_t(t) -> str:
    try:
        return f"{float(t):g}s"
    except (TypeError, ValueError):
        return f"{t}s"


def fmt_res(r) -> str:
    if isinstance(r, (tuple, list)) and len(r) >= 2:
        return f"{r[0]}×{r[1]}"
    return str(r)


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def framed(img_rgb: np.ndarray, color_hex: str, px: int = 5) -> np.ndarray:
    """Draw a clean solid border around the frame indicating its status."""
    img = np.ascontiguousarray(img_rgb)
    return cv2.copyMakeBorder(
        img, px, px, px, px, cv2.BORDER_CONSTANT, value=hex_to_rgb(color_hex)
    )


def section(title: str, hint: str | None = None) -> None:
    hint_html = f'<p class="ff-section-hint">{html.escape(hint)}</p>' if hint else ""
    _html(f"""
    <div class="ff-section">
        <h2 class="ff-section-title">{html.escape(title)}</h2>
        {hint_html}
    </div>
    """)




# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    _html('<p class="side-title">Input Source</p>')
    source_mode = st.radio(
        "Input source",
        ["Video file", "Webcam (Live)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    is_webcam = "Webcam" in source_mode

    uploaded_file = None
    video_path: str | None = None
    if is_webcam:
        st.caption("Captures live video from your laptop camera.")
    else:
        upload_mode = st.radio(
            "Video mode",
            ["Sample video", "Upload video"],
            horizontal=True,
            label_visibility="collapsed",
        )
        if upload_mode == "Upload video":
            uploaded_file = st.file_uploader(
                "Upload a video file",
                type=["mp4", "avi", "mov", "mkv"],
                label_visibility="collapsed",
            )
        elif os.path.exists(SAMPLE_VIDEO):
            video_path = SAMPLE_VIDEO
            st.caption(f"Loaded: `{os.path.basename(SAMPLE_VIDEO)}`")
        else:
            st.warning(f"Sample video not found at `{SAMPLE_VIDEO}`.")

    _html('<p class="side-title">⏱ Sampling Parameters</p>')
    duration = st.slider(
        "Recording duration (s)" if is_webcam else "Analyze duration (s)",
        min_value=4, max_value=60, value=10, step=2,
        help="Total duration to process from the stream or video.",
    )
    sample_interval = st.slider(
        "Sample interval (s)",
        min_value=1, max_value=10, value=2, step=1,
        help="Extract exactly 1 keyframe every N seconds.",
    )
    expected_frames = max(1, duration // sample_interval)
    st.caption(f"Will extract ~{expected_frames} keyframes")

    _html('<p class="side-title">🎨 Visualizations</p>')
    show_badges = st.checkbox("Timestamp badges on frames", value=True)
    show_heatmaps = st.checkbox("Difference heatmaps", value=True)
    show_edges = st.checkbox("Canny edge overlays", value=False)
    gallery_cols = st.slider("Gallery columns", 2, 6, 4)

    st.markdown("---")
    st.button("🔄 Reset / Clear", on_click=clear_results, **STRETCH)


# ═══════════════════════════════════════════════════════════════════════
#  HERO HEADER & STEPPER
# ═══════════════════════════════════════════════════════════════════════
_html("""
<div class="ff-hero">
    <div class="ff-hero-title">
        <span>🔬 PyVision </span>
    </div>
    <p class="ff-hero-sub">
        Automated Video Capture · Frame Sampling · Multi-Metric Structural Degradation Analysis
    </p>
</div>
""")


# ═══════════════════════════════════════════════════════════════════════
#  STEP 1 — CAPTURE & SAMPLE
# ═══════════════════════════════════════════════════════════════════════
section("1. Capture & Sampling Pipeline", "Extract keyframes at precise 2-second boundaries.")

if is_webcam:
    source_label = "Webcam (Live)"
    can_capture = True
elif uploaded_file is not None:
    source_label = uploaded_file.name
    can_capture = True
elif video_path:
    source_label = os.path.basename(video_path)
    can_capture = True
else:
    source_label = "None selected"
    can_capture = False

_html(f"""
<div class="ff-plan-box">
    <dl class="ff-plan">
        <div><dt>Source Stream</dt><dd>{html.escape(source_label)}</dd></div>
        <div><dt>Capture Duration</dt><dd>{duration} seconds</dd></div>
        <div><dt>Sampling Rate</dt><dd>1 frame / {sample_interval}s</dd></div>
        <div><dt>Target Keyframes</dt><dd>~{expected_frames} frames</dd></div>
    </dl>
</div>
""")

btn_col, _spacer = st.columns([1, 2])
with btn_col:
    capture_clicked = st.button(
        "🚀 Start Live Capture" if is_webcam else "📂 Load & Sample Video",
        type="primary",
        disabled=not can_capture,
        **STRETCH,
    )
if not can_capture:
    st.caption("Upload a video or select sample video in the sidebar to proceed.")

if capture_clicked:
    progress_bar = st.progress(0.0, text="Initializing capture pipeline…")

    def _update_progress(frac: float) -> None:
        frac = min(max(float(frac), 0.0), 1.0)
        progress_bar.progress(frac, text=f"Processing frames… {frac * 100:.0f}%")

    if is_webcam:
        result = capture_and_sample(
            duration=duration,
            sample_interval=sample_interval,
            progress_callback=_update_progress,
        )
    else:
        path = video_path
        if uploaded_file is not None:
            os.makedirs("video", exist_ok=True)
            path = os.path.join("video", os.path.basename(uploaded_file.name))
            with open(path, "wb") as fh:
                fh.write(uploaded_file.getbuffer())

        if path and os.path.exists(path):
            result = sample_from_video_file(
                path=path,
                sample_interval=sample_interval,
                max_duration=duration,
                progress_callback=_update_progress,
            )
        else:
            result = CaptureResult()
            result.error_message = "Selected video file could not be accessed."

    progress_bar.empty()

    if result.success:
        st.session_state.capture_result = result
        st.session_state.reference_idx = 0
        st.session_state.analysis_done = False
        st.session_state.metrics_list = []
        st.session_state.summary = {}
        st.toast(f"Captured {len(result.frames)} keyframes successfully!", icon="✅")
    else:
        st.error(
            f"Capture error: {result.error_message} "
            "Please check camera availability or video file integrity."
        )


# ═══════════════════════════════════════════════════════════════════════
#  STEP 2 — INTERACTIVE GALLERY & REFERENCE SELECTION
# ═══════════════════════════════════════════════════════════════════════
result: CaptureResult | None = st.session_state.capture_result

if not (result and result.success):
    # Empty placeholder slots
    slots = "".join(
        f'<div class="ff-slot">T={i * sample_interval}s</div>'
        for i in range(min(expected_frames, 6))
    )
    _html(f'<div class="ff-slots">{slots}</div>')
    st.caption("Sampled keyframes will appear here once captured.")

else:
    frames = result.frames
    ref_idx = min(st.session_state.reference_idx, len(frames) - 1)
    analysed = st.session_state.analysis_done
    metrics_by_idx = (
        dict(enumerate(st.session_state.metrics_list)) if analysed else {}
    )

    st.caption(
        f"✅ {len(frames)} keyframes extracted from {result.duration_seconds}s clip "
        f"at {result.fps:.1f} FPS ({result.total_frames_read} frames processed)."
    )

    section(
        "2. Keyframe Gallery & Reference Selection",
        "Select any frame below to act as the golden baseline. All other frames will be compared against it.",
    )

    cols = st.columns(gallery_cols)
    for idx, f in enumerate(frames):
        is_ref = idx == ref_idx
        m = metrics_by_idx.get(idx)

        # Border color encoding
        if is_ref:
            outline = STATUS_COLOR["ref"]
        elif m is not None and m.ssim is not None:
            outline = STATUS_COLOR[status_of(m.ssim)[0]]
        else:
            outline = STATUS_COLOR["neutral"]

        with cols[idx % gallery_cols]:
            img = bgr_to_rgb(resize_for_display(f.image, max_width=480))
            if show_badges:
                img = overlay_timestamp_badge(img, f"T={fmt_t(f.timestamp)}")
            st.image(framed(img, outline), **STRETCH)

            if is_ref:
                tag = chip("ref", "Reference")
            elif m is not None and m.ssim is not None:
                key, _label = status_of(m.ssim)
                tag = chip(key, f"SSIM {m.ssim:.3f}")
            else:
                tag = ""
            _html(
                f'<div class="ff-meta">'
                f'<span><span class="name">Frame {idx}</span> '
                f'<span class="time">{fmt_t(f.timestamp)}</span></span>'
                f'{tag}</div>'
            )
            st.button(
                "★ Active Reference" if is_ref else "Set as Reference",
                key=f"ref_{idx}",
                on_click=set_reference,
                args=(idx,),
                disabled=is_ref,
                **STRETCH,
            )

    # ═══════════════════════════════════════════════════════════════════
    #  STEP 3 — MULTI-METRIC DEGRADATION ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    section(
        "3. Multi-Metric Structural Analysis",
        "Computes SSIM, MSE, Laplacian Sharpness, Brightness, and Canny Edge Density against the reference frame.",
    )

    with st.container(border=True):
        st.markdown(
            f"Comparing all frames against **Frame {ref_idx}** "
            f"(`T={fmt_t(frames[ref_idx].timestamp)}`, resolution `{frames[ref_idx].resolution[0]}×{frames[ref_idx].resolution[1]}`)."
        )
        run_col, _spacer2 = st.columns([1, 2])
        with run_col:
            run_clicked = st.button(
                "🔍 Run Analysis" if not analysed else "🔄 Re-run Analysis",
                type="primary",
                **STRETCH,
            )

    if run_clicked:
        with st.spinner("Computing structural metrics and degradation vectors…"):
            metrics_list, summary = analyse_all_frames(
                frames, reference_index=ref_idx
            )
        st.session_state.metrics_list = metrics_list
        st.session_state.summary = summary
        st.session_state.analysis_done = True
        st.rerun()

    # ═══════════════════════════════════════════════════════════════════
    #  STEP 4 — RESULTS & VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════
    if st.session_state.analysis_done:
        metrics_list = st.session_state.metrics_list
        summary = st.session_state.summary

        section("4. Analysis Findings & Degradation Metrics")

        # ── Diagnostic Verdict Banner ────────────────────────────────
        scored = [(i, m) for i, m in enumerate(metrics_list) if m.ssim is not None]
        worst_idx: int | None = None
        if scored:
            worst_idx, worst_m = min(scored, key=lambda p: p[1].ssim)
            key, _label = status_of(worst_m.ssim)
            titles = {
                "good": "🟢 Structural Consistency Maintained",
                "fair": "🟡 Moderate Structural Drift Detected",
                "poor": "🔴 Significant Structural Degradation Detected",
            }
            worst_time = (
                fmt_t(frames[worst_idx].timestamp) if worst_idx < len(frames) else "—"
            )
            _html(f"""
            <div class="ff-verdict v-{key}">
                <h3>{titles[key]}</h3>
                <p>
                    Maximum drift observed at <strong>Frame {worst_idx}</strong> ({worst_time}) with 
                    <strong>SSIM = {worst_m.ssim:.4f}</strong>. 
                    Benchmark: SSIM &ge; {SSIM_CONSISTENT:.2f} is structurally consistent; SSIM &lt; {SSIM_DEGRADED:.2f} indicates notable degradation.
                </p>
            </div>
            """)
        else:
            st.info("Capture multiple frames to compute degradation trends.")

        # ── Headline Key Metric Cards ────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Avg Brightness", f"{summary.get('avg_brightness', '—')}", help="Grayscale mean intensity [0-255]")
        k2.metric("Avg Sharpness", f"{summary.get('avg_sharpness', '—')}", help="Variance of Laplacian operator")
        k3.metric("Avg SSIM", f"{summary.get('avg_ssim', '—')}", help="1.0 = identical structural similarity")
        k4.metric("Max MSE", f"{summary.get('max_mse', '—')}", help="Peak Mean Squared Error pixel deviation")

        st.markdown("")
        tab_trend, tab_compare, tab_data = st.tabs(
            ["📈 SSIM Degradation Curve", "🔬 Frame Comparison & Heatmaps", "📋 Structured Metrics Table"]
        )

        # ── Tab 1: SSIM Degradation Curve ────────────────────────────
        with tab_trend:
            if len(scored) >= 2:
                import altair as alt

                chart_df = pd.DataFrame({
                    "Frame": [m.frame_label for _, m in scored],
                    "Time": [fmt_t(frames[i].timestamp) if i < len(frames) else "" for i, _ in scored],
                    "SSIM": [m.ssim for _, m in scored],
                    "Status": [status_of(m.ssim)[1] for _, m in scored],
                })
                y_min = max(0.0, min(chart_df["SSIM"]) - 0.05)
                y_enc = alt.Y(
                    "SSIM:Q",
                    scale=alt.Scale(domain=[y_min, 1.0]),
                    axis=alt.Axis(format=".2f", title="SSIM vs Reference", labelColor="#94A3B8", titleColor="#F8FAFC"),
                )
                x_enc = alt.X(
                    "Frame:N", sort=None,
                    axis=alt.Axis(labelAngle=0, title=None, labelColor="#94A3B8"),
                )
                base = alt.Chart(chart_df).encode(x=x_enc)
                line = base.mark_line(color="#6366F1", strokeWidth=2.5).encode(y=y_enc)
                points = base.mark_point(filled=True, size=140, opacity=1).encode(
                    y=y_enc,
                    color=alt.Color(
                        "Status:N",
                        scale=alt.Scale(
                            domain=["Consistent", "Drifting", "Degraded"],
                            range=[C["good"], C["fair"], C["poor"]],
                        ),
                        legend=alt.Legend(title=None, orient="top", labelColor="#F8FAFC"),
                    ),
                    tooltip=["Frame", "Time", alt.Tooltip("SSIM:Q", format=".4f"), "Status"],
                )
                rules = (
                    alt.Chart(pd.DataFrame({"y": [SSIM_CONSISTENT, SSIM_DEGRADED]}))
                    .mark_rule(strokeDash=[4, 4], color="#64748B", opacity=0.7, clip=True)
                    .encode(y="y:Q")
                )
                chart = (
                    (rules + line + points)
                    .properties(height=340)
                    .configure_view(stroke=None, fill="rgba(17, 24, 39, 0.4)")
                    .configure_axis(gridColor="#1E293B", domainColor="#334155")
                )
                st.altair_chart(chart, theme=None, **STRETCH)
                st.caption(
                    f"Dashed thresholds: **{SSIM_CONSISTENT:.2f}** (Consistent) and **{SSIM_DEGRADED:.2f}** (Degraded limit)."
                )
            else:
                st.info("At least two evaluated frames are required to plot the degradation trend.")

        # ── Tab 2: Side-by-Side Comparison & Heatmaps ────────────────
        with tab_compare:
            candidates = [i for i in range(len(frames)) if i != ref_idx]
            if not candidates:
                st.info("Single frame captured. Additional frames required for comparative analysis.")
            else:
                default_pos = candidates.index(worst_idx) if worst_idx in candidates else 0
                pick = st.selectbox(
                    "Select Frame to inspect against Reference",
                    options=candidates,
                    index=default_pos,
                    format_func=lambda i: f"Frame {i} · T={fmt_t(frames[i].timestamp)}",
                    key=f"inspect_{ref_idx}",
                )
                ref_bgr = frames[ref_idx].image
                sel = frames[pick]
                sel_m = metrics_by_idx.get(pick) or (
                    metrics_list[pick] if pick < len(metrics_list) else None
                )
                if sel_m is not None and sel_m.ssim is not None:
                    key, label = status_of(sel_m.ssim)
                    mse_txt = f"{sel_m.mse:.2f}" if sel_m.mse is not None else "—"
                    _html(
                        f'<div class="ff-meta">{chip(key, label)}'
                        f'<span class="time font-mono">SSIM {sel_m.ssim:.4f} &nbsp;|&nbsp; MSE {mse_txt}</span></div>'
                    )

                n_panels = 2 + int(show_heatmaps) + int(show_edges)
                pcols = st.columns(n_panels)
                with pcols[0]:
                    st.image(
                        bgr_to_rgb(resize_for_display(ref_bgr, 480)),
                        caption=f"Reference · Frame {ref_idx}",
                        **STRETCH,
                    )
                with pcols[1]:
                    st.image(
                        bgr_to_rgb(resize_for_display(sel.image, 480)),
                        caption=f"Frame {pick}",
                        **STRETCH,
                    )
                nxt = 2
                if show_heatmaps:
                    with pcols[nxt]:
                        heat = generate_diff_heatmap(ref_bgr, sel.image)
                        st.image(
                            bgr_to_rgb(resize_for_display(heat, 480)),
                            caption="Difference Heatmap (Jet)",
                            **STRETCH,
                        )
                    nxt += 1
                if show_edges:
                    with pcols[nxt]:
                        gray = cv2.cvtColor(sel.image, cv2.COLOR_BGR2GRAY)
                        edge_vis = generate_edge_overlay(gray)
                        st.image(
                            bgr_to_rgb(resize_for_display(edge_vis, 480)),
                            caption="Canny Edge Topology",
                            **STRETCH,
                        )

                if show_heatmaps and len(candidates) > 1:
                    with st.expander("🖼️ View Difference Heatmaps for All Frames"):
                        grid = st.columns(gallery_cols)
                        for n, i in enumerate(candidates):
                            with grid[n % gallery_cols]:
                                heat = generate_diff_heatmap(ref_bgr, frames[i].image)
                                st.image(
                                    bgr_to_rgb(resize_for_display(heat, 400)),
                                    caption=f"Frame {i} (T={fmt_t(frames[i].timestamp)})",
                                    **STRETCH,
                                )

        # ── Tab 3: Structured Metrics Data Table ─────────────────────
        with tab_data:
            rows = []
            for m in metrics_list:
                rows.append({
                    "Frame": m.frame_label,
                    "Status": status_of(m.ssim)[1],
                    "SSIM": m.ssim,
                    "MSE": m.mse,
                    "Brightness": m.brightness,
                    "Δ Brightness": m.brightness_delta,
                    "Sharpness": m.sharpness,
                    "Δ Sharpness": m.sharpness_delta,
                    "Edge Density": m.edge_density,
                    "Resolution": fmt_res(m.resolution),
                })
            df = pd.DataFrame(rows)
            st.dataframe(
                df,
                hide_index=True,
                column_config={
                    "SSIM": st.column_config.ProgressColumn(
                        "SSIM", min_value=0.0, max_value=1.0, format="%.4f"
                    ),
                    "MSE": st.column_config.NumberColumn(format="%.2f"),
                    "Edge Density": st.column_config.NumberColumn(format="%.4f"),
                    "Brightness": st.column_config.NumberColumn(format="%.2f"),
                    "Sharpness": st.column_config.NumberColumn(format="%.2f"),
                },
                **STRETCH,
            )
            st.download_button(
                "📥 Export CSV Report",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="frameforensics_metrics_report.csv",
                mime="text/csv",
            )
            with st.expander("📖 Metric Formulations & Theoretical Background"):
                st.markdown(
                    """
                    - **SSIM (Structural Similarity Index)**: Measures luminance, contrast, and structural degradation against the reference image (Wang et al., 2004). Value of `1.0` is mathematically identical.
                    - **MSE (Mean Squared Error)**: Quantifies cumulative pixel-level intensity variance: $\\frac{1}{HW} \\sum (I_{ref} - I_{cmp})^2$.
                    - **Sharpness (Variance of Laplacian)**: Focus measure computed via $\\nabla^2 I$. High variance denotes high-frequency edge content; low values denote motion blur or defocus.
                    - **Brightness (Mean Luminance)**: Spatial mean pixel intensity $\\mu = \\frac{1}{HW} \\sum I(x, y)$.
                    - **Edge Density (Canny Edge Ratio)**: Ratio of detected edge pixels to total image pixels under dual-threshold hysteresis.
                    """
                )

