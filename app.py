"""
app.py — FrameForensics: Video Capture & Degradation Analysis
=============================================================
Production-grade Streamlit web application for automated video capture,
keyframe sampling, interactive frame selection, and multi-metric
structural degradation analysis.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import time

import cv2
import numpy as np
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


# ─── Page configuration ─────────────────────────────────────────────────
st.set_page_config(
    page_title="FrameForensics — Video Degradation Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for a polished, dark-mode-friendly UI ────────────────────
st.markdown("""
<style>
/* ── Root palette ─────────────────────────────────── */
:root {
    --accent:    #6C63FF;
    --accent2:   #00D4AA;
    --surface:   rgba(30, 30, 46, 0.65);
    --card-bg:   rgba(40, 40, 60, 0.55);
    --text:      #E2E2F0;
    --text-dim:  #9090B0;
    --success:   #00D4AA;
    --warning:   #FFB347;
    --danger:    #FF6B6B;
}

/* ── Main container ───────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

/* ── Hero header ──────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
}
.hero-header h1 {
    background: linear-gradient(90deg, #6C63FF, #00D4AA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: var(--text-dim);
    font-size: 1rem;
    margin: 0;
}

/* ── Glass card ───────────────────────────────────── */
.glass-card {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(108,99,255,0.2);
}

/* ── Metric pill ──────────────────────────────────── */
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(108,99,255,0.15), rgba(0,212,170,0.10));
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 22px;
    padding: 6px 16px;
    font-size: 0.85rem;
    color: var(--text);
    margin: 3px;
}
.metric-pill .val {
    font-weight: 700;
    color: var(--accent2);
}

/* ── Badge ────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-ref  { background: rgba(108,99,255,0.25); color: #a5a0ff; }
.badge-ok   { background: rgba(0,212,170,0.20);  color: #00D4AA; }
.badge-warn { background: rgba(255,179,71,0.20);  color: #FFB347; }
.badge-bad  { background: rgba(255,107,107,0.20); color: #FF6B6B; }

/* ── Section heading ──────────────────────────────── */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin: 1.5rem 0 0.7rem 0;
    padding-left: 12px;
    border-left: 4px solid var(--accent);
}

/* ── Sidebar polish ───────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12122a 0%, #1a1a3e 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #c4c0ff;
}

/* ── Dataframe header ─────────────────────────────── */
.stDataFrame thead th {
    background: rgba(108,99,255,0.15) !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* ── Smoother image corners ───────────────────────── */
.stImage > img {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state defaults ─────────────────────────────────────────────
if "capture_result" not in st.session_state:
    st.session_state.capture_result: CaptureResult | None = None
if "reference_idx" not in st.session_state:
    st.session_state.reference_idx: int = 0
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done: bool = False
if "metrics_list" not in st.session_state:
    st.session_state.metrics_list = []
if "summary" not in st.session_state:
    st.session_state.summary = {}


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR — Controls
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Capture Settings")

    source_mode = st.radio(
        "📹 Input source",
        ["🎥 Webcam (live)", "📂 Video file"],
        index=1,
        help="Choose between a live webcam feed or an existing video file.",
    )

    st.markdown("---")

    duration = st.slider(
        "⏱ Recording duration (s)",
        min_value=4, max_value=60, value=10, step=2,
        help="How many seconds to capture from the webcam.",
    )
    sample_interval = st.slider(
        "📸 Sample every (s)",
        min_value=1, max_value=10, value=2, step=1,
        help="Extract one keyframe at this interval.",
    )

    uploaded_file = None
    video_path = None
    if "Video file" in source_mode:
        st.markdown("---")
        st.markdown("### 📁 Video Source")
        upload_mode = st.radio(
            "Choose video input method",
            ["Upload a file", "Use sample video"],
            index=1,
        )
        if upload_mode == "Upload a file":
            uploaded_file = st.file_uploader(
                "Drop a video", type=["mp4", "avi", "mov", "mkv"]
            )
        else:
            sample_path = os.path.join("video", "Indian_classroom_video.mp4")
            if os.path.exists(sample_path):
                video_path = sample_path
                st.success(f"✅ Using **{os.path.basename(sample_path)}**")
            else:
                st.warning("Sample video not found in `video/` folder.")

    st.markdown("---")
    st.markdown("### 🎨 Display Options")
    show_badges = st.checkbox("Timestamp badges on frames", value=True)
    show_heatmaps = st.checkbox("Diff heatmaps", value=True)
    show_edges = st.checkbox("Edge overlays", value=False)
    gallery_cols = st.slider("Gallery columns", 2, 6, 3)


# ═══════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero-header">
    <h1>🔬 FrameForensics</h1>
    <p>Automated Video Capture · Frame Sampling · Structural Degradation Analysis</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  STEP 1 — Capture / Load
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-title">① Capture & Sample Frames</div>',
            unsafe_allow_html=True)

col_btn, col_info = st.columns([1, 2])

with col_btn:
    capture_clicked = st.button(
        "🚀  Start Capture" if "Webcam" in source_mode else "📂  Load & Sample",
        use_container_width=True,
        type="primary",
    )

if capture_clicked:
    progress_bar = st.progress(0, text="Initialising…")

    def _update_progress(frac: float):
        progress_bar.progress(frac, text=f"Capturing… {frac*100:.0f}%")

    if "Webcam" in source_mode:
        result = capture_and_sample(
            duration=duration,
            sample_interval=sample_interval,
            progress_callback=_update_progress,
        )
    else:
        # Resolve video path
        path = video_path
        if uploaded_file is not None:
            tmp_path = os.path.join("video", uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            path = tmp_path

        if path and os.path.exists(path):
            result = sample_from_video_file(
                path=path,
                sample_interval=sample_interval,
                max_duration=duration,
                progress_callback=_update_progress,
            )
        else:
            result = CaptureResult()
            result.error_message = "No video file selected or file not found."

    progress_bar.empty()

    if result.success:
        st.session_state.capture_result = result
        st.session_state.reference_idx = 0
        st.session_state.analysis_done = False
        st.toast(
            f"✅ Captured **{len(result.frames)}** keyframes "
            f"({result.duration_seconds}s @ {result.fps:.1f} FPS)",
            icon="🎬",
        )
    else:
        st.error(f"⚠️ Capture failed: {result.error_message}")


# ═══════════════════════════════════════════════════════════════════════
#  STEP 2 — Frame Gallery & Reference Selection
# ═══════════════════════════════════════════════════════════════════════

result: CaptureResult | None = st.session_state.capture_result

if result and result.success:
    frames = result.frames

    # ── Session info pills ────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:1rem;">
            <span class="metric-pill">🎞 Frames <span class="val">{len(frames)}</span></span>
            <span class="metric-pill">🕐 Duration <span class="val">{result.duration_seconds}s</span></span>
            <span class="metric-pill">📐 FPS <span class="val">{result.fps:.1f}</span></span>
            <span class="metric-pill">📊 Total read <span class="val">{result.total_frames_read}</span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">② Frame Gallery — Select a Reference</div>',
                unsafe_allow_html=True)
    st.caption("Click a frame's radio button to set it as the **Reference Image** for comparison.")

    cols = st.columns(gallery_cols)
    for idx, f in enumerate(frames):
        with cols[idx % gallery_cols]:
            display_img = bgr_to_rgb(resize_for_display(f.image, max_width=480))
            if show_badges:
                display_img = overlay_timestamp_badge(display_img, f"T={f.timestamp}s")

            st.image(display_img, use_container_width=True)

            is_ref = idx == st.session_state.reference_idx
            badge_html = (
                '<span class="badge badge-ref">★ REFERENCE</span>'
                if is_ref else ""
            )
            st.markdown(
                f"{badge_html} &nbsp; **Frame {idx}** — `{f.resolution[0]}×{f.resolution[1]}`",
                unsafe_allow_html=True,
            )

    # Reference selector
    st.markdown("")
    ref_choice = st.selectbox(
        "🖼️  Choose Reference Frame",
        options=list(range(len(frames))),
        format_func=lambda i: f"Frame {i}  —  T={frames[i].timestamp}s",
        index=st.session_state.reference_idx,
        key="ref_select",
    )
    st.session_state.reference_idx = ref_choice

    # ═══════════════════════════════════════════════════════════════════
    #  STEP 3 — Run Analysis
    # ═══════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-title">③ Multi-Metric Degradation Analysis</div>',
                unsafe_allow_html=True)

    if st.button("🔍  Analyse Frames", use_container_width=True, type="primary"):
        with st.spinner("Computing metrics across all frames…"):
            metrics_list, summary = analyse_all_frames(
                frames, reference_index=st.session_state.reference_idx
            )
        st.session_state.metrics_list = metrics_list
        st.session_state.summary = summary
        st.session_state.analysis_done = True
        st.toast("Analysis complete!", icon="✅")

    # ─── Results ─────────────────────────────────────────────────────
    if st.session_state.analysis_done:
        metrics_list = st.session_state.metrics_list
        summary = st.session_state.summary

        # ── Summary cards ────────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Avg Brightness", f"{summary.get('avg_brightness', '—')}")
        s2.metric("Avg Sharpness", f"{summary.get('avg_sharpness', '—')}")
        s3.metric(
            "Avg SSIM",
            f"{summary.get('avg_ssim', '—')}",
            help="1.0 = identical to reference",
        )
        s4.metric(
            "Max MSE",
            f"{summary.get('max_mse', '—')}",
            help="Higher = more pixel deviation",
        )

        # ── Data table ───────────────────────────────────────────────
        import pandas as pd

        table_data = []
        for m in metrics_list:
            ssim_badge = ""
            if m.ssim is not None:
                if m.ssim >= 0.95:
                    ssim_badge = "🟢"
                elif m.ssim >= 0.80:
                    ssim_badge = "🟡"
                else:
                    ssim_badge = "🔴"

            table_data.append({
                "Frame":         m.frame_label,
                "Resolution":    m.resolution,
                "Brightness":    m.brightness,
                "Δ Brightness":  m.brightness_delta,
                "Sharpness":     m.sharpness,
                "Δ Sharpness":   m.sharpness_delta,
                "Edge Density":  m.edge_density,
                "SSIM":          m.ssim if m.ssim is not None else "—",
                "Status":        ssim_badge,
                "MSE":           m.mse if m.mse is not None else "—",
            })

        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "SSIM": st.column_config.NumberColumn(format="%.4f"),
                "MSE":  st.column_config.NumberColumn(format="%.2f"),
                "Edge Density": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        # ── Visual comparisons ───────────────────────────────────────
        if show_heatmaps or show_edges:
            st.markdown(
                '<div class="section-title">④ Visual Comparison Maps</div>',
                unsafe_allow_html=True,
            )

            ref_bgr = frames[st.session_state.reference_idx].image
            vis_cols = st.columns(gallery_cols)
            for idx, f in enumerate(frames):
                if idx == st.session_state.reference_idx:
                    continue
                with vis_cols[idx % gallery_cols]:
                    st.markdown(
                        f"**Frame {idx}** — T={f.timestamp}s",
                    )
                    if show_heatmaps:
                        heatmap = generate_diff_heatmap(ref_bgr, f.image)
                        st.image(
                            bgr_to_rgb(resize_for_display(heatmap, 400)),
                            caption="Diff Heatmap",
                            use_container_width=True,
                        )
                    if show_edges:
                        gray = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
                        edge_vis = generate_edge_overlay(gray)
                        st.image(
                            bgr_to_rgb(resize_for_display(edge_vis, 400)),
                            caption="Edge Map",
                            use_container_width=True,
                        )

        # ── Degradation chart ────────────────────────────────────────
        ssim_vals = [
            m.ssim for m in metrics_list if m.ssim is not None
        ]
        if len(ssim_vals) > 1:
            st.markdown(
                '<div class="section-title">⑤ SSIM Degradation Curve</div>',
                unsafe_allow_html=True,
            )
            import pandas as pd

            chart_df = pd.DataFrame({
                "Frame": [m.frame_label for m in metrics_list if m.ssim is not None],
                "SSIM":  ssim_vals,
            })
            st.line_chart(chart_df, x="Frame", y="SSIM", height=320)

            # Interpretation
            min_ssim = min(ssim_vals)
            if min_ssim >= 0.95:
                st.success(
                    "🟢 **Excellent structural consistency** — all frames are "
                    "nearly identical to the reference (SSIM ≥ 0.95)."
                )
            elif min_ssim >= 0.80:
                st.warning(
                    "🟡 **Moderate degradation detected** — some frames show "
                    f"structural differences (min SSIM = {min_ssim:.4f})."
                )
            else:
                st.error(
                    "🔴 **Significant structural degradation** — at least one "
                    f"frame diverges substantially from the reference "
                    f"(min SSIM = {min_ssim:.4f})."
                )

else:
    # ── Empty state ──────────────────────────────────────────────────
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:3rem 2rem;">
            <h3 style="color:#6C63FF;">No frames captured yet</h3>
            <p style="color:#9090B0;">
                Configure your settings in the sidebar, then click
                <strong>Start Capture</strong> or <strong>Load & Sample</strong>
                to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#606080; font-size:0.82rem;">
        <strong>FrameForensics</strong> · Built with Python, OpenCV, scikit-image & Streamlit<br>
        Structural Similarity (SSIM) · Laplacian Sharpness · Canny Edge Density · MSE
    </div>
    """,
    unsafe_allow_html=True,
)
