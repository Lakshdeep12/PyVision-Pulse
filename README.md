# 🔬 FrameForensics — Video Capture & Degradation Analyzer

A production-grade Streamlit web application for **automated video capture, keyframe sampling, image comparison, and structural degradation analysis** powered by OpenCV and scikit-image.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Live Webcam Capture** | Record from your laptop camera with auto-detected FPS |
| **Video File Analysis** | Load `.mp4`, `.avi`, `.mov`, `.mkv` files for offline analysis |
| **Keyframe Sampling** | Configurable interval (default: 1 frame every 2 seconds) |
| **Interactive Gallery** | Multi-column frame gallery with timestamp badges |
| **Reference Selection** | Choose any frame as the baseline for comparison |
| **6 Quality Metrics** | Resolution, Brightness, Sharpness, Edge Density, SSIM, MSE |
| **Diff Heatmaps** | Colour-mapped absolute-difference visualisations |
| **Edge Overlays** | Canny edge detection visualisation per frame |
| **SSIM Degradation Curve** | Line chart showing structural similarity over time |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — Responsive web UI with sidebar controls
- **OpenCV** (`cv2`) — Video capture, frame extraction, image processing
- **NumPy** — Numerical computation
- **scikit-image** — `structural_similarity` (SSIM)
- **Pandas** — Structured data tables

---

## 📁 Project Structure

```
Dip lab project/
├── app.py              # Main Streamlit application (UI + orchestration)
├── capture.py          # Video capture & frame sampling engine
├── metrics.py          # Multi-metric image comparison module
├── utils.py            # Colour conversion, display helpers, heatmaps
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── video/
    └── Indian_classroom_video.mp4   # Sample video for testing
```

---

## 🚀 Setup & Run

### 1. Clone / navigate to the project

```bash
cd "Dip lab project"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** in your default browser.

---

## 📖 How to Use

1. **Choose input source** in the sidebar — Webcam or Video file.
2. **Adjust capture settings** — recording duration and sampling interval.
3. Click **Start Capture** (webcam) or **Load & Sample** (file).
4. Browse the **Frame Gallery** and select a **Reference Frame**.
5. Click **Analyse Frames** to compute all metrics.
6. Review the **data table**, **heatmaps**, **edge maps**, and **SSIM curve**.

---

## 📊 Metrics Explained

| Metric | Formula / Method | Interpretation |
|---|---|---|
| **Brightness** | `μ = mean(grayscale)` | Average luminance [0–255] |
| **Sharpness** | `Var(Laplacian(I))` | High = sharp; Low = blurry |
| **Edge Density** | `count(Canny edges) / total pixels` | Texture richness [0–1] |
| **SSIM** | Wang et al. (2004) structural similarity | 1.0 = identical structure |
| **MSE** | `mean((I_ref − I_cmp)²)` | 0 = identical pixels |

---

## ⚠️ Troubleshooting

| Issue | Fix |
|---|---|
| Webcam not opening | Close other apps using the camera; try device index 1 |
| Black frames | Ensure camera isn't physically blocked; add a 0.5 s warm-up |
| SSIM errors | Handled automatically via dimension matching & window-size clamping |
| Module not found | Run `pip install -r requirements.txt` in the project directory |

---

## 📝 License

Academic project — Digital Image Processing Lab.