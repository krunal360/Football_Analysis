# ⚽ AI-Powered Football Match Analysis Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-00FFFF?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![ByteTrack](https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-orange)](https://github.com/ifzhang/ByteTrack)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Unsupervised%20Clustering-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

An automated, end-to-end Computer Vision and Machine Learning system that extracts tactical match intelligence, player tracking, team affiliations, and performance kinematics directly from standard broadcast football video footage—eliminating the need for expensive GPS sensor vests or manual match tagging.

Developed as a Major Academic Project for **B.Tech 7th Semester - Machine Learning (4IT47)** by **Krunal Rohit (23IT419)**.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture & Pipeline](#-system-architecture--pipeline)
- [Directory Structure](#-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [1. Running the Interactive Streamlit Dashboard](#1-running-the-interactive-streamlit-dashboard)
  - [2. Running the Pipeline via CLI](#2-running-the-pipeline-via-cli)
  - [3. Running on Google Colab (Fast GPU Inference)](#3-running-on-google-colab-fast-gpu-inference)
- [Model Performance & Evaluation](#-model-performance--evaluation)
- [Known Limitations & Future Scope](#-known-limitations--future-scope)
- [Author & Academic Details](#-author--academic-details)

---

## 📖 Project Overview

Modern professional football clubs rely heavily on tactical analytics (possession percentages, player running speeds, tactical pressing zones). Historically, gathering these metrics required either:
1. **Manual Tagging:** Match analysts spending 4 to 6 hours after each fixture manually annotating video events.
2. **Wearable Sensors:** Expensive GPS tracker vests that cannot be used on historical broadcast television footage or youth matches.

### The Solution:
This project processes standard 1080p broadcast video to automatically:
- Detect **players**, **referees**, **goalkeepers**, and the **ball**.
- Track every individual continuously across occlusions and tackles with persistent Track IDs.
- Cluster team jersey kits into **Team 0** and **Team 1** using unsupervised K-Means clustering.
- Filter and smooth ball trajectories using rolling median outlier rejection and linear interpolation.
- Calibrate player pixel movements into **real-world physical units (meters and km/h)**.
- Compute **live ball possession percentages** (excluding dead-ball stoppages) and generate **2D tactical density heatmaps**.
- Present all match intelligence via an interactive **Streamlit Web Application** supporting dynamic video upload.

---

## 🚀 Key Features

- **🎯 YOLOv8 Deep Learning Detector:** Fine-tuned YOLOv8m on broadcast match footage with per-class confidence thresholds (0.10 for the small, fast ball; 0.25 for outfield players).
- **🏃 ByteTrack Multi-Object Tracker:** Employs Kalman filters to predict motion and maintain track identities through collisions, coupled with majority-voting role consistency.
- **⚽ Ball Trajectory Smoothing:** A 2-step geometric filter combining a 25-frame rolling median outlier detector with 15-frame linear gap interpolation, boosting ball visibility from **78.7% to 91.7%** (+13% recovery).
- **👕 Unsupervised Kit Color Assignment:** Automatically crops upper torsos to isolate shirt fabrics and clusters colors in CIELAB color space using K-Means ($K=2$). Achieves **99.1% consistency** without manual labels.
- **⏱️ Real-World Kinematics:** Anchors tracking to the foot-contact point on the ground and uses human height scaling ($1.75\text{ m} \approx 95\text{ px}$) to deliver speeds in **km/h** and distances in **meters**.
- **📈 Possession & Tactical Heatmaps:** Evaluates possession proximity (<80 px) with a 3-frame hysteresis hold, dead-ball exclusion, and 2D Gaussian pitch occupancy heatmaps.
- **💻 Full-Featured Streamlit Web App:** Multi-tab interactive UI offering live video playback with tactical HUD rings, sortable player tables, team summary charts, and custom video analysis.

---

## 🏗️ System Architecture & Pipeline

The system is organized into a clean 3-layer architecture:

```
+--------------------------------------------------------------------------+
|                     1. COMPUTER VISION LAYER                             |
|  - 1080p Broadcast Video Ingestion (25 FPS)                              |
|  - YOLOv8m Neural Network Detection (Players, Referees, Keepers, Ball)   |
|  - Per-Class Confidence Thresholds (0.10 for ball, 0.25 for people)      |
|  - Pickled Detection Caching                                             |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+------------------------------------+-------------------------------------+
|                     2. MACHINE LEARNING LAYER                            |
|  - ByteTrack Association + Kalman Filter Motion Prediction               |
|  - Rolling Median Outlier Rejection + Linear Gap Interpolation           |
|  - Upper-Torso Bounding Box Cropping                                     |
|  - CIELAB Color Space Conversion + K-Means Kit Clustering (K=2)          |
+------------------------------------+-------------------------------------+
                                     |
                                     v
+------------------------------------+-------------------------------------+
|                     3. ANALYTICS & UI LAYER                              |
|  - Pitch Ground Foot-Contact Anchoring                                   |
|  - 1.75m Height Calibration (Pixels -> Meters & km/h)                   |
|  - Possession Proximity (<80px) + Dead-Ball Filter + Turnover Detection  |
|  - 2D Gaussian Tactical Density Heatmaps                                 |
|  - Interactive Streamlit Multi-Tab Web Dashboard                         |
+--------------------------------------------------------------------------+
```

### End-to-End 6-Step Workflow:
```
[ Step 1: Input Video ] 
        ↓
[ Step 2: YOLOv8 Detection ] 
        ↓
[ Step 3: ByteTrack Tracking ] 
        ↓
[ Step 4: Kit Color Clustering (K-Means) ] 
        ↓
[ Step 5: Metrics & Kinematics Engine ] 
        ↓
[ Step 6: Streamlit Web Dashboard ]
```

---

## 📂 Directory Structure

```text
Football_Analysis/
├── app.py                      # Interactive Streamlit Web Dashboard
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration and metadata
├── .gitignore                  # Git exclusions for videos, weights, and caches
├── README.md                   # Project documentation
│
├── src/
│   └── football_analysis/      # Core modular Python package
│       ├── __init__.py
│       ├── config.py           # Thresholds, FPS, paths, and calibration settings
│       ├── detector.py         # YOLOv8 inference & detection caching
│       ├── tracker.py          # ByteTrack integration & ball smoothing
│       ├── team_assigner.py    # Torso cropping & K-Means kit clustering
│       ├── metrics.py          # Speed (km/h), distance (m), & possession
│       ├── extra_stats.py      # Tactical heatmaps & turnover analysis
│       ├── visualizer.py       # Pitch annotations, player rings & HUD
│       └── pipeline.py         # End-to-end execution pipeline
│
├── notebooks/
│   └── football_colab.ipynb    # Google Colab notebook for GPU training/inference
│
├── data/
│   ├── input_videos/           # Place raw match videos here (.mp4, .avi)
│   ├── datasets/               # Training datasets (Roboflow soccer format)
│   └── models/                 # Model checkpoints
│
├── outputs/                    # Generated match outputs (CSV, JSON, Heatmaps)
│   ├── stats.json              # Possession & match statistics
│   ├── player_stats.csv        # Speed, distance, and time per player
│   ├── team_summary.csv        # Aggregate metrics per team
│   ├── holder_time.csv         # Individual possession time
│   ├── heatmap_team0.png       # Team 0 tactical pressing density heatmap
│   ├── heatmap_team1.png       # Team 1 tactical pressing density heatmap
│   └── eval/                   # YOLOv8 validation curves & confusion matrices
│
└── weights/                    # Saved fine-tuned model weights
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.
- (Optional) NVIDIA GPU with CUDA support for accelerated model inference.

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Football_Analysis.git
cd Football_Analysis
```

### 2. Create and Activate a Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🖥️ Usage Guide

### 1. Running the Interactive Streamlit Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

**Dashboard Features:**
- **Overview Tab:** Live ball possession breakdown, dead-ball stoppage counts, and pipeline execution facts.
- **Video Tab:** Video player displaying the tracked match footage with color-coded player rings and tactical HUD.
- **Players Tab:** Interactive, sortable table of all 35 players with distance covered in meters and top speeds in km/h.
- **Teams Tab:** Aggregate team distance, sprint bursts, and tactical pitch density heatmaps.
- **Model Tab:** Training validation loss curves, precision-recall charts, and confusion matrix.
- **Sidebar:** Choose existing match analysis, select local test videos, or upload any new `.mp4` clip to trigger automatic analysis.

---

### 2. Running the Pipeline via CLI
To analyze a match video directly from the command line:
```bash
python -m football_analysis.pipeline --video data/input_videos/test_clip.mp4 --out-name annotated.mp4
```

---

### 3. Running on Google Colab (Fast GPU Inference)
YOLOv8 inference and tracking on 1080p video takes ~25 seconds on a Colab GPU (NVIDIA T4), compared to 15+ minutes on a CPU.
1. Upload the project folder or open `notebooks/football_colab.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime > Change runtime type > GPU (T4)**.
3. Run the cells to process your video. The outputs generated in `outputs/` can be downloaded directly to view in the Streamlit app.

---

## 📊 Model Performance & Evaluation

The YOLOv8m model was fine-tuned on football broadcast footage. Below are the validation metrics on the hold-out validation split:

| Class | Precision | Recall | mAP@50 | mAP@50-95 | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Player** | **98.0%** | **97.7%** | **99.1%** | 72.3% | Near-perfect detection across dense pitch scenes |
| **Referee** | **94.8%** | **93.3%** | **96.1%** | 61.4% | Clean separation from outfield players |
| **Goalkeeper** | **79.6%** | **88.9%** | **93.6%** | 69.0% | Distinct jersey identification |
| **Ball** | **71.0%** | **42.9%** | **42.6%** | 19.5% | Small size (~15px) resolved via trajectory smoothing |

### Ball Trajectory Filter Results:
- **Raw YOLOv8 Detection:** The ball was detected in 78.7% of raw frames due to motion blur and occlusions.
- **Post-Median & Interpolation Filter:** Ball tracking continuity increased to **91.7% of all frames** (+13.0% recovery).

---

## ⚠️ Known Limitations & Future Scope

1. **Broadcast Camera Panning:** In broadcast television footage, camera panning and zooming adds apparent pixel movement. Measured velocities combine player running with camera motion.
   - *Future Work:* Implement **2D-to-3D planar pitch homography** (detecting pitch lines and penalty boxes) to project coordinates onto a fixed bird's-eye pitch template.
2. **Pack Occlusions:** When several players tackle simultaneously in a crowded box, IDs can occasionally swap.
   - *Future Work:* Integrate a deep appearance Re-Identification (Re-ID) network to remember player faces and shirt numbers after prolonged camera exits.

---

## 👨‍💻 Author & Academic Details

- **Student Name:** Krunal Rohit
- **Student ID:** 23IT419
- **Batch:** A10
- **Semester:** 7th Semester (B.Tech)
- **Subject:** Machine Learning (4IT47)
- **Department:** Department of Information Technology

---

## 📜 Acknowledgments & References
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for state-of-the-art real-time object detection.
- [Roboflow Supervision](https://github.com/roboflow/supervision) & [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking.
- [Streamlit](https://streamlit.io/) for the web dashboard framework.
