import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
EVAL = OUT / "eval"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TEAM_NAME = {0: "Team 0 (green kit)", 1: "Team 1 (white kit)", -1: "No team"}
M_PER_PX = 1.75 / 95.0  # Approx calibration: 1.75m avg player height ~ 95px in 1080p
FPS = 25.0

# Facts copied from the pipeline's printed output for this clip
RUN_FACTS = {
    "Video": "30 s, 25 FPS, 1920x1080 (750 frames)",
    "Tracks": "45 IDs (37 player, 8 referee); 35 last 50+ frames; 30 used for stats",
    "Ball detected": "78.7% of frames by the model, 91.7% after outlier removal and interpolation",
    "Team assignment": "99.1% agreement with the visible shirt colour (consistency check)",
}

VAL = pd.DataFrame(
    {"class": ["ball", "goalkeeper", "player", "referee"],
     "precision": [0.710, 0.796, 0.980, 0.948], "recall": [0.429, 0.889, 0.977, 0.933],
     "mAP50": [0.426, 0.936, 0.991, 0.961], "mAP50-95": [0.195, 0.690, 0.723, 0.614]})
TEST = pd.DataFrame(
    {"class": ["ball", "goalkeeper", "player", "referee"],
     "precision": [0.926, 0.856, 0.964, 0.844], "recall": [0.364, 0.909, 0.973, 0.931],
     "mAP50": [0.365, 0.924, 0.988, 0.953], "mAP50-95": [0.125, 0.634, 0.719, 0.613]})

st.set_page_config(page_title="Football Match Analysis", layout="wide")


# --- Sidebar: Pipeline & Video Controls ---
st.sidebar.title("Match Analysis Pipeline")
st.sidebar.subheader("Analyze Video")

video_source_mode = st.sidebar.radio(
    "Video Source:",
    ["Current Analysis (outputs/)", "Select Input Video", "Upload New Video (.mp4)"]
)

target_video_path = None
if video_source_mode == "Select Input Video":
    input_dir = ROOT / "data" / "input_videos"
    existing_videos = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi")) + list(input_dir.glob("*.mov"))
    if existing_videos:
        target_video_path = st.sidebar.selectbox("Choose video:", existing_videos, format_func=lambda x: x.name)
    else:
        st.sidebar.info("No videos found in data/input_videos/")

elif video_source_mode == "Upload New Video (.mp4)":
    uploaded = st.sidebar.file_uploader("Upload match video:", type=["mp4", "mov", "avi"])
    if uploaded is not None:
        save_dir = ROOT / "data" / "input_videos"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.sidebar.success(f"Uploaded: {uploaded.name}")
        target_video_path = save_path

if target_video_path is not None:
    st.sidebar.write(f"**Selected:** `{target_video_path.name}`")
    if st.sidebar.button("🚀 Run Analysis Pipeline", type="primary"):
        try:
            from football_analysis.pipeline import run as run_pipeline
            with st.spinner("Processing video: Running YOLOv8 detection, ByteTrack, kit clustering, and metrics..."):
                run_pipeline(target_video_path, out_name="annotated.mp4", use_cache=False)
            st.sidebar.success("Analysis complete! Reloading dashboard...")
            st.cache_data.clear()
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Error during analysis: {e}")

with st.sidebar.expander("ℹ️ Colab vs Local Execution"):
    st.caption(
        "**Fast GPU Processing (Google Colab):**\n\n"
        "YOLOv8 inference takes ~25s on a Colab GPU (T4), but can take 15+ mins on a laptop CPU.\n\n"
        "For full matches, run `notebooks/football_colab.ipynb` on Colab, and the generated outputs in "
        "`outputs/` will immediately appear in this dashboard."
    )


@st.cache_data
def load_json(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_csv(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def show_image(path, caption):
    if Path(path).exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Missing file: {Path(path).name}")


stats = load_json("stats.json")
extra = load_json("extra_stats.json")
players = load_csv("player_stats.csv")
teams = load_csv("team_summary.csv")
holders = load_csv("holder_time.csv")

st.title("Football Match Analysis: YOLOv8 + ByteTrack")
st.caption("AI-powered broadcast soccer analytics. Distances and speeds are image-based and camera-dependent.")

tab_over, tab_video, tab_players, tab_teams, tab_model, tab_limits = st.tabs(
    ["Overview", "Video", "Players", "Teams", "Model", "Limitations"])

with tab_over:
    if stats is None:
        st.error("outputs/stats.json not found. Run the pipeline first.")
    else:
        p = stats["possession"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Live play: green kit", f'{p["team0_pct"]}%')
        c2.metric("Live play: white kit", f'{p["team1_pct"]}%')
        c3.metric("All frames (green / white)", f'{p["team0_pct_all_frames"]} / {p["team1_pct_all_frames"]}')
        c4.metric("Possession changes (approx.)", extra["possession_changes_live_play"] if extra else "n/a")
        st.write(
            f'{p["dead_ball_frames"]} frames in which the ball stayed still (a restart) are left out of the '
            f'live-play figure. The two teams are roughly even in live play; one 30-second clip is too '
            f'short to call it a lead.')
    st.subheader("Pipeline results for this clip")
    for k, v in RUN_FACTS.items():
        st.write(f"**{k}:** {v}")

with tab_video:
    video = OUT / "annotated_h264.mp4"
    if not video.exists():
        video = OUT / "annotated.mp4"

    if video.exists():
        st.video(str(video))
        st.caption("Orange = green kit, blue = white kit, yellow = referee, white dot = ball holder. "
                   "The panel counts all frames, so its final value matches the 'all frames' figure.")
    else:
        st.warning("outputs/annotated_h264.mp4 or annotated.mp4 not found. Run the pipeline first.")

with tab_players:
    if players is None:
        st.warning("outputs/player_stats.csv not found.")
    else:
        col_u1, col_u2 = st.columns([3, 1])
        with col_u1:
            unit_mode = st.radio(
                "Display Units:",
                ["Intuitive Units (Seconds, Meters, km/h)", "Raw Computer Vision Units (Frames, Pixels, px/s)"],
                horizontal=True
            )
        with col_u2:
            only_stable = st.checkbox("Stable tracks only (50+ frames)", value=True)

        df = players.copy()
        df["team"] = df["team"].map(TEAM_NAME)
        if only_stable:
            df = df[~df["fragment"]]

        if unit_mode.startswith("Intuitive"):
            df_display = pd.DataFrame({
                "Track ID": df["track_id"],
                "Role": df["role"].astype(str).str.capitalize(),
                "Team": df["team"],
                "Time Active": (df["frames_tracked"] / FPS).round(1).astype(str) + " s",
                "Est. Distance": (df["distance_px"] * M_PER_PX).round(1).astype(str) + " m",
                "Avg Speed": (df["avg_speed_px_s"] * M_PER_PX * 3.6).round(1).astype(str) + " km/h",
                "Top Sprint": (df["max_speed_px_s"] * M_PER_PX * 3.6).round(1).astype(str) + " km/h",
                "Position Jitters": df["glitches"],
                "Track Status": df["fragment"].map({True: "Fragment (<2s)", False: "Stable"})
            })
            st.dataframe(df_display, use_container_width=True)

            st.subheader("Top 10 tracks by estimated distance (meters)")
            top = df.head(10).copy()
            top["track_id"] = top["track_id"].astype(str)
            top["dist_m"] = (top["distance_px"] * M_PER_PX).round(1)
            st.bar_chart(top.set_index("track_id")["dist_m"])

        else:
            st.dataframe(df, use_container_width=True)
            st.caption("One row = one tracker ID. The same player can appear under several IDs, so these are "
                       "tracks, not people. Units are pixels and pixels per second.")
            st.subheader("Top 10 tracks by distance (px)")
            top = df.head(10).copy()
            top["track_id"] = top["track_id"].astype(str)
            st.bar_chart(top.set_index("track_id")["distance_px"])

        with st.expander("📖 Parameter Guide: What do these metrics mean?"):
            st.markdown("""
            - **Track ID**: Unique ID assigned by ByteTrack to track an individual entity across consecutive frames.
            - **Role**: Entity category detected by YOLOv8 (`Player`, `Goalkeeper`, or `Referee`).
            - **Team**: Kit classification based on jersey torso color clustering (`Team 0` green vs `Team 1` white).
            - **Time Active / frames_tracked**: Duration the player was visible on camera in seconds (`frames / 25 FPS`).
            - **Est. Distance / distance_px**: Total movement across the pitch. Converted from pixel space to **meters** using the player height ratio ($1.75$m average height).
            - **Avg Speed & Top Sprint / px_s**: Average movement speed and peak sprint burst converted into **km/h**.
            - **Position Jitters / glitches**: Sudden single-step position jumps ($>120$ px) filtered out to prevent bounding box jitter.
            - **Track Status / fragment**: Tracks lasting $< 2.0$ seconds are flagged as fragments (brief edge-of-frame appearances or tracker switches).
            """)

with tab_teams:
    if teams is not None:
        t = teams.rename(columns={"players": "tracks"})
        st.subheader("Team Summary")

        # Converted real-world units table
        t_friendly = pd.DataFrame({
            "Team": t.index if "team" not in t.columns else t["team"],
            "Tracks": t["tracks"],
            "Total Est. Distance": (t["total_distance_px"] * M_PER_PX).round(1).astype(str) + " m",
            "Avg Est. Distance": (t["avg_distance_px"] * M_PER_PX).round(1).astype(str) + " m",
            "Avg Speed": (t["avg_speed_px_s"] * M_PER_PX * 3.6).round(1).astype(str) + " km/h",
            "Top Sprint": (t["top_speed_px_s"] * M_PER_PX * 3.6).round(1).astype(str) + " km/h",
        })
        st.dataframe(t_friendly, use_container_width=True)

        with st.expander("View Raw Pixel Units Table"):
            st.dataframe(t)
            st.caption("Average speeds are almost equal because camera panning moves everyone across the image. "
                       "Differences in distance depend on track counts and ID switches.")

    c1, c2 = st.columns(2)
    with c1:
        show_image(OUT / "heatmap_team0.png", "Team 0 (green kit): Tactical pitch occupancy heatmap")
    with c2:
        show_image(OUT / "heatmap_team1.png", "Team 1 (white kit): Tactical pitch occupancy heatmap")

    if holders is not None and len(holders):
        st.subheader("Seconds with the ball (live play)")
        h = holders.head(8).copy()
        h["track_id"] = h["track_id"].astype(str)
        st.bar_chart(h.set_index("track_id")["seconds_with_ball"])

with tab_model:
    st.subheader("Detector: YOLOv8m fine-tuned on Roboflow football dataset")
    st.write("1,078 training, 38 validation and 13 test images; classes: ball, goalkeeper, player, referee.")
    a, b = st.columns(2)
    with a:
        st.write("Validation split")
        st.dataframe(VAL)
    with b:
        st.write("Test split")
        st.dataframe(TEST)

    with st.expander("📖 How to Read Model Accuracy Metrics"):
        st.markdown("""
        - **Precision**: When the model predicts a class, how often is it right? ($98.0\%$ for players means only $2\%$ false positives).
        - **Recall**: Out of all actual players/balls on screen, how many did the model detect? ($97.7\%$ for players; lower for the small ball at $42.9\%$).
        - **mAP50**: Mean Average Precision at $0.50$ Intersection-over-Union (IoU). A score of $0.991$ for players means near-perfect bounding boxes.
        - **mAP50-95**: The strict metric measuring detection quality across multiple IoU overlap thresholds ($0.50$ to $0.95$).
        """)

    a, b = st.columns(2)
    with a:
        show_image(EVAL / "results.png", "Training & Validation Loss Curves (results.png)")
        st.caption("Shows box_loss, cls_loss, and dfl_loss decreasing over epochs as the detector learns.")
    with b:
        show_image(EVAL / "confusion_matrix.png", "Confusion Matrix (confusion_matrix.png)")
        st.caption("Strong diagonal line confirms minimal misclassification between player, referee, and goalkeeper.")

    show_image(EVAL / "val_batch0_pred.jpg", "Example Validation Predictions (val_batch0_pred.jpg)")
    st.caption("Sample validation batch showing YOLOv8 bounding box predictions and confidence scores.")

with tab_limits:
    st.markdown("""
    - **Camera Movement**: Distances and speeds are measured in image space. Because broadcast cameras pan and zoom, player speeds reflect both running and camera motion.
    - **Tracker ID Swaps**: ByteTrack may occasionally swap player IDs when players cross paths or overlap closely during tackles.
    - **Ball Occlusion**: The football is small and moves fast, detected in ~79% of raw frames; gaps are bridged with linear interpolation.
    - **Possession Definition**: Possession uses a spatial proximity rule (ball within $80$ px of a player's feet) with a 3-frame hysteresis buffer, not physiological control.
    - **Dataset Scope**: Tested on broadcast clips to illustrate computer vision methodology.
    """)