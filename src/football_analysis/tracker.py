from collections import Counter

import numpy as np
import pandas as pd
import supervision as sv

from . import config

PEOPLE = ("player", "goalkeeper", "referee")


def track_people(det):
    """Run ByteTrack on players, goalkeepers and referees together.
    Returns one dict per frame: {track_id: {"bbox", "cls", "role"}}.
    Each track gets a single role (majority vote), so IDs never flip between classes."""
    names = det["names"]
    people_ids = [i for i, n in names.items() if n in PEOPLE]

    tracker = sv.ByteTrack(
        track_activation_threshold=config.TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=config.LOST_TRACK_BUFFER,
        minimum_matching_threshold=config.MATCH_THRESHOLD,
        frame_rate=int(round(det["fps"])),
        minimum_consecutive_frames=config.MIN_CONSEC_FRAMES,
    )

    per_frame = []
    for f in det["frames"]:
        mask = np.isin(f["cls"], people_ids)
        d = sv.Detections(
            xyxy=f["xyxy"][mask],
            confidence=f["conf"][mask],
            class_id=f["cls"][mask],
        )
        d = d.with_nms(threshold=0.6, class_agnostic=True)   # one box per person
        d = tracker.update_with_detections(d)
        frame_tracks = {}
        for xyxy, tid, cid in zip(d.xyxy, d.tracker_id, d.class_id):
            frame_tracks[int(tid)] = {"bbox": xyxy.astype(float), "cls": int(cid)}
        per_frame.append(frame_tracks)

    # one role per track: the class it was detected as most often
    votes = {}
    for ft in per_frame:
        for tid, t in ft.items():
            votes.setdefault(tid, []).append(t["cls"])
    role_of = {tid: names[Counter(v).most_common(1)[0][0]] for tid, v in votes.items()}
    for ft in per_frame:
        for tid, t in ft.items():
            t["role"] = role_of[tid]
    return per_frame

def best_ball_per_frame(det):
    """Keep only the highest-confidence ball box in each frame (or None)."""
    ball_id = [i for i, n in det["names"].items() if n == "ball"][0]
    out = []
    for f in det["frames"]:
        idx = np.where(f["cls"] == ball_id)[0]
        if len(idx) == 0:
            out.append(None)
        else:
            k = idx[np.argmax(f["conf"][idx])]
            out.append(f["xyxy"][k].astype(float))
    return out


def clean_ball(raw, window=None, outlier_px=None, max_gap=None):
    """1) drop ball boxes far from the local median position (false positives)
       2) linearly interpolate gaps of at most max_gap frames
       Longer gaps stay empty (None).
       Returns (cleaned_list, stats)."""
    window = window or config.BALL_ROLLING_WINDOW
    outlier_px = outlier_px or config.BALL_OUTLIER_PX
    max_gap = max_gap if max_gap is not None else config.MAX_BALL_GAP

    arr = np.full((len(raw), 4), np.nan)
    for i, b in enumerate(raw):
        if b is not None:
            arr[i] = b
    df = pd.DataFrame(arr, columns=["x1", "y1", "x2", "y2"])
    before = int(df["x1"].notna().sum())

    # 1) outlier removal
    cx = (df["x1"] + df["x2"]) / 2
    cy = (df["y1"] + df["y2"]) / 2
    med_x = cx.rolling(window, center=True, min_periods=1).median()
    med_y = cy.rolling(window, center=True, min_periods=1).median()
    bad = np.hypot(cx - med_x, cy - med_y) > outlier_px
    df.loc[bad, :] = np.nan
    after_filter = int(df["x1"].notna().sum())

    # 2) interpolate, then blank out gaps that were too long
    isna = df["x1"].isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    long_gap = isna & (run_len > max_gap)
    filled = df.interpolate(method="linear", limit_area="inside")
    filled.loc[long_gap, :] = np.nan

    cleaned = [None if np.isnan(r[0]) else r.copy() for r in filled.to_numpy()]
    n = len(raw)
    after = sum(c is not None for c in cleaned)
    stats = {
        "frames": n,
        "raw_detected": before,
        "outliers_removed": before - after_filter,
        "final_with_ball": after,
        "raw_rate_%": round(100 * before / n, 1),
        "final_rate_%": round(100 * after / n, 1),
    }
    return cleaned, stats


def track_summary(per_frame):
    """Quick fragmentation check for the report."""
    roles = {}
    lengths = Counter()
    for ft in per_frame:
        for tid, t in ft.items():
            roles[tid] = t["role"]
            lengths[tid] += 1
    print("Unique track IDs:", len(roles), dict(Counter(roles.values())))
    long_tracks = sum(1 for c in lengths.values() if c >= 50)
    print("Tracks lasting >= 50 frames:", long_tracks)
    print("Max people in one frame:", max(len(ft) for ft in per_frame))

