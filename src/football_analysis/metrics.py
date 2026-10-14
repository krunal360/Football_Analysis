import json
from collections import Counter

import numpy as np
import pandas as pd

from . import config


def foot(bbox):
    """Foot position = bottom-centre of the box."""
    return (bbox[0] + bbox[2]) / 2, bbox[3]


def collect_positions(people):
    tracks = {}
    for i, ft in enumerate(people):
        for tid, t in ft.items():
            if t["role"] == "referee":
                continue
            x, y = foot(t["bbox"])
            rec = tracks.setdefault(tid, {"role": t["role"], "teams": [],
                                          "frames": [], "x": [], "y": []})
            rec["teams"].append(t["team"])
            rec["frames"].append(i)
            rec["x"].append(x)
            rec["y"].append(y)
    # one summary team per track: the team it had in most frames
    for rec in tracks.values():
        rec["team"] = Counter(rec["teams"]).most_common(1)[0][0]
    return tracks


def player_stats(people, fps):
    """Distance and speed per track, in pixels (not metres). Returns a DataFrame."""
    step = config.DISTANCE_STEP
    dt = step / fps
    rows = []
    for tid, r in collect_positions(people).items():
        df = pd.DataFrame({"x": r["x"], "y": r["y"]}, index=r["frames"])
        df = df.rolling(config.SMOOTH_WINDOW, center=True, min_periods=1).mean()
        sample = df[df.index % step == 0]
        fr = sample.index.to_numpy()
        xy = sample[["x", "y"]].to_numpy()

        if len(fr) < 2:
            dist, avg_speed, max_speed, glitches = 0.0, 0.0, 0.0, 0
        else:
            d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
            consecutive = np.diff(fr) == step
            ok = consecutive & (d <= config.MAX_JUMP_PX)
            glitches = int((consecutive & (d > config.MAX_JUMP_PX)).sum())
            dist = float(d[ok].sum())
            speeds = d[ok] / dt
            avg_speed = float(dist / (ok.sum() * dt)) if ok.sum() else 0.0
            max_speed = float(speeds.max()) if len(speeds) else 0.0

        rows.append({
            "track_id": tid,
            "role": r["role"],
            "team": r["team"],
            "frames_tracked": len(r["frames"]),
            "distance_px": round(dist, 1),
            "avg_speed_px_s": round(avg_speed, 1),
            "max_speed_px_s": round(max_speed, 1),
            "glitches": glitches,
            "fragment": len(r["frames"]) < config.MIN_TRACK_FRAMES,
        })
    return pd.DataFrame(rows).sort_values("distance_px", ascending=False).reset_index(drop=True)
def dead_ball_flags(ball):
    """True for frames where the ball has barely moved over the surrounding window."""
    n, w = len(ball), config.DEAD_BALL_WINDOW
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    for i, b in enumerate(ball):
        if b is not None:
            cx[i], cy[i] = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        xs, ys = cx[lo:hi], cy[lo:hi]
        if np.isnan(xs).all():
            continue
        spread = np.hypot(np.nanmax(xs) - np.nanmin(xs), np.nanmax(ys) - np.nanmin(ys))
        flags[i] = spread < config.DEAD_BALL_MAX_MOVE
        # keep only long stationary stretches (restarts); brief slow moments stay live play
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            if j - i >= config.DEAD_BALL_MIN_FRAMES:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def compute_possession(people, ball):
    """Per-frame ball holder (track id or None) + team possession summary."""
    # 1) nearest player to the ball, only if within the distance threshold
    raw = []
    for i, ft in enumerate(people):
        b = ball[i]
        if b is None:
            raw.append(None)
            continue
        bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        best, best_d = None, config.POSSESSION_THRESHOLD
        for tid, t in ft.items():
            if t["role"] == "referee":
                continue
            fx, fy = foot(t["bbox"])
            d = float(np.hypot(fx - bx, fy - by))
            if d < best_d:
                best, best_d = tid, d
        raw.append(best)

    # 2) hysteresis: a change of holder must last POSSESSION_MIN_HOLD frames
    holders, current, pending, count = [], None, None, 0
    for r in raw:
        if r == current:
            pending, count = None, 0
        else:
            if r == pending:
                count += 1
            else:
                pending, count = r, 1
            if count >= config.POSSESSION_MIN_HOLD:
                current, pending, count = pending, None, 0
        holders.append(current)

        # 3) team totals, live play only (dead-ball frames are not counted)
    dead = dead_ball_flags(ball)
    last_team, f0, f1, f0_all, f1_all = {}, 0, 0, 0, 0
    for i, h in enumerate(holders):
        for tid, t in people[i].items():
            last_team[tid] = t["team"]
        if h is None:
            continue
        tm = last_team.get(h)
        if tm == 0:
            f0_all += 1
        elif tm == 1:
            f1_all += 1
        if dead[i]:
            continue
        if tm == 0:
            f0 += 1
        elif tm == 1:
            f1 += 1
    total, total_all = f0 + f1, f0_all + f1_all
    summary = {
        "frames": len(holders),
        "dead_ball_frames": int(dead.sum()),
        "team0_frames": f0,
        "team1_frames": f1,
        "no_possession_frames": len(holders) - total,
        "team0_pct": round(100 * f0 / total, 1) if total else None,
        "team1_pct": round(100 * f1 / total, 1) if total else None,
        "team0_pct_all_frames": round(100 * f0_all / total_all, 1) if total_all else None,
        "team1_pct_all_frames": round(100 * f1_all / total_all, 1) if total_all else None,
    }
    return holders, summary

def save_stats(df, poss_summary, out_dir=None):
    out_dir = out_dir or config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "player_stats.csv", index=False)
    with open(out_dir / "stats.json", "w") as f:
        json.dump({"possession": poss_summary,
                   "top5_by_distance_px": df.head(5)[["track_id", "team", "distance_px"]].to_dict("records")},
                  f, indent=2)
    print("Saved", out_dir / "player_stats.csv", "and", out_dir / "stats.json")