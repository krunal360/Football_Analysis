import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import config
from .metrics import dead_ball_flags


def team_summary(stats_df):
    """Per-team table from stable tracks only (fragments excluded)."""
    s = stats_df[(~stats_df["fragment"]) & (stats_df["team"].isin([0, 1]))]
    g = s.groupby("team").agg(
        players=("track_id", "count"),
        total_distance_px=("distance_px", "sum"),
        avg_distance_px=("distance_px", "mean"),
        avg_speed_px_s=("avg_speed_px_s", "mean"),
        top_speed_px_s=("max_speed_px_s", "max"),
    ).round(1)
    g.index = g.index.map({0: "Team 0 (green kit)", 1: "Team 1 (white kit)"})
    return g


def holder_time(holders, people, fps, dead):
    """Seconds each player held the ball in live play (dead-ball frames excluded)."""
    counts = {}
    for i, h in enumerate(holders):
        if h is not None and not dead[i]:
            counts[h] = counts.get(h, 0) + 1
    rows = []
    for tid, n in counts.items():
        team = next((people[i][tid]["team"] for i in range(len(people)) if tid in people[i]), -1)
        rows.append({"track_id": tid, "team": team, "seconds_with_ball": round(n / fps, 1)})
    if not rows:
        return pd.DataFrame(columns=["track_id", "team", "seconds_with_ball"])
    return pd.DataFrame(rows).sort_values("seconds_with_ball", ascending=False).reset_index(drop=True)


def possession_changes(holders, people, dead):
    """How many times the ball moved from one team's player to the other's, in live play."""
    last_team, seq = {}, []
    for i, h in enumerate(holders):
        for tid, t in people[i].items():
            last_team[tid] = t["team"]
        if h is None or dead[i]:
            continue
        tm = last_team.get(h)
        if tm in (0, 1):
            seq.append(tm)
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


import cv2
import numpy as np


def draw_pitch_background(ax, width, height):
    """Draw a professional soccer pitch canvas with turf stripes and field markings."""
    ax.set_facecolor("#0e2417")
    num_stripes = 12
    stripe_w = width / num_stripes
    for i in range(num_stripes):
        col = "#112c1d" if i % 2 == 0 else "#0c2014"
        ax.axvspan(i * stripe_w, (i + 1) * stripe_w, facecolor=col, alpha=0.95, zorder=1)

    pad_x, pad_y = 60, 45
    pw, ph = width - 2 * pad_x, height - 2 * pad_y
    line_col = (1.0, 1.0, 1.0, 0.45)
    lw = 1.8

    # Outer pitch boundary
    rect = plt.Rectangle((pad_x, pad_y), pw, ph, edgecolor=line_col, facecolor="none", linewidth=lw, zorder=2)
    ax.add_patch(rect)

    # Halfway line
    mid_x = width / 2
    ax.plot([mid_x, mid_x], [pad_y, pad_y + ph], color=line_col, linewidth=lw, zorder=2)

    # Center circle and spot
    center_circle = plt.Circle((mid_x, height / 2), 140, edgecolor=line_col, facecolor="none", linewidth=lw, zorder=2)
    ax.add_patch(center_circle)
    ax.plot(mid_x, height / 2, "o", color=line_col, markersize=4, zorder=2)

    # Penalty areas (18-yard boxes)
    box_w, box_h = 260, 480
    box_y = (height - box_h) / 2
    ax.add_patch(plt.Rectangle((pad_x, box_y), box_w, box_h, edgecolor=line_col, facecolor="none", linewidth=lw, zorder=2))
    ax.add_patch(plt.Rectangle((width - pad_x - box_w, box_y), box_w, box_h, edgecolor=line_col, facecolor="none", linewidth=lw, zorder=2))

    # Goal areas (6-yard boxes)
    gbox_w, gbox_h = 100, 240
    gbox_y = (height - gbox_h) / 2
    ax.add_patch(plt.Rectangle((pad_x, gbox_y), gbox_w, gbox_h, edgecolor=line_col, facecolor="none", linewidth=lw, zorder=2))
    ax.add_patch(plt.Rectangle((width - pad_x - gbox_w, gbox_y), gbox_w, gbox_h, edgecolor=line_col, facecolor="none", linewidth=lw, zorder=2))

    # Penalty spots
    ax.plot(pad_x + 190, height / 2, "o", color=line_col, markersize=4, zorder=2)
    ax.plot(width - pad_x - 190, height / 2, "o", color=line_col, markersize=4, zorder=2)


def heatmap(people, team, out_path, size=(1920, 1080)):
    w, h = size
    xs, ys = [], []
    for ft in people:
        for t in ft.values():
            if t["role"] != "referee" and t["team"] == team:
                xs.append((t["bbox"][0] + t["bbox"][2]) / 2)
                ys.append(t["bbox"][3])

    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="#08140c")
    draw_pitch_background(ax, w, h)

    team_label = "Team 0 (Green Kit)" if team == 0 else ("Team 1 (White Kit)" if team == 1 else f"Team {team}")

    if len(xs) > 5:
        # High resolution 2D histogram
        bins_x, bins_y = 384, 216
        hist, _, _ = np.histogram2d(xs, ys, bins=[bins_x, bins_y], range=[[0, w], [0, h]])
        density = hist.T

        # Gaussian smoothing for smooth, organic football heat contours
        smoothed = cv2.GaussianBlur(density.astype(np.float32), (0, 0), sigmaX=5.5, sigmaY=5.5)
        peak = smoothed.max()
        norm_heat = (smoothed / peak) if peak > 0 else smoothed

        # Mask out low density areas so the pitch grass shows through
        masked_heat = np.ma.masked_where(norm_heat < 0.04, norm_heat)

        im = ax.imshow(
            masked_heat,
            extent=[0, w, h, 0],
            cmap="turbo",
            alpha=0.75,
            interpolation="bicubic",
            zorder=3,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
        cbar.set_ticks([0.1, 0.9])
        cbar.set_ticklabels(["Low Activity", "High Activity"])
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#cfd8dc", fontsize=8.5)
        cbar.outline.set_edgecolor("#263238")

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()

    ax.set_title(
        f"{team_label} — Tactical Heatmap (Smoothed Density)",
        color="white",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def run_extra(people, ball, holders, stats_df, fps):
    out = config.OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    dead = dead_ball_flags(ball)

    ts = team_summary(stats_df)
    ts.to_csv(out / "team_summary.csv")

    ht = holder_time(holders, people, fps, dead)
    ht.to_csv(out / "holder_time.csv", index=False)

    changes = possession_changes(holders, people, dead)
    with open(out / "extra_stats.json", "w") as f:
        json.dump({"possession_changes_live_play": changes}, f, indent=2)

    for team in (0, 1):
        heatmap(people, team, out / f"heatmap_team{team}.png")

    print("\nTeam summary (stable tracks, pixel units):")
    print(ts.to_string())
    print("\nTop ball holders (live play):")
    print(ht.head(5).to_string(index=False))
    print("\nPossession changes in live play:", changes)