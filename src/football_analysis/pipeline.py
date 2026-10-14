import argparse
from pathlib import Path

from . import config
from .detector import detect_video
from .tracker import track_people, best_ball_per_frame, clean_ball, track_summary
from .team_assigner import TeamAssigner
from .metrics import player_stats, compute_possession, save_stats
from .extra_stats import run_extra
from .visualizer import draw_video


def run(video_path, out_name="annotated.mp4", cache_name="detections.pkl", use_cache=True):
    video_path = Path(video_path)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    det = detect_video(video_path, cache_name=cache_name, use_cache=use_cache)
    people = track_people(det)
    track_summary(people)

    ball, ball_stats = clean_ball(best_ball_per_frame(det))
    print("Ball:", ball_stats)

    ta = TeamAssigner()
    people = ta.fit_assign(video_path, people)
    print("Team consistency:", round(ta.consistency, 3))

    stats_df = player_stats(people, det["fps"])
    holders, poss = compute_possession(people, ball)
    print("Possession:", poss)
    stable = stats_df[~stats_df["fragment"]]
    print("Stable tracks:", len(stable), "| median distance (px):", stable["distance_px"].median())
    save_stats(stats_df, poss)

    run_extra(people, ball, holders, stats_df, det["fps"])

    out_path = config.OUTPUT_DIR / out_name
    draw_video(video_path, out_path, people, ball, holders, det["fps"])
    return out_path


def main():
    p = argparse.ArgumentParser(description="Football match analysis pipeline")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="annotated.mp4")
    p.add_argument("--no-cache", action="store_true")
    a = p.parse_args()
    run(a.input, a.output, use_cache=not a.no_cache)


if __name__ == "__main__":
    main()