import cv2
import numpy as np

from . import config

TEAM_DRAW = {0: (0, 140, 255), 1: (255, 100, 0)}    # BGR: team 0 orange, team 1 blue
REF_COLOR = (0, 255, 255)                            # yellow
NEUTRAL = (180, 180, 180)


def _foot(b):
    return int((b[0] + b[2]) / 2), int(b[3])


def draw_video(video_path, out_path, people, ball, holders, fps, trail_len=25):
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    life = {}
    for ft in people:
        for tid in ft:
            life[tid] = life.get(tid, 0) + 1
    stable = {tid for tid, n in life.items() if n >= config.MIN_TRACK_FRAMES}

    history = {}
    team_of = {}                      # latest known team per track, updated every frame
    c0 = c1 = 0
    for i in range(len(people)):
        ok, frame = cap.read()
        if not ok:
            break

        for tid, t in people[i].items():
            team_of[tid] = t["team"]

        holder = holders[i]
        if holder is not None:
            if team_of.get(holder) == 0:
                c0 += 1
            elif team_of.get(holder) == 1:
                c1 += 1

        for tid, t in people[i].items():
            if t["role"] != "referee":
                history.setdefault(tid, []).append(_foot(t["bbox"]))

        for tid, t in people[i].items():
            x1, y1, x2, y2 = t["bbox"]
            cx, cy = _foot(t["bbox"])
            bw = max(int(x2 - x1), 12)
            if t["role"] == "referee":
                col = REF_COLOR
            else:
                col = TEAM_DRAW.get(t["team"], NEUTRAL)

            # trajectory (stable tracks only, to keep the picture clean)
            if tid in stable and t["role"] != "referee":
                pts = np.array(history[tid][-trail_len:], dtype=np.int32).reshape(-1, 1, 2)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, col, 2)

            thick = 5 if tid == holder else 2
            cv2.ellipse(frame, (cx, cy), (bw // 2 + 4, max(bw // 4, 5)), 0, -45, 235, col, thick)

            if t["role"] != "referee":
                cv2.putText(frame, str(tid), (cx - 10, cy + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

                # approximate speed over the last 10 frames (0.4 s)
                show = config.SPEED_LABELS == "all" or (config.SPEED_LABELS == "holder" and tid == holder)
                if show and len(history[tid]) > 10:
                    px, py = history[tid][-11]
                    d_px = float(np.hypot(cx - px, cy - py))
                    m_per_px = config.PLAYER_HEIGHT_M / max(y2 - y1, 20)
                    kmh = d_px * m_per_px / (10 / fps) * 3.6
                    if kmh <= 40:                      # above this it is a tracking glitch, not running
                        cv2.putText(frame, f"~{kmh:.0f} km/h", (cx - 28, cy + 44),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            if tid == holder:
                cv2.circle(frame, (cx, int(y1) - 12), 7, (255, 255, 255), -1)

        b = ball[i]
        if b is not None:
            bx, by = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
            tri = np.array([[bx, by - 8], [bx - 9, by - 26], [bx + 9, by - 26]], np.int32)
            cv2.fillPoly(frame, [tri], (0, 255, 0))

        # possession panel (cumulative, all frames up to this one)
        total = c0 + c1
        p0 = 100 * c0 / total if total else 0
        p1 = 100 * c1 / total if total else 0
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (470, 140), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        cv2.putText(frame, "Possession so far", (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Team 0 (green kit): {p0:4.1f}%", (35, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEAM_DRAW[0], 2)
        cv2.putText(frame, f"Team 1 (white kit): {p1:4.1f}%", (35, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.65, TEAM_DRAW[1], 2)
        cv2.putText(frame, "Speed: approx., camera-dependent", (35, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(frame)
    cap.release()
    out.release()
    print("Saved", out_path)