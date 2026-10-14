from collections import Counter

import cv2
import numpy as np
from sklearn.cluster import KMeans

from . import config


def shirt_color(frame, bbox):
    """Return the shirt colour (Lab, 3 floats) from the torso crop, or None."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, int(x1)), min(w_img, int(x2))
    y1, y2 = max(0, int(y1)), min(h_img, int(y2))
    h, w = y2 - y1, x2 - x1
    if h < 8 or w < 5:
        return None

    top = y1 + int(config.TORSO_TOP * h)
    bottom = y1 + int(config.TORSO_BOTTOM * h)
    left = x1 + int(config.TORSO_SIDE * w)
    right = x2 - int(config.TORSO_SIDE * w)
    crop = frame[top:bottom, left:right]
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return None

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    try:
        km = KMeans(n_clusters=2, n_init=2, random_state=0).fit(lab)
    except Exception:
        return lab.mean(axis=0)

    labels = km.labels_.reshape(crop.shape[:2])
    corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
    background = Counter(corners).most_common(1)[0][0]     # the corner colour = background
    return km.cluster_centers_[1 - background]


class TeamAssigner:
    def __init__(self):
        self.km = None
        self.team_colors_bgr = {}      # {0: (b,g,r), 1: (b,g,r)} for drawing
        self.consistency = None        # mean share of samples agreeing within a track

    def fit_assign(self, video_path, people):
        """Sets t["team"] on every track entry, per frame (0, 1, or -1 for referees/unknown).
        The team is a majority vote over a short window around each frame, so an ID swap
        between two players corrects itself within about a second."""
        colors = {}                    # track id -> list of (frame index, Lab colour)

        cap = cv2.VideoCapture(str(video_path))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(people):
                break
            if i % config.TEAM_SAMPLE_EVERY == 0:
                for tid, t in people[i].items():
                    if t["role"] != "player":
                        continue
                    c = shirt_color(frame, t["bbox"])
                    if c is not None:
                        colors.setdefault(tid, []).append((i, c))
            i += 1
        cap.release()

        all_colors = np.array([c for lst in colors.values() for _, c in lst])
        self.km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(all_colors)

        # stable numbering: team 0 = greener cluster (lower Lab "a" value)
        order = np.argsort(self.km.cluster_centers_[:, 1])
        remap = {int(old): new for new, old in enumerate(order)}

        for old_k, center in enumerate(self.km.cluster_centers_):
            lab = np.uint8([[np.clip(center, 0, 255)]])
            self.team_colors_bgr[remap[old_k]] = tuple(
                int(v) for v in cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0])

        # team of every colour sample, per track
        samples, agree = {}, []
        for tid, lst in colors.items():
            frs = np.array([f for f, _ in lst])
            raw = self.km.predict(np.array([c for _, c in lst]))
            teams = np.array([remap[int(p)] for p in raw])
            samples[tid] = (frs, teams)
            agree.append(max(np.mean(teams == 0), np.mean(teams == 1)))
        self.consistency = float(np.mean(agree))

        # per-frame team = majority of the samples within +/- TEAM_WINDOW_FRAMES
        for i, ft in enumerate(people):
            for tid, t in ft.items():
                if t["role"] == "referee":
                    t["team"] = -1
                elif t["role"] == "player":
                    if tid not in samples:
                        t["team"] = -1
                        continue
                    frs, teams = samples[tid]
                    near = np.abs(frs - i) <= config.TEAM_WINDOW_FRAMES
                    vote = teams[near] if near.any() else teams
                    t["team"] = int(np.bincount(vote, minlength=2).argmax())

        self._assign_goalkeepers(people)
        return people

    @staticmethod
    def _assign_goalkeepers(people):
        """A goalkeeper joins the team whose players are, on average, closer in x at each frame."""
        votes = {}
        for ft in people:
            xs = {0: [], 1: []}
            for t in ft.values():
                if t["role"] == "player" and t["team"] in (0, 1):
                    xs[t["team"]].append((t["bbox"][0] + t["bbox"][2]) / 2)
            if not xs[0] or not xs[1]:
                continue
            m0, m1 = np.mean(xs[0]), np.mean(xs[1])
            for tid, t in ft.items():
                if t["role"] == "goalkeeper":
                    gx = (t["bbox"][0] + t["bbox"][2]) / 2
                    votes.setdefault(tid, []).append(0 if abs(gx - m0) < abs(gx - m1) else 1)
        team_of = {tid: Counter(v).most_common(1)[0][0] for tid, v in votes.items()}
        for ft in people:
            for tid, t in ft.items():
                if t["role"] == "goalkeeper":
                    t["team"] = team_of.get(tid, -1)