import pickle
import cv2
import numpy as np
from ultralytics import YOLO

from . import config


def load_model(path=config.MODEL_PATH):
    return YOLO(str(path))


def detect_video(video_path, model=None, cache_name="detections.pkl",
                 use_cache=True, max_frames=None):
    """Run YOLO over a video. Returns a dict:
       {"names": {id: name}, "fps": float, "frames": [ {"xyxy","conf","cls"}, ... ]}
    """
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = config.CACHE_DIR / cache_name

    if use_cache and cache_path.exists():
        with open(cache_path, "rb") as f:
            print(f"Loaded cached detections from {cache_path}")
            return pickle.load(f)

    model = model or load_model()
    names = model.names
    thresholds = np.array([config.CONF_THRESHOLDS[names[i]] for i in range(len(names))])

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    frames = []
    stream = model.predict(source=str(video_path), stream=True, imgsz=config.IMGSZ,
                           conf=config.MIN_CONF, verbose=False)
    for i, r in enumerate(stream):
        if max_frames is not None and i >= max_frames:
            break
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        keep = conf >= thresholds[cls]          # per-class threshold
        frames.append({"xyxy": xyxy[keep], "conf": conf[keep], "cls": cls[keep]})

    result = {"names": names, "fps": fps, "frames": frames}
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    print(f"Processed {len(frames)} frames, saved to {cache_path}")
    return result


def summarize(det):
    """Quick sanity numbers for the report."""
    names = det["names"]
    n = len(det["frames"])
    print(f"Frames: {n}")
    for cid, cname in names.items():
        counts = [int((f["cls"] == cid).sum()) for f in det["frames"]]
        present = sum(c > 0 for c in counts)
        print(f"{cname:11s} avg/frame: {np.mean(counts):5.1f}   "
              f"frames with >=1: {present}/{n} ({100 * present / n:.0f}%)")