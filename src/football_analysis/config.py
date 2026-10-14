from pathlib import Path

# Project root = Football_Analysis/ (two levels above this file)
ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "data" / "models" / "best.pt"
CACHE_DIR = ROOT / "outputs" / "cache"
OUTPUT_DIR = ROOT / "outputs"

# Detection
IMGSZ = 1280
CONF_THRESHOLDS = {          # per-class minimum confidence
    "ball": 0.10,            # low on purpose: we keep only the best ball per frame later
    "goalkeeper": 0.25,
    "player": 0.25,
    "referee": 0.25,
}
MIN_CONF = min(CONF_THRESHOLDS.values())

# Tracking (ByteTrack)
TRACK_ACTIVATION_THRESHOLD = 0.25
LOST_TRACK_BUFFER = 50          # frames a lost track is kept alive (2 s at 25 fps)
MATCH_THRESHOLD = 0.8

# Ball cleaning
BALL_ROLLING_WINDOW = 25        # frames, for the median filter
BALL_OUTLIER_PX = 150           # distance from local median above which a ball is dropped
MAX_BALL_GAP = 15               # longest gap (frames) we interpolate across

# Team assignment
TEAM_SAMPLE_EVERY = 5            # use every 5th frame for colour sampling
TORSO_TOP, TORSO_BOTTOM = 0.15, 0.55   # fraction of box height
TORSO_SIDE = 0.20                # trim 20% off each side of the box width

DISTANCE_STEP = 5            # measure distance every 5 frames (reduces box jitter)
SMOOTH_WINDOW = 5            # moving-average window on foot positions
MAX_JUMP_PX = 120            # a move bigger than this in one step is treated as a glitch
MIN_TRACK_FRAMES = 50        # tracks shorter than this are flagged as fragments

# Possession
POSSESSION_THRESHOLD = 80    # max ball-to-foot distance (px) to count as having the ball
POSSESSION_MIN_HOLD = 3 

TRACK_ACTIVATION_THRESHOLD = 0.35
MIN_CONSEC_FRAMES = 5   

TEAM_WINDOW_FRAMES = 25       # +/- 1 second (at 25 fps) for the per-frame team vote

DEAD_BALL_WINDOW = 25        # frames (1 s)
DEAD_BALL_MAX_MOVE = 40      # ball moved less than this many px over the window = stationary
DEAD_BALL_MIN_FRAMES = 50      # a stationary stretch must last 2 s to count as a restart


# Extra statistics
PLAYER_HEIGHT_M = 1.75        # pixels -> approximate metres, using the player's box height
SPEED_LABELS = "all"          # "all" = every player, "holder" = only the ball holder, "off" = none