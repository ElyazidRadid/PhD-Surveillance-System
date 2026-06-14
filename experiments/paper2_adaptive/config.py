# experiments/paper2_adaptive/config.py
"""Configuration for the spatially-adaptive thresholding study (Paper 2)."""

# --- Spatial grid ---
GRID_ROWS = 4
GRID_COLS = 4

# --- Threshold search grid (the candidate operating points) ---
# Detection logs must contain detections down to at least the minimum here.
THRESHOLD_GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# --- Matching ---
IOU_THRESHOLD = 0.5
TARGET_CLASS = "person"

# --- Frame geometry (MOT20 sequences are 1920x1080) ---
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# --- Data sources ---
# Map sequence -> (full-candidate detection log, ground-truth annotation).
# Uses the existing low-threshold (0.05) logs as the candidate pool.
SEQUENCES = {
    "MOT20-02": {
        "detections": "benchmark/results/yolov8s_0.05/MOT20-02.json",
        "annotations": "benchmark/annotations/MOT20-02.json",
    },
    "MOT20-05": {
        "detections": "benchmark/results/yolov8s_0.05/MOT20-05.json",
        "annotations": "benchmark/annotations/MOT20-05.json",
    },
}

# --- Output ---
OUTPUT_DIR = "experiments/paper2_adaptive/outputs"
