# experiments/paper2_phantom_recall/config.py
"""Configuration for the Phantom Recall (temporal recovery) study."""

# Operating threshold = the F1-optimal point from Paper 1 (recall is maxed here,
# so remaining false negatives are the hard, threshold-irrecoverable cases).
OPERATING_THRESHOLD = 0.05

# IoU for matching detections to ground truth (consistent with Paper 1).
IOU_THRESHOLD = 0.5

# IoU for deciding that an adjacent-frame detection covers the same object as a
# missed GT box (boxes move little frame-to-frame in dense crowds).
RECOVERY_IOU = 0.5

# Temporal window: an FN is "recoverable" if covered in t-k .. t+k (k=1 default).
TEMPORAL_WINDOW = 1

TARGET_CLASS = "person"

# Reuse Paper 1's low-threshold detection logs (full candidate set) + annotations.
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

# --- Recovery method ---
# IoU to link a detection to an existing tracklet across consecutive frames.
ASSOCIATION_IOU = 0.3
# Only interpolate gaps no longer than this many frames (precision/recall knob).
MAX_GAP = 3
# Confidence assigned to a recovered (interpolated) box.
RECOVERED_CONFIDENCE = 0.5

OUTPUT_DIR = "experiments/paper2_phantom_recall/outputs"
