# utils/logger.py

import json
import os


class DetectionLogger:
    def __init__(self, output_path):
        self.output_path = output_path
        self.data = []

    def log(self, frame_id, detections):
        self.data.append({
            "frame_id": int(frame_id),
            "detections": detections
        })

    def save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.data, f, indent=2)