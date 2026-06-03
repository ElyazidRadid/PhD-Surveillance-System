# tracker.py

import numpy as np
import supervision as sv


class Tracker:
    def __init__(self, config):
        self.config = config

        self.tracker = sv.ByteTrack(
            track_activation_threshold=config["track_activation_threshold"],
            lost_track_buffer=config["lost_track_buffer"],
            minimum_matching_threshold=config["minimum_matching_threshold"],
            frame_rate=config["frame_rate"]
        )

    def update(self, detections):
        """
        Input:
            detections: list of dicts from detector.py
        Output:
            tracked_objects: list of dicts with track_id added
        """

        if not detections:
            return []

        xyxy = []
        confidences = []
        class_ids = []

        for det in detections:
            xyxy.append(det["bbox"])
            confidences.append(det["confidence"])
            class_ids.append(det["class_id"])

        xyxy = np.array(xyxy, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidences,
            class_id=class_ids
        )

        tracked = self.tracker.update_with_detections(sv_detections)

        tracked_objects = []

        for i in range(len(tracked.xyxy)):
            bbox = tracked.xyxy[i].astype(int).tolist()
            confidence = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1

            x1, y1, x2, y2 = bbox
            centroid = [int((x1 + x2) / 2), int((y1 + y2) / 2)]

            tracked_objects.append({
                "track_id": track_id,
                "class_id": class_id,
                "class_name": "person",   # current benchmark is person-only
                "confidence": confidence,
                "bbox": bbox,
                "centroid": centroid
            })

        return tracked_objects