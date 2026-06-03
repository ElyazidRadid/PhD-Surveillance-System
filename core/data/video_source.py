# video_source.py

import cv2
import time
import os
from glob import glob


class VideoSource:
    def __init__(self, source=0, target_fps=None, use_frame_skip=False, frame_skip_interval=1):
        self.source = source
        self.target_fps = target_fps
        self.use_frame_skip = use_frame_skip
        self.frame_skip_interval = frame_skip_interval

        self.frame_id = 0
        self.last_read_time = 0.0
        self.frame_interval = 1.0 / self.target_fps if self.target_fps else None

        self.mode = None
        self.cap = None
        self.image_files = []
        self.image_index = 0

        # Case 1: source is a folder of images
        if isinstance(source, str) and os.path.isdir(source):
            self.mode = "image_folder"
            self.image_files = sorted(
                glob(os.path.join(source, "*.jpg")) +
                glob(os.path.join(source, "*.png")) +
                glob(os.path.join(source, "*.jpeg"))
            )

            if not self.image_files:
                raise ValueError(f"No image files found in folder: {source}")

        # Case 2: source is webcam / video path / RTSP
        else:
            self.mode = "video"
            self.cap = cv2.VideoCapture(source)

            if not self.cap.isOpened():
                raise ValueError(f"Could not open video source: {source}")

    def is_opened(self):
        if self.mode == "image_folder":
            return self.image_index < len(self.image_files)
        return self.cap.isOpened()

    def read(self):
        if self.frame_interval is not None:
            current_time = time.time()
            elapsed = current_time - self.last_read_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)

        # Image sequence mode
        if self.mode == "image_folder":
            if self.image_index >= len(self.image_files):
                return False, None

            image_path = self.image_files[self.image_index]
            frame = cv2.imread(image_path)

            if frame is None:
                return False, None

            self.image_index += 1
            self.frame_id += 1
            self.last_read_time = time.time()

            if self.use_frame_skip and self.frame_skip_interval > 1:
                if self.frame_id % self.frame_skip_interval != 0:
                    return self.read()

            frame_data = {
                "frame": frame,
                "frame_id": self.frame_id,
                "timestamp": self.last_read_time,
                "path": image_path
            }

            return True, frame_data

        # Video mode
        success, frame = self.cap.read()
        if not success:
            return False, None

        self.frame_id += 1
        self.last_read_time = time.time()

        if self.use_frame_skip and self.frame_skip_interval > 1:
            if self.frame_id % self.frame_skip_interval != 0:
                return self.read()

        frame_data = {
            "frame": frame,
            "frame_id": self.frame_id,
            "timestamp": self.last_read_time,
            "path": None
        }

        return True, frame_data

    def release(self):
        if self.cap is not None:
            self.cap.release()