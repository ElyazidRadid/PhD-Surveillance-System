# preprocessing.py

import cv2
import numpy as np


class Preprocessor:
    def __init__(self, config):
        self.config = config

    def resize(self, frame):
        return cv2.resize(
            frame,
            (self.config["resize_width"], self.config["resize_height"])
        )

    def apply_gaussian_blur(self, frame):
        return cv2.GaussianBlur(frame, self.config["gaussian_kernel"], 0)

    def apply_bilateral_filter(self, frame):
        return cv2.bilateralFilter(
            frame,
            self.config["bilateral_d"],
            self.config["bilateral_sigma_color"],
            self.config["bilateral_sigma_space"]
        )

    def normalize(self, frame):
        return frame.astype(np.float32) / 255.0

    def histogram_equalization(self, frame):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def apply(self, frame):
        processed = frame.copy()
        meta = {
            "resized": False,
            "gaussian_blur": False,
            "bilateral_filter": False,
            "normalized": False,
            "hist_eq": False
        }

        if self.config.get("resize", False):
            processed = self.resize(processed)
            meta["resized"] = True

        if self.config.get("gaussian_blur", False):
            processed = self.apply_gaussian_blur(processed)
            meta["gaussian_blur"] = True

        if self.config.get("bilateral_filter", False):
            processed = self.apply_bilateral_filter(processed)
            meta["bilateral_filter"] = True

        if self.config.get("hist_eq", False):
            if processed.dtype != np.uint8:
                temp = (processed * 255).astype(np.uint8)
                temp = self.histogram_equalization(temp)
                processed = temp
            else:
                processed = self.histogram_equalization(processed)
            meta["hist_eq"] = True

        if self.config.get("normalize", False):
            if processed.dtype == np.uint8:
                processed = self.normalize(processed)
            meta["normalized"] = True

        return processed, meta