# detector.py

from ultralytics import YOLO
import cv2
import numpy as np


class Detector:
    def __init__(self, config):
        self.config = config
        self.model_type = config["model_type"]
        self.conf_threshold = config["confidence_threshold"]
        self.target_classes = set(config["target_classes"])

        if self.model_type == "yolov3_opencv":
            self.net = cv2.dnn.readNetFromDarknet(
                config["cfg_path"],
                config["weights_path"]
            )
            self.classes = self._load_classes(config["classes_path"])
            self.output_layer_names = self._get_output_layers()
            self.input_width = config["input_width"]
            self.input_height = config["input_height"]
            self.nms_threshold = config["nms_threshold"]
        elif self.model_type in {"yolov5_ultralytics", "yolov8_ultralytics", "yolo26_ultralytics"}:
            self.model = YOLO(config["weights_path"])
            self.class_names = self.model.names
        else:
            raise NotImplementedError(f"Unsupported model type: {self.model_type}")

    def _load_classes(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def _get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = self.net.getUnconnectedOutLayers()
        return [layer_names[i - 1] for i in output_layers.flatten()]

    def _compute_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return [int((x1 + x2) / 2), int((y1 + y2) / 2)]

    def _detect_yolov3(self, frame):
        if frame.dtype != np.uint8:
            input_frame = (frame * 255).astype(np.uint8)
        else:
            input_frame = frame.copy()

        h, w = input_frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            input_frame,
            scalefactor=1 / 255.0,
            size=(self.input_width, self.input_height),
            swapRB=True,
            crop=False
        )

        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layer_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.conf_threshold:
                    continue

                class_name = self.classes[class_id]
                if class_name not in self.target_classes:
                    continue

                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                box_width = int(detection[2] * w)
                box_height = int(detection[3] * h)

                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)

                boxes.append([x, y, box_width, box_height])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.conf_threshold,
            self.nms_threshold
        )

        detections = []

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w, x + bw)
                y2 = min(h, y + bh)
                bbox = [x1, y1, x2, y2]

                detections.append({
                    "class_id": class_ids[i],
                    "class_name": self.classes[class_ids[i]],
                    "confidence": float(confidences[i]),
                    "bbox": bbox,
                    "centroid": self._compute_centroid(bbox)
                })

        return detections

    def _detect_ultralytics(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=0,
            verbose=False
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = self.class_names[class_id]
                confidence = float(box.conf.item())

                if class_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = [int(x1), int(y1), int(x2), int(y2)]

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "centroid": self._compute_centroid(bbox)
                })

        return detections

    def detect(self, frame):
        if self.model_type == "yolov3_opencv":
            return self._detect_yolov3(frame)
        return self._detect_ultralytics(frame)