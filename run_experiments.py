# run_experiments.py

import os
import csv
import copy
import cv2

from config import VIDEO_CONFIG, PREPROCESS_CONFIG
from video_source import VideoSource
from preprocessing import Preprocessor
from detector import Detector
from utils.logger import DetectionLogger
from utils.timer import Timer
from evaluation.evaluator import evaluate_sequence


EXPERIMENT_MODELS = [
    {
        "label": "yolov3",
        "config": {
            "model_type": "yolov3_opencv",
            "cfg_path": "models/yolov3.cfg",
            "weights_path": "models/yolov3.weights",
            "classes_path": "models/coco.names",
            "confidence_threshold": 0.3,
            "nms_threshold": 0.4,
            "target_classes": ["person"],
            "input_width": 416,
            "input_height": 416
        }
    },
    {
        "label": "yolov5s",
        "config": {
            "model_type": "yolov5_ultralytics",
            "weights_path": "models/yolov5su.pt",
            "confidence_threshold": 0.3,
            "target_classes": ["person"]
        }
    },
    {
        "label": "yolov8n",
        "config": {
            "model_type": "yolov8_ultralytics",
            "weights_path": "models/yolov8n.pt",
            "confidence_threshold": 0.3,
            "target_classes": ["person"]
        }
    },
    {
        "label": "yolov8s",
        "config": {
            "model_type": "yolov8_ultralytics",
            "weights_path": "models/yolov8s.pt",
            "confidence_threshold": 0.3,
            "target_classes": ["person"]
        }
    },
    {
        "label": "yolov8x",
        "config": {
            "model_type": "yolov8_ultralytics",
            "weights_path": "models/yolov8x.pt",
            "confidence_threshold": 0.3,
            "target_classes": ["person"]
        }
    },
    {
        "label": "yolo26s",
        "config": {
            "model_type": "yolo26_ultralytics",
            "weights_path": "models/yolo26s.pt",
            "confidence_threshold": 0.3,
            "target_classes": ["person"]
        }
    }
]

THRESHOLDS = [0.50, 0.30, 0.20, 0.10, 0.05]
SEQUENCES = ["MOT20-02", "MOT20-05"]

INPUT_ROOT = "datasets/MOT20/train"
ANNOTATION_ROOT = "benchmark/annotations"
RESULT_ROOT = "benchmark/results"
CSV_OUTPUT = "benchmark/experiment_results.csv"

SHOW_PREVIEW = False


def ensure_csv_header(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model",
                "threshold",
                "sequence",
                "frames_processed",
                "avg_latency_ms",
                "fps",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "f1_score",
                "prediction_json",
                "annotation_json"
            ])


def append_result(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def convert_for_display(processed_frame, raw_frame):
    if processed_frame.dtype != raw_frame.dtype:
        return (processed_frame * 255).astype("uint8")
    return processed_frame.copy()


def draw_detections(frame, detections):
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f'{det["class_name"]}: {det["confidence"]:.2f}'
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            output,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )
    return output


def run_single_sequence(detector_config, model_label, threshold, sequence_name):
    source_path = os.path.join(INPUT_ROOT, sequence_name, "img1")
    annotation_json = os.path.join(ANNOTATION_ROOT, f"{sequence_name}.json")

    prediction_dir = os.path.join(RESULT_ROOT, f"{model_label}_{threshold}")
    os.makedirs(prediction_dir, exist_ok=True)
    prediction_json = os.path.join(prediction_dir, f"{sequence_name}.json")

    source = VideoSource(
        source=source_path,
        target_fps=VIDEO_CONFIG["target_fps"],
        use_frame_skip=VIDEO_CONFIG["use_frame_skip"],
        frame_skip_interval=VIDEO_CONFIG["frame_skip_interval"]
    )

    preprocessor = Preprocessor(PREPROCESS_CONFIG)
    detector = Detector(detector_config)
    logger = DetectionLogger(prediction_json)
    timer = Timer()

    frame_count = 0

    while source.is_opened():
        success, frame_data = source.read()
        if not success:
            break

        raw_frame = frame_data["frame"]
        frame_id = frame_data["frame_id"]

        processed_frame, _ = preprocessor.apply(raw_frame)

        timer.start()
        detections = detector.detect(processed_frame)
        timer.stop()

        logger.log(frame_id, detections)
        frame_count += 1

        if SHOW_PREVIEW:
            display_frame = convert_for_display(processed_frame, raw_frame)
            output_frame = draw_detections(display_frame, detections)
            cv2.imshow(f"{model_label} - {sequence_name}", output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    logger.save()
    source.release()
    cv2.destroyAllWindows()

    avg_latency = timer.average()
    fps = 1 / avg_latency if avg_latency > 0 else 0

    eval_results = evaluate_sequence(
        prediction_json=prediction_json,
        annotation_json=annotation_json,
        iou_threshold=0.5,
        target_class="person"
    )

    return {
        "frames_processed": frame_count,
        "avg_latency_ms": avg_latency * 1000,
        "fps": fps,
        "TP": eval_results["TP"],
        "FP": eval_results["FP"],
        "FN": eval_results["FN"],
        "precision": eval_results["precision"],
        "recall": eval_results["recall"],
        "f1_score": eval_results["f1_score"],
        "prediction_json": prediction_json,
        "annotation_json": annotation_json
    }


def main():
    ensure_csv_header(CSV_OUTPUT)

    for model_entry in EXPERIMENT_MODELS:
        model_label = model_entry["label"]

        for threshold in THRESHOLDS:
            detector_config = copy.deepcopy(model_entry["config"])
            detector_config["confidence_threshold"] = threshold

            print(f"\n=== Running {model_label} @ threshold={threshold} ===")

            for sequence_name in SEQUENCES:
                print(f"Sequence: {sequence_name}")

                result = run_single_sequence(
                    detector_config=detector_config,
                    model_label=model_label,
                    threshold=threshold,
                    sequence_name=sequence_name
                )

                append_result(CSV_OUTPUT, [
                    model_label,
                    threshold,
                    sequence_name,
                    result["frames_processed"],
                    result["avg_latency_ms"],
                    result["fps"],
                    result["TP"],
                    result["FP"],
                    result["FN"],
                    result["precision"],
                    result["recall"],
                    result["f1_score"],
                    result["prediction_json"],
                    result["annotation_json"]
                ])

                print(
                    f"Done | FPS={result['fps']:.2f}, "
                    f"P={result['precision']:.4f}, "
                    f"R={result['recall']:.4f}, "
                    f"F1={result['f1_score']:.4f}"
                )


if __name__ == "__main__":
    main()