# main.py

import os
import cv2

from config import (
    RUN_MODE,
    VIDEO_CONFIG,
    PREPROCESS_CONFIG,
    DETECTION_CONFIG,
    BENCHMARK_CONFIG
)
from video_source import VideoSource
from preprocessing import Preprocessor
from detector import Detector
from utils.logger import DetectionLogger
from utils.timer import Timer
from tracker import Tracker
from config import TRACKER_CONFIG


def draw_tracked_objects(frame, tracked_objects):
    output = frame.copy()

    for obj in tracked_objects:
        x1, y1, x2, y2 = obj["bbox"]
        track_id = obj["track_id"]
        confidence = obj["confidence"]
        class_name = obj["class_name"]

        label = f'ID {track_id} | {class_name}: {confidence:.2f}'

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


def convert_for_display(processed_frame, raw_frame):
    if processed_frame.dtype != raw_frame.dtype:
        return (processed_frame * 255).astype("uint8")
    return processed_frame.copy()


def run_webcam_mode():
    source = VideoSource(
        source=VIDEO_CONFIG["source"],
        target_fps=VIDEO_CONFIG["target_fps"],
        use_frame_skip=VIDEO_CONFIG["use_frame_skip"],
        frame_skip_interval=VIDEO_CONFIG["frame_skip_interval"]
    )

    tracker = Tracker(TRACKER_CONFIG)
    preprocessor = Preprocessor(PREPROCESS_CONFIG)
    detector = Detector(DETECTION_CONFIG)

    while source.is_opened():
        success, frame_data = source.read()
        if not success:
            break

        raw_frame = frame_data["frame"]

        processed_frame, prep_meta = preprocessor.apply(raw_frame)
        detections = detector.detect(processed_frame)
        tracked_objects = tracker.update(detections)

        display_frame = convert_for_display(processed_frame, raw_frame)
        output_frame = draw_tracked_objects(display_frame, tracked_objects)

        cv2.imshow("Detections", output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    source.release()
    cv2.destroyAllWindows()


def run_benchmark_mode():
    input_root = BENCHMARK_CONFIG["input_root"]
    output_folder = BENCHMARK_CONFIG["output_folder"]
    model_name = BENCHMARK_CONFIG["model_name"]
    input_type = BENCHMARK_CONFIG["input_type"]
    sequence_names = BENCHMARK_CONFIG["sequence_names"]

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, model_name), exist_ok=True)

    tracker = Tracker(TRACKER_CONFIG)
    preprocessor = Preprocessor(PREPROCESS_CONFIG)
    detector = Detector(DETECTION_CONFIG)

    for seq_name in sequence_names:
        print(f"\nProcessing sequence: {seq_name}")

        if input_type == "image_folder":
            source_path = os.path.join(input_root, seq_name, "img1")
        else:
            source_path = os.path.join(input_root, seq_name)

        result_path = os.path.join(
            output_folder,
            model_name,
            f"{seq_name}.json"
        )

        source = VideoSource(
            source=source_path,
            target_fps=VIDEO_CONFIG["target_fps"],
            use_frame_skip=VIDEO_CONFIG["use_frame_skip"],
            frame_skip_interval=VIDEO_CONFIG["frame_skip_interval"]
        )

        logger = DetectionLogger(result_path)
        timer = Timer()

        frame_count = 0

        while source.is_opened():
            success, frame_data = source.read()
            if not success:
                break

            raw_frame = frame_data["frame"]
            frame_id = frame_data["frame_id"]

            processed_frame, prep_meta = preprocessor.apply(raw_frame)

            timer.start()
            detections = detector.detect(processed_frame)
            tracked_objects = tracker.update(detections)
            timer.stop()

            logger.log(frame_id, detections)
            frame_count += 1

            if BENCHMARK_CONFIG.get("show_preview", False):
                display_frame = convert_for_display(processed_frame, raw_frame)
                output_frame = draw_tracked_objects(display_frame, tracked_objects)
                cv2.imshow(f"Benchmark Preview - {seq_name}", output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        logger.save()
        source.release()
        cv2.destroyAllWindows()

        avg_latency = timer.average()
        fps = 1 / avg_latency if avg_latency > 0 else 0

        print(f"Finished: {seq_name}")
        print(f"Frames processed: {frame_count}")
        print(f"Average latency: {avg_latency * 1000:.2f} ms/frame")
        print(f"Approximate FPS: {fps:.2f}")


def main():
    if RUN_MODE == "webcam":
        run_webcam_mode()
    elif RUN_MODE == "benchmark":
        run_benchmark_mode()
    else:
        raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")


if __name__ == "__main__":
    main()