# core/metrics/evaluator.py

import json
import os

from core.metrics.iou import compute_iou
from core.metrics.metrics import precision, recall, f1


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def index_by_frame(data, key_name):
    """
    Converts:
    [
      {"frame_id": 1, "objects": [...]},
      {"frame_id": 2, "objects": [...]}
    ]
    into:
    {
      1: [...],
      2: [...]
    }
    """
    indexed = {}
    for item in data:
        frame_id = int(item["frame_id"])
        indexed[frame_id] = item.get(key_name, [])
    return indexed


def match_detections_to_ground_truth(preds, gts, iou_threshold=0.5, target_class="person"):
    """
    preds: list of predicted detections for one frame
    gts: list of ground-truth objects for one frame

    Returns:
        tp, fp, fn
    """
    # Keep only target class
    preds = [p for p in preds if p.get("class_name") == target_class]
    gts = [g for g in gts if g.get("class_name") == target_class]

    matched_gt = set()
    tp = 0
    fp = 0

    for pred in preds:
        pred_box = pred["bbox"]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gts):
            if gt_idx in matched_gt:
                continue

            gt_box = gt["bbox"]
            iou = compute_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gts) - len(matched_gt)
    return tp, fp, fn


def evaluate_sequence(prediction_json, annotation_json, iou_threshold=0.5, target_class="person"):
    pred_data = load_json(prediction_json)
    gt_data = load_json(annotation_json)

    pred_frames = index_by_frame(pred_data, "detections")
    gt_frames = index_by_frame(gt_data, "objects")

    all_frame_ids = sorted(set(pred_frames.keys()) | set(gt_frames.keys()))

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for frame_id in all_frame_ids:
        preds = pred_frames.get(frame_id, [])
        gts = gt_frames.get(frame_id, [])

        tp, fp, fn = match_detections_to_ground_truth(
            preds, gts,
            iou_threshold=iou_threshold,
            target_class=target_class
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    p = precision(total_tp, total_fp)
    r = recall(total_tp, total_fn)
    f = f1(p, r)

    return {
        "prediction_file": prediction_json,
        "annotation_file": annotation_json,
        "target_class": target_class,
        "iou_threshold": iou_threshold,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": p,
        "recall": r,
        "f1_score": f
    }


def print_results(results):
    print("\n=== Evaluation Results ===")
    print(f"Prediction file : {results['prediction_file']}")
    print(f"Annotation file : {results['annotation_file']}")
    print(f"Target class    : {results['target_class']}")
    print(f"IoU threshold   : {results['iou_threshold']}")
    print(f"TP              : {results['TP']}")
    print(f"FP              : {results['FP']}")
    print(f"FN              : {results['FN']}")
    print(f"Precision       : {results['precision']:.4f}")
    print(f"Recall          : {results['recall']:.4f}")
    print(f"F1-score        : {results['f1_score']:.4f}")


if __name__ == "__main__":
    # Change these paths to your actual files
    prediction_json = "../benchmark/results/yolov8m_0.3_GPU_t/MOT20-05.json"
    annotation_json = "../benchmark/annotations/MOT20-05.json"

    results = evaluate_sequence(
        prediction_json=prediction_json,
        annotation_json=annotation_json,
        iou_threshold=0.5,
        target_class="person"
    )

    print_results(results)