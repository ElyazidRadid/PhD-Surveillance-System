# experiments/paper2_adaptive/matching.py
"""Spatial-cell assignment and per-cell detection/GT matching.

Reuses the greedy IoU matching from core.metrics so results stay consistent
with Paper 1's evaluation.
"""

from core.metrics.evaluator import match_detections_to_ground_truth


def cell_of(box, grid_rows, grid_cols, frame_w, frame_h):
    """Return the (row, col) grid cell index of a box, keyed by its centroid."""
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    col = min(int(cx / frame_w * grid_cols), grid_cols - 1)
    row = min(int(cy / frame_h * grid_rows), grid_rows - 1)
    return row, col


def partition_by_cell(items, grid_rows, grid_cols, frame_w, frame_h):
    """Group a list of detection/GT dicts (each with a 'bbox') into grid cells.

    Returns a dict {(row, col): [items...]}.
    """
    cells = {}
    for it in items:
        key = cell_of(it["bbox"], grid_rows, grid_cols, frame_w, frame_h)
        cells.setdefault(key, []).append(it)
    return cells


def count_at_threshold(preds, gts, tau, iou_threshold, target_class):
    """Filter preds by confidence >= tau, then match to gts. Returns (tp, fp, fn)."""
    kept = [p for p in preds if p.get("confidence", 1.0) >= tau]
    return match_detections_to_ground_truth(
        kept, gts, iou_threshold=iou_threshold, target_class=target_class
    )
