# experiments/paper2_adaptive/spatial_gap.py
"""Spatial-gap analysis (PLAN.md section 6).

The key go/no-go experiment for Paper 2: before training any model, measure how
much headroom a *spatial* (per-cell) threshold has over a *per-frame* threshold.

For each sequence we compute sequence-level F1 under five strategies:

  fixed_default      : single tau = 0.30 for every frame
  fixed_best_global  : the single tau (swept) that maximises sequence F1
  per_frame_oracle   : best tau chosen independently per frame   (uses GT)
  per_cell_oracle    : best tau chosen independently per grid cell (uses GT)

The gap (per_cell_oracle - per_frame_oracle) upper-bounds the benefit the
learned spatial method can ever deliver. A large gap justifies the whole paper.

Run from the repo root:
    python -m experiments.paper2_adaptive.spatial_gap
"""

import os
import json

from core.metrics.metrics import precision, recall, f1
from core.metrics.evaluator import load_json, index_by_frame
from .config import (
    GRID_ROWS, GRID_COLS, THRESHOLD_GRID, IOU_THRESHOLD, TARGET_CLASS,
    FRAME_WIDTH, FRAME_HEIGHT, SEQUENCES, OUTPUT_DIR,
)
from .matching import partition_by_cell, count_at_threshold


def _f1_from_counts(tp, fp, fn):
    return f1(precision(tp, fp), recall(tp, fn))


def _best_tau_counts(preds, gts):
    """Best (tp, fp, fn) over the threshold grid for one (preds, gts) group."""
    best = None
    best_f1 = -1.0
    for tau in THRESHOLD_GRID:
        tp, fp, fn = count_at_threshold(preds, gts, tau, IOU_THRESHOLD, TARGET_CLASS)
        score = _f1_from_counts(tp, fp, fn)
        if score > best_f1:
            best_f1 = score
            best = (tp, fp, fn)
    return best


def analyse_sequence(det_path, gt_path):
    det_frames = index_by_frame(load_json(det_path), "detections")
    gt_frames = index_by_frame(load_json(gt_path), "objects")
    frame_ids = sorted(set(det_frames) & set(gt_frames))

    # Accumulators for each strategy: dict tau -> [tp, fp, fn] for global sweep,
    # and running totals for the oracle strategies.
    global_sweep = {tau: [0, 0, 0] for tau in THRESHOLD_GRID}
    default_totals = [0, 0, 0]
    perframe_totals = [0, 0, 0]
    percell_totals = [0, 0, 0]

    for fid in frame_ids:
        preds = det_frames[fid]
        gts = gt_frames[fid]

        # fixed_default (tau = 0.30)
        tp, fp, fn = count_at_threshold(preds, gts, 0.30, IOU_THRESHOLD, TARGET_CLASS)
        default_totals[0] += tp; default_totals[1] += fp; default_totals[2] += fn

        # global sweep (for fixed_best_global, chosen after the loop)
        for tau in THRESHOLD_GRID:
            tp, fp, fn = count_at_threshold(preds, gts, tau, IOU_THRESHOLD, TARGET_CLASS)
            global_sweep[tau][0] += tp
            global_sweep[tau][1] += fp
            global_sweep[tau][2] += fn

        # per_frame_oracle
        tp, fp, fn = _best_tau_counts(preds, gts)
        perframe_totals[0] += tp; perframe_totals[1] += fp; perframe_totals[2] += fn

        # per_cell_oracle
        pred_cells = partition_by_cell(preds, GRID_ROWS, GRID_COLS, FRAME_WIDTH, FRAME_HEIGHT)
        gt_cells = partition_by_cell(gts, GRID_ROWS, GRID_COLS, FRAME_WIDTH, FRAME_HEIGHT)
        all_keys = set(pred_cells) | set(gt_cells)
        for key in all_keys:
            cp = pred_cells.get(key, [])
            cg = gt_cells.get(key, [])
            tp, fp, fn = _best_tau_counts(cp, cg)
            percell_totals[0] += tp; percell_totals[1] += fp; percell_totals[2] += fn

    best_global_f1 = max(_f1_from_counts(*global_sweep[t]) for t in THRESHOLD_GRID)

    return {
        "fixed_default":     _f1_from_counts(*default_totals),
        "fixed_best_global": best_global_f1,
        "per_frame_oracle":  _f1_from_counts(*perframe_totals),
        "per_cell_oracle":   _f1_from_counts(*percell_totals),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    for seq, paths in SEQUENCES.items():
        if not (os.path.exists(paths["detections"]) and os.path.exists(paths["annotations"])):
            print(f"[skip] {seq}: missing data")
            continue
        r = analyse_sequence(paths["detections"], paths["annotations"])
        r["spatial_gap"] = r["per_cell_oracle"] - r["per_frame_oracle"]
        results[seq] = r
        print(f"\n=== {seq} (grid {GRID_ROWS}x{GRID_COLS}) ===")
        for k in ["fixed_default", "fixed_best_global", "per_frame_oracle",
                  "per_cell_oracle", "spatial_gap"]:
            print(f"  {k:18s}: {r[k]:.4f}")

    out = os.path.join(OUTPUT_DIR, "spatial_gap.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
