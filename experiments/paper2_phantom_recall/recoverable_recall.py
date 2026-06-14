# experiments/paper2_phantom_recall/recoverable_recall.py
"""Phantom Recall go/no-go: how many missed detections are temporally recoverable?

At the optimal operating threshold, recall is already saturated w.r.t. the
detector. This measures the headroom *beyond* the threshold ceiling: the fraction
of false negatives that reappear as detections in adjacent frames and could thus
be recovered by a temporal / tracking mechanism.

Run from repo root:
    python -m experiments.paper2_phantom_recall.recoverable_recall
"""

import os
import json

from core.metrics.iou import compute_iou
from core.metrics.metrics import recall as recall_fn
from core.metrics.evaluator import load_json, index_by_frame
from .config import (
    OPERATING_THRESHOLD, IOU_THRESHOLD, RECOVERY_IOU, TEMPORAL_WINDOW,
    TARGET_CLASS, SEQUENCES, OUTPUT_DIR,
)


def _filter(dets, tau):
    return [d for d in dets
            if d.get("class_name") == TARGET_CLASS and d.get("confidence", 1.0) >= tau]


def _unmatched_gts(preds, gts):
    """Greedy IoU match; return (tp, list_of_unmatched_gt_boxes)."""
    gts = [g for g in gts if g.get("class_name") == TARGET_CLASS]
    matched = set()
    tp = 0
    for p in preds:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            iou = compute_iou(p["bbox"], g["bbox"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= IOU_THRESHOLD and best_j != -1:
            tp += 1
            matched.add(best_j)
    unmatched = [g["bbox"] for j, g in enumerate(gts) if j not in matched]
    return tp, unmatched


def _covered_in_frame(box, dets, iou_thr):
    """True if any detection in `dets` covers `box` (IoU >= iou_thr)."""
    for d in dets:
        if compute_iou(box, d["bbox"]) >= iou_thr:
            return True
    return False


def analyse_sequence(det_path, gt_path, k, require_all):
    """Compute recoverable-recall ceiling for window +/-k.

    require_all=True  : FN box must be covered in EVERY neighbour frame (strict).
    require_all=False : covered in AT LEAST ONE neighbour frame (loose).
    """
    det_frames = index_by_frame(load_json(det_path), "detections")
    gt_frames = index_by_frame(load_json(gt_path), "objects")
    fids = sorted(set(det_frames) & set(gt_frames))

    # Pre-filter detections at the operating threshold per frame (cache once).
    dets_at = analyse_sequence._cache.get(det_path)
    if dets_at is None:
        dets_at = {fid: _filter(det_frames[fid], OPERATING_THRESHOLD) for fid in fids}
        analyse_sequence._cache[det_path] = dets_at

    # Cache per-frame (tp, unmatched FN boxes) too — independent of k.
    fn_cache = analyse_sequence._fncache.get(det_path)
    if fn_cache is None:
        fn_cache = {fid: _unmatched_gts(dets_at[fid], gt_frames[fid]) for fid in fids}
        analyse_sequence._fncache[det_path] = fn_cache

    total_tp = sum(tp for tp, _ in fn_cache.values())
    total_fn = sum(len(u) for _, u in fn_cache.values())
    recoverable = 0

    for fid in fids:
        _, unmatched = fn_cache[fid]
        if not unmatched:
            continue
        neighbours = [fid + d for d in range(-k, k + 1) if d != 0]
        present = [nf for nf in neighbours if nf in dets_at]
        if require_all and len(present) != len(neighbours):
            continue
        if not present:
            continue
        for box in unmatched:
            hits = (_covered_in_frame(box, dets_at[nf], RECOVERY_IOU) for nf in present)
            if (all(hits) if require_all else any(hits)):
                recoverable += 1

    base = recall_fn(total_tp, total_fn)
    ceil = recall_fn(total_tp + recoverable, total_fn - recoverable)
    return {
        "window": k,
        "criterion": "all-neighbours" if require_all else "any-neighbour",
        "tp": total_tp, "fn": total_fn, "recoverable_fn": recoverable,
        "fraction_of_fn_recoverable": recoverable / (total_fn + 1e-6),
        "baseline_recall": base, "ceiling_recall": ceil, "recall_gain": ceil - base,
    }


analyse_sequence._cache = {}
analyse_sequence._fncache = {}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    settings = [(1, True), (2, True), (3, True), (1, False), (2, False), (3, False)]
    results = {}
    for seq, paths in SEQUENCES.items():
        if not (os.path.exists(paths["detections"]) and os.path.exists(paths["annotations"])):
            print(f"[skip] {seq}: missing data")
            continue
        print(f"\n=== {seq} (tau={OPERATING_THRESHOLD}) ===")
        print(f"  {'window':>7} {'criterion':>14} {'recov.FN%':>10} {'recall':>16} {'gain':>8}")
        results[seq] = []
        for k, req in settings:
            r = analyse_sequence(paths["detections"], paths["annotations"], k, req)
            results[seq].append(r)
            print(f"  {('+/-'+str(k)):>7} {r['criterion']:>14} "
                  f"{100*r['fraction_of_fn_recoverable']:>9.1f}% "
                  f"{r['baseline_recall']:.3f}->{r['ceiling_recall']:.3f}  "
                  f"+{r['recall_gain']:>6.4f}")

    out = os.path.join(OUTPUT_DIR, "recoverable_recall_sweep.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
