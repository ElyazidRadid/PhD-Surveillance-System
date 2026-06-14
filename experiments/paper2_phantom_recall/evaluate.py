# experiments/paper2_phantom_recall/evaluate.py
"""Evaluate the Phantom Recall method: recall AND precision vs detection-only.

For each sequence we compute precision / recall / F1 for:
  baseline  : detector outputs at the operating threshold (no recovery)
  recovered : baseline + track-validated interpolated boxes, swept over max_gap

The key question is whether recovery raises recall meaningfully without paying
too much precision (interpolated boxes that don't correspond to a real object).

Run from repo root:
    python -m experiments.paper2_phantom_recall.evaluate
"""

import os
import json

from core.metrics.iou import compute_iou
from core.metrics.metrics import precision as prec_fn, recall as rec_fn, f1 as f1_fn
from core.metrics.evaluator import load_json, index_by_frame
from .config import (
    OPERATING_THRESHOLD, IOU_THRESHOLD, ASSOCIATION_IOU, RECOVERED_CONFIDENCE,
    TARGET_CLASS, SEQUENCES, OUTPUT_DIR,
)
from .recover import augmented_detections


def _counts(preds, gts):
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
        else:
            tp += 0
    fp = len(preds) - tp
    fn = len(gts) - len(matched)
    return tp, fp, fn


def _prf(tp, fp, fn):
    p, r = prec_fn(tp, fp), rec_fn(tp, fn)
    return p, r, f1_fn(p, r)


def _aggregate(pred_by_frame, gt_frames):
    tp = fp = fn = 0
    for fid in gt_frames:
        t, f, n = _counts(pred_by_frame.get(fid, []), gt_frames[fid])
        tp += t; fp += f; fn += n
    return _prf(tp, fp, fn)


def evaluate_sequence(det_path, gt_path, max_gaps):
    gt_frames = index_by_frame(load_json(gt_path), "objects")

    # Baseline: detector only at operating threshold.
    base_aug, _ = augmented_detections(
        det_path, ASSOCIATION_IOU, max_gap=0, tau=OPERATING_THRESHOLD,
        target_class=TARGET_CLASS, recovered_conf=RECOVERED_CONFIDENCE,
    )
    bp, br, bf = _aggregate(base_aug, gt_frames)
    rows = [{"setting": "baseline", "max_gap": 0,
             "precision": bp, "recall": br, "f1": bf}]

    for g in max_gaps:
        aug, _ = augmented_detections(
            det_path, ASSOCIATION_IOU, max_gap=g, tau=OPERATING_THRESHOLD,
            target_class=TARGET_CLASS, recovered_conf=RECOVERED_CONFIDENCE,
        )
        p, r, f = _aggregate(aug, gt_frames)
        rows.append({"setting": f"recovered(max_gap={g})", "max_gap": g,
                     "precision": p, "recall": r, "f1": f,
                     "d_recall": r - br, "d_precision": p - bp, "d_f1": f - bf})
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    max_gaps = [1, 2, 3]
    results = {}
    for seq, paths in SEQUENCES.items():
        if not (os.path.exists(paths["detections"]) and os.path.exists(paths["annotations"])):
            print(f"[skip] {seq}: missing data")
            continue
        rows = evaluate_sequence(paths["detections"], paths["annotations"], max_gaps)
        results[seq] = rows
        print(f"\n=== {seq} (tau={OPERATING_THRESHOLD}, assoc_iou={ASSOCIATION_IOU}) ===")
        print(f"  {'setting':>22} {'precision':>10} {'recall':>8} {'f1':>8}")
        for r in rows:
            extra = ""
            if "d_f1" in r:
                extra = f"  (dR {r['d_recall']:+.4f}, dP {r['d_precision']:+.4f}, dF1 {r['d_f1']:+.4f})"
            print(f"  {r['setting']:>22} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>8.4f}{extra}")

    out = os.path.join(OUTPUT_DIR, "recovery_evaluation.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
