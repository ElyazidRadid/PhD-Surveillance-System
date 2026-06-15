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
    TRACKER_CONFIG, MIN_TRACK_LENGTH, TARGET_CLASS, SEQUENCES, OUTPUT_DIR,
)
from .recover import (
    assemble, build_tracklets, recovered_boxes_by_frame,
    build_track_timelines, recovered_boxes_filtered,
)


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
    det_frames = index_by_frame(load_json(det_path), "detections")
    fids = sorted(det_frames)

    def aug_for(recovered):
        return assemble(det_frames, fids, recovered, OPERATING_THRESHOLD,
                        TARGET_CLASS, RECOVERED_CONFIDENCE)

    # Baseline: detector only (no recovered boxes).
    bp, br, bf = _aggregate(aug_for({}), gt_frames)
    rows = [{"setting": "baseline", "method": "none", "max_gap": 0,
             "precision": bp, "recall": br, "f1": bf}]

    def add_row(name, method, g, recovered):
        p, r, f = _aggregate(aug_for(recovered), gt_frames)
        rows.append({"setting": name, "method": method, "max_gap": g,
                     "precision": p, "recall": r, "f1": f,
                     "d_recall": r - br, "d_precision": p - bp, "d_f1": f - bf})

    # v1: build greedy tracklets ONCE, apply each max_gap.
    tracklets = build_tracklets(det_frames, fids, ASSOCIATION_IOU,
                                max(max_gaps), OPERATING_THRESHOLD, TARGET_CLASS)
    for g in max_gaps:
        add_row(f"v1 greedy(gap={g})", "greedy", g,
                recovered_boxes_by_frame(tracklets, g))

    # v2: build ByteTrack timelines ONCE, apply each max_gap.
    _, _, timelines = build_track_timelines(
        det_path, TRACKER_CONFIG, OPERATING_THRESHOLD, TARGET_CLASS)
    for g in max_gaps:
        add_row(f"v2 bytetrack(gap={g})", "bytetrack", g,
                recovered_boxes_filtered(timelines, g, MIN_TRACK_LENGTH))

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
        print(f"\n=== {seq} (tau={OPERATING_THRESHOLD}, min_track_len={MIN_TRACK_LENGTH}) ===")
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
