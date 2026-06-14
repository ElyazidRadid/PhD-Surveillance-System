# experiments/paper2_adaptive/tau_distribution.py
"""Confirmatory check: is the per-frame F1-optimal threshold concentrated?

If the optimal tau is (nearly) the same low value on almost every frame, then
adaptive per-frame thresholding cannot beat a single global threshold — which
is what the spatial-gap analysis implied.

Run from repo root:
    python -m experiments.paper2_adaptive.tau_distribution
"""

from collections import Counter

from core.metrics.metrics import precision, recall, f1
from core.metrics.evaluator import load_json, index_by_frame
from .config import (
    THRESHOLD_GRID, IOU_THRESHOLD, TARGET_CLASS, SEQUENCES,
)
from .matching import count_at_threshold


def best_tau_per_frame(det_path, gt_path):
    det_frames = index_by_frame(load_json(det_path), "detections")
    gt_frames = index_by_frame(load_json(gt_path), "objects")
    frame_ids = sorted(set(det_frames) & set(gt_frames))

    counts = Counter()
    for fid in frame_ids:
        preds, gts = det_frames[fid], gt_frames[fid]
        best_tau, best_f1 = None, -1.0
        for tau in THRESHOLD_GRID:
            tp, fp, fn = count_at_threshold(preds, gts, tau, IOU_THRESHOLD, TARGET_CLASS)
            score = f1(precision(tp, fp), recall(tp, fn))
            if score > best_f1:
                best_f1, best_tau = score, tau
        counts[best_tau] += 1
    return counts, len(frame_ids)


def main():
    for seq, paths in SEQUENCES.items():
        counts, n = best_tau_per_frame(paths["detections"], paths["annotations"])
        print(f"\n=== {seq}: per-frame optimal tau over {n} frames ===")
        for tau in THRESHOLD_GRID:
            c = counts.get(tau, 0)
            if c:
                bar = "#" * int(60 * c / n)
                print(f"  tau={tau:<4}: {c:5d} ({100*c/n:5.1f}%) {bar}")
        top_tau, top_c = counts.most_common(1)[0]
        print(f"  -> dominant tau={top_tau} on {100*top_c/n:.1f}% of frames")


if __name__ == "__main__":
    main()
