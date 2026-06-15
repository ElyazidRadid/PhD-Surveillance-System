# experiments/paper2_phantom_recall/min_track_sweep.py
"""Diagnose and tune the ByteTrack recovery: sweep min_track_length.

The aggressive min_track_length=5 filter deleted almost all recovery. This sweep
builds the ByteTrack timeline once per sequence, then varies min_track_length to
find the sweet spot, reporting how many boxes are recovered (diagnostic) and the
resulting precision/recall/F1. Baseline and the v1 greedy linker are shown for
reference.

Run from repo root:
    python -m experiments.paper2_phantom_recall.min_track_sweep
"""

import os
import json

from core.metrics.evaluator import load_json, index_by_frame
from .config import (
    OPERATING_THRESHOLD, TRACKER_CONFIG, ASSOCIATION_IOU, RECOVERED_CONFIDENCE,
    TARGET_CLASS, SEQUENCES, OUTPUT_DIR,
)
from .recover import (
    assemble, build_tracklets, recovered_boxes_by_frame,
    build_track_timelines, recovered_boxes_filtered,
)
from .evaluate import _aggregate

MAX_GAP = 3
MIN_TRACK_LENGTHS = [1, 2, 3, 5]


def _n_recovered(recovered):
    return sum(len(v) for v in recovered.values())


def run_sequence(det_path, gt_path):
    gt_frames = index_by_frame(load_json(gt_path), "objects")
    det_frames = index_by_frame(load_json(det_path), "detections")
    fids = sorted(det_frames)

    def evalu(recovered):
        aug = assemble(det_frames, fids, recovered, OPERATING_THRESHOLD,
                       TARGET_CLASS, RECOVERED_CONFIDENCE)
        return _aggregate(aug, gt_frames)

    rows = []
    bp, br, bf = evalu({})
    rows.append(("baseline", 0, bp, br, bf))

    # v1 greedy reference at max_gap.
    tracklets = build_tracklets(det_frames, fids, ASSOCIATION_IOU, MAX_GAP,
                                OPERATING_THRESHOLD, TARGET_CLASS)
    rec_v1 = recovered_boxes_by_frame(tracklets, MAX_GAP)
    p, r, f = evalu(rec_v1)
    rows.append((f"v1 greedy(gap={MAX_GAP})", _n_recovered(rec_v1), p, r, f))

    # v2 ByteTrack: build timeline once, sweep min_track_length.
    _, _, timelines = build_track_timelines(
        det_path, TRACKER_CONFIG, OPERATING_THRESHOLD, TARGET_CLASS)
    n_tracks = len(timelines)
    for mtl in MIN_TRACK_LENGTHS:
        rec = recovered_boxes_filtered(timelines, MAX_GAP, mtl)
        p, r, f = evalu(rec)
        rows.append((f"v2 byte(mtl={mtl})", _n_recovered(rec), p, r, f))

    return rows, n_tracks, bf


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}
    for seq, paths in SEQUENCES.items():
        if not (os.path.exists(paths["detections"]) and os.path.exists(paths["annotations"])):
            print(f"[skip] {seq}: missing data")
            continue
        rows, n_tracks, base_f1 = run_sequence(paths["detections"], paths["annotations"])
        results[seq] = [
            {"setting": s, "recovered_boxes": n, "precision": p, "recall": r,
             "f1": f, "d_f1": f - base_f1}
            for (s, n, p, r, f) in rows
        ]
        print(f"\n=== {seq} (gap<={MAX_GAP}, ByteTrack tracks={n_tracks}) ===")
        print(f"  {'setting':>20} {'recov.boxes':>12} {'prec':>7} {'recall':>7} {'f1':>7} {'dF1':>8}")
        for (s, n, p, r, f) in rows:
            print(f"  {s:>20} {n:>12} {p:>7.4f} {r:>7.4f} {f:>7.4f} {f-base_f1:>+8.4f}")

    out = os.path.join(OUTPUT_DIR, "min_track_sweep.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
