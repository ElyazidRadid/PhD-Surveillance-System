# experiments/paper2_phantom_recall/recover.py
"""Track-validated detection recovery (the Phantom Recall method).

Pipeline:
  1. Link per-frame detections into tracklets by greedy IoU association.
  2. Within each tracklet, find temporal gaps (frames the detector missed
     between two confirmed detections of the same object), up to `max_gap`.
  3. Linearly interpolate the box across each gap and emit it as a *recovered*
     detection.

The recovered boxes fill confirmed gaps, so they reinstate objects that the
detector momentarily missed — precisely the false negatives no confidence
threshold can surface.
"""

from core.metrics.iou import compute_iou
from core.metrics.evaluator import load_json, index_by_frame


def _filter(dets, tau, target_class):
    return [d for d in dets
            if d.get("class_name") == target_class and d.get("confidence", 1.0) >= tau]


def _lerp_box(b0, b1, t):
    """Linear interpolation between two boxes; t in (0,1)."""
    return [b0[i] + (b1[i] - b0[i]) * t for i in range(4)]


def build_tracklets(det_frames, fids, assoc_iou, max_gap, tau, target_class):
    """Greedy IoU tracker. Returns a list of tracklets {frame -> bbox}."""
    tracklets = []          # finished + active, each: {"boxes": {fid: bbox}, "last": fid}
    active = []             # indices into tracklets that can still be extended

    for fid in fids:
        dets = _filter(det_frames.get(fid, []), tau, target_class)

        # Build all candidate (iou, det_idx, active_idx) and assign greedily.
        cands = []
        for ai, ti in enumerate(active):
            last_box = tracklets[ti]["boxes"][tracklets[ti]["last"]]
            for di, d in enumerate(dets):
                iou = compute_iou(d["bbox"], last_box)
                if iou >= assoc_iou:
                    cands.append((iou, di, ai))
        cands.sort(reverse=True)

        used_det, used_active = set(), set()
        for iou, di, ai in cands:
            if di in used_det or ai in used_active:
                continue
            ti = active[ai]
            tracklets[ti]["boxes"][fid] = dets[di]["bbox"]
            tracklets[ti]["last"] = fid
            used_det.add(di); used_active.add(ai)

        # Unmatched detections start new tracklets.
        for di, d in enumerate(dets):
            if di not in used_det:
                tracklets.append({"boxes": {fid: d["bbox"]}, "last": fid})
                active.append(len(tracklets) - 1)

        # Retire tracklets idle longer than max_gap.
        active = [ti for ti in active if fid - tracklets[ti]["last"] <= max_gap]

    return tracklets


def recovered_boxes_by_frame(tracklets, max_gap):
    """For each tracklet, interpolate gaps <= max_gap. Returns {fid: [bbox,...]}."""
    recovered = {}
    for tr in tracklets:
        frames = sorted(tr["boxes"])
        for a, b in zip(frames, frames[1:]):
            gap = b - a
            if 2 <= gap <= max_gap + 1:   # at least one missing frame in between
                box_a, box_b = tr["boxes"][a], tr["boxes"][b]
                for f in range(a + 1, b):
                    t = (f - a) / gap
                    recovered.setdefault(f, []).append(_lerp_box(box_a, box_b, t))
    return recovered


def augmented_detections(det_path, assoc_iou, max_gap, tau, target_class, recovered_conf):
    """Return {fid: [detection dicts]} = original (>=tau) + recovered boxes."""
    det_frames = index_by_frame(load_json(det_path), "detections")
    fids = sorted(det_frames)

    tracklets = build_tracklets(det_frames, fids, assoc_iou, max_gap, tau, target_class)
    recovered = recovered_boxes_by_frame(tracklets, max_gap)

    out = {}
    for fid in fids:
        dets = _filter(det_frames[fid], tau, target_class)
        merged = [dict(d) for d in dets]
        for box in recovered.get(fid, []):
            merged.append({
                "class_name": target_class,
                "confidence": recovered_conf,
                "bbox": [int(round(v)) for v in box],
                "recovered": True,
            })
        out[fid] = merged
    return out, recovered
