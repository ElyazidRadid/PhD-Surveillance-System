# Paper 2 (pivoted) — Phantom Recall: Temporal Recovery of Missed Detections

**Working title:** Beyond the Threshold Ceiling: Temporal Recovery of Missed
Detections in Dense Surveillance

**Target journal:** Signal, Image and Video Processing (Q2/Q3 CV venue)

---

## 1. Motivation — and why this replaces the adaptive-threshold idea

We first asked whether *adaptive* confidence thresholding (per-frame or spatial)
could beat a fixed threshold in dense scenes. A go/no-go analysis on MOT20 showed
it cannot:

- The per-frame F1-optimal threshold is **invariant**: τ = 0.05 on **100 %** of
  MOT20-05 frames and 61 % of MOT20-02 frames (rest at 0.10, negligible gain).
- Per-cell (spatial) oracle thresholds give **no** improvement over per-frame
  (+0.001 / −0.010 F1) — splitting dense frames even hurts.

**Conclusion:** thresholds are already maxed out. At τ = 0.05 the detector fires
on everything it can; the remaining false negatives are objects it *never* sees
at any confidence (heavy occlusion). A threshold cannot recover them — but
**temporal context can**. This negative result is the motivation for the paper.

## 2. Core idea

In a video, an object missed in frame *t* is often clearly detected in *t−1* and
*t+1*. Such misses are **temporally recoverable**: a tracker / temporal model can
carry the object through the gap. We:

1. **Quantify recoverable recall** — the fraction of false negatives that are
   recoverable from temporal neighbours (the headroom beyond the threshold
   ceiling). *(This is the go/no-go.)*
2. **Propose a recovery method** — detector + tracker fusion that re-instates
   missed detections validated by track consistency, with controlled precision
   cost.
3. **Evaluate** the recall/precision trade-off vs. detection-only baselines.

## 3. Why it's novel and not threshold-capped

Unlike thresholding (which we proved is saturated), temporal recovery operates on
a different axis — it adds detections that *no* confidence threshold could surface
because the detector produced nothing for them in that frame. It directly attacks
the recall collapse Paper 1 documented.

## 4. Research questions

- **RQ1** What fraction of missed detections in dense scenes are temporally
  recoverable (present in adjacent frames)? *(go/no-go)*
- **RQ2** Can a detector+tracker fusion realise that recovery, and at what
  precision cost?
- **RQ3** How does recoverable recall scale with crowd density and temporal
  window size?
- **RQ4** Does it generalise across detectors and sequences?

## 5. Go/no-go metric (this folder, `recoverable_recall.py`)

At the optimal operating threshold (τ = 0.05):

```
FN(t)            = GT boxes in frame t not matched by any detection
recoverable FN   = FN boxes that ARE matched by a detection in BOTH t-1 and t+1
recoverable_recall_ceiling = (TP + recoverable_FN) / (TP + FN)
```

If a meaningful fraction of FN is recoverable, the paper is viable. If not, we
pivot again (to the Density-Cliff study).

## 6. Build stages

- [x] `config.py` — operating threshold, IoU thresholds, temporal window
- [x] `recoverable_recall.py` — go/no-go ceiling analysis
- [ ] `recover.py` — actual fusion method (interpolate / track-validated revival)
- [ ] `evaluate.py` — recall/precision vs. baselines, across models & sequences
- [ ] `analysis.R` — figures (reuse Paper-1 journal style)
