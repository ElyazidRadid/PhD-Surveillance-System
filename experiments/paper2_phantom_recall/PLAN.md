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
- [x] `recover.py` — recovery method (greedy-IoU linker + ByteTrack variant)
- [x] `evaluate.py` — recall/precision vs. baselines
- [x] `min_track_sweep.py` — diagnostic / tuning sweep
- [ ] `analysis.R` — figures (deferred)

## 7. Status — SHELVED (with findings recorded)

**Go/no-go: PASSED.** Recoverable-recall ceiling is real: 6–31 % of false
negatives reappear in adjacent frames (+0.04 strict to +0.15 loose recall,
beyond the threshold ceiling).

**Method result (best so far): modest.** A simple greedy-IoU linker with linear
gap interpolation gives:
- MOT20-05 (dense): **+0.017 F1** (recall +0.022, precision −0.014) at gap≤3.
- MOT20-02 (moderate): essentially flat (+0.0007).

**Two attempted improvements both FAILED (important, counterintuitive):**
1. **ByteTrack association** recovers 6.7× *fewer* boxes than the naive linker
   (3.5k vs 24k on MOT20-05). ByteTrack is conservative by design (avoids ID
   switches) → fragments tracks in dense crowds → exposes few interpolatable
   gaps. For recall recovery we *want* liberal gap-bridging.
2. **`min_track_length` filtering** had no effect (mtl=1 and mtl=5 recover the
   same count) — the filter was never the bottleneck; ByteTrack's fragmentation
   was.

**Conclusion / why shelved:** the simple linker is the best method but the gain
is modest, and the two sophistication attempts did not help. Captures only part
of the recoverable ceiling. Shelved pending a fresh idea to close the gap
(motion-aware interpolation, larger gaps with confidence-gating) or a pivot.

**If resumed, start here:** push the greedy linker (larger `max_gap` with
confidence-gated revivals; tune `ASSOCIATION_IOU`; motion-aware interpolation)
toward the +0.04 strict ceiling — not ByteTrack.
