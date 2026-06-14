# Paper 2 — Spatially-Adaptive, Density-Conditioned Confidence Thresholding

**Working title:** Learning Where to Trust: Density-Conditioned Spatial
Confidence Thresholding for Object Detection in Dense Scenes

**Target journal:** Signal, Image and Video Processing (or similar Q2/Q3 CV venue)

---

## 1. Motivation (from Paper 1)

Paper 1 showed that a single fixed confidence threshold (a) collapses recall in
dense regions, (b) destabilises model rankings, and (c) is consistently
sub-optimal — the best threshold varies by scene *and* by location within a
scene. Paper 1 *diagnosed* this. Paper 2 *fixes* it.

## 2. Core idea (combines per-frame learning + spatial density-awareness)

Instead of one threshold per image, predict a **spatially-varying threshold
map** for each frame. The threshold in each region is conditioned on **local
crowd density** and local detection statistics, on top of a learned
**frame-level base threshold**. No ground truth is needed at deployment.

Two-level formulation:

```
tau(cell) = base(global_frame_features)  +  delta(local_cell_features)
```

- **base(.)** — a frame-level learned predictor (idea #1): captures the overall
  difficulty / optimal operating point of the whole frame.
- **delta(.)** — a local correction (idea #2): lowers the threshold in dense,
  occluded cells and raises it in sparse cells.

Both are learned jointly as a single regressor that takes global + local
features and outputs a per-cell threshold.

## 3. Why it is novel

- Prior work (e.g. Wenkel et al., 2021) selects **one global optimal threshold
  per model**. We predict a **per-region** threshold that **adapts within a
  single frame**, conditioned on density.
- It is **label-free at inference** — uses only signals computable from the
  detector's own raw outputs.
- It is **model-agnostic** — trained and tested across the six YOLO variants
  from Paper 1.

## 4. Research questions

- **RQ1** Can per-cell optimal thresholds be predicted from label-free local +
  global features?
- **RQ2** How much of the fixed-vs-oracle F1 gap does the spatial method recover,
  and how much of that gain is specifically attributable to the *spatial*
  component over a per-frame-only predictor? (key ablation)
- **RQ3** Does it generalise across models and across sequences/datasets
  (MOT20-02 ↔ MOT20-05, and ideally a second dataset)?
- **RQ4** What is the inference overhead vs. the accuracy gain?

## 5. Method pipeline

1. **Candidate detections.** Reuse the existing low-threshold (τ=0.05) detection
   logs as the full candidate set; optionally re-run at τ=0.01 for headroom.
2. **Spatial grid.** Partition each frame into an `GRID_ROWS × GRID_COLS` grid.
   Assign each detection / GT box to a cell by its centroid.
3. **Features per cell** (label-free): local detection count (density proxy),
   local confidence distribution (mean, std, skew, entropy, fraction above
   cutoffs), mean/var box area, cell position. Plus **global frame features**
   (total count, global confidence-histogram shape).
4. **Oracle target.** For each cell, sweep thresholds and record the F1-optimal
   τ computed against ground truth (offline only — training signal).
5. **Model.** Gradient-boosted trees (interpretable, fast) mapping features →
   per-cell τ. Light MLP as an alternative.
6. **Apply.** At inference, predict per-cell τ from label-free features, threshold
   detections per cell, evaluate.

## 6. Baselines & upper bounds

| Name | Description | Role |
|------|-------------|------|
| Fixed-default | τ = 0.3 (or model default) | weak baseline |
| Fixed-best-global | single best τ per model (Paper 1) | strong baseline |
| Per-frame-learned | predict one τ per frame (idea #1 only) | ablation |
| **Spatial-adaptive (ours)** | per-cell density-conditioned τ | proposed |
| Per-frame-oracle | best single τ per frame (uses GT) | upper bound A |
| Per-cell-oracle | best τ per cell (uses GT) | upper bound B |

The gap between *Per-frame-oracle* and *Per-cell-oracle* quantifies the maximum
benefit the spatial component can ever provide — a clean way to justify the
spatial idea before even training.

## 7. Evaluation protocol

- **Cross-sequence:** train on MOT20-02, test on MOT20-05 and vice versa.
- **Cross-model:** leave-one-model-out; report per-model and averaged.
- **Metrics:** F1 (primary), recall, precision; gap-recovery % vs. oracles;
  inference overhead (ms/frame).
- **Generalisation (stretch):** evaluate on a held-out dataset (MOT17 / a
  CrowdHuman subset).

## 8. Expected contributions

1. A label-free, density-conditioned, **spatially-adaptive** thresholding method.
2. Evidence that *where* you threshold matters within a frame, not just *what*
   global threshold you pick (the spatial-gap analysis).
3. A model-agnostic recipe that recovers a large fraction of the oracle gain at
   negligible cost, validated across six detectors and two sequences.

## 9. Risks & mitigations

- **Spatial may not beat per-frame.** → The oracle-gap analysis (§6) tells us
  up front whether spatial can help; if the gap is small we pivot to the
  per-frame story (still publishable).
- **Grid artifacts.** → Try overlapping cells / per-detection local density as a
  continuous alternative.
- **Overfitting to MOT20.** → Cross-sequence + cross-dataset validation.

## 10. Build stages (code in this folder)

- [ ] `config.py` — grid size, threshold grid, radius, paths
- [ ] `matching.py` — IoU greedy matching (reuse `core.metrics.iou`)
- [ ] `features.py` — per-cell + global feature extraction
- [ ] `oracle.py` — per-cell / per-frame oracle thresholds from GT
- [ ] `dataset.py` — assemble (features → oracle τ) training table
- [ ] `model.py` — train/predict the threshold regressor
- [ ] `evaluate.py` — apply thresholds, compute all baselines, write results
- [ ] `analysis.R` — figures (reuse Paper-1 journal style)
