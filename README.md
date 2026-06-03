# PhD Surveillance System

Research codebase for studying object detection under confidence-threshold
variation in dense surveillance scenes.

## Repository layout

```
phd-surveillance/
├── core/                       # Shared, stable building blocks (imported by all papers)
│   ├── detection/              # YOLO detector wrappers
│   ├── data/                   # Video / image-sequence loading
│   ├── preprocessing/          # Frame preprocessing
│   ├── tracking/               # Multi-object tracker
│   ├── metrics/                # IoU, precision/recall/F1, sequence evaluation
│   └── utils/                  # Logging, timing
├── experiments/                # One self-contained package per paper
│   ├── paper1_thresholds/      # Paper 1: threshold sensitivity & evaluation bias
│   ├── paper2_adaptive/        # Paper 2: (planned)
│   └── paper3_density_cliff/   # Paper 3: (planned)
├── tools/                      # Standalone utilities (GPU check, MOT conversion)
├── datasets/                   # Input data (git-ignored)
├── models/                     # Model weights (git-ignored)
└── benchmark/                  # Results & logs (git-ignored)
```

## Design principle

`core/` holds everything shared across papers. Each paper lives in its own
package under `experiments/` and **imports** from `core` rather than copying it.
Fix a bug in `core` once and every paper benefits.

## Running

Always run from the repository root using module syntax so the `core` package
resolves and relative config imports work:

```bash
# Paper 1 — live/benchmark detection pipeline
python -m experiments.paper1_thresholds.main

# Paper 1 — threshold sweep experiments
python -m experiments.paper1_thresholds.run_experiments

# Tools
python -m tools.check_gpu
```

## Reproducibility

Each submitted paper is pinned by a git tag (e.g. `paper1-sivp-submission`) and,
where applicable, a Zenodo deposit for its data/results. To recover the exact
code behind a paper:

```bash
git checkout paper1-sivp-submission
```
