# Paper 1 — Confidence Threshold Sensitivity & Evaluation Bias

**Title:** Rethinking Confidence Thresholds in Object Detection: Performance
Instability and Evaluation Bias in Dense Surveillance Environments

**Target journal:** Signal, Image and Video Processing (Springer)

## Contents

| File | Purpose |
|------|---------|
| `config.py` | Experiment configuration (models, thresholds, dataset paths) |
| `main.py` | Detection + tracking pipeline (webcam or benchmark mode) |
| `run_experiments.py` | Threshold sweep across models, writes `experiment_results.csv` |

## Run

From the repository root:

```bash
python -m experiments.paper1_thresholds.run_experiments
```

## Data & results

- Input: MOT20 (`datasets/MOT20/`), publicly available from MOTChallenge.
- Generated results: archived on Zenodo at
  https://doi.org/10.5281/zenodo.20476266
