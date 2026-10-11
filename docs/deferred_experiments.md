# Deferred experiments (constant sensitivity / graded enhancement strength)

Two reviewer-facing extensions were specified as possible robustness checks; **both have now been run** (RTX 4090, same protocol as the paper) and are reported in the revised manuscript (Sec. 4.3). Results: `results_deferred/`. The runner is `scripts/deferred_experiments.py` and uses the
same protocol as the paper: YOLOv8n from scratch, 150 epochs, batch 8, imgsz 640, AdamW lr0=0.00125,
patience=0 (early stopping disabled), deterministic, seed 0; noise band taken from the released 3-seed
baselines (sample s.d. ~0.0045 mAP50).

## 1. Constant sensitivity (`sensitivity`, 6 runs)
Each hand-set constant (`wb_max=0.9`, `alpha_clip=0.25`, `sat_max=1.35`) perturbed by +/-20%, everything
else fixed. Output: `summary_sensitivity.json`. If every variant stays inside the seed-noise band, the
conclusion can be stated as insensitive to +/-20% perturbation of the constants.

## 2. Graded enhancement strength (`graded`, 2 runs + 1 training-free evaluation)
(a) train 50% / test 50%; (b) train 50% / test 100% (mismatch); (c) the existing 100%-strength checkpoint
evaluated on a 50%-strength test set (reverse mismatch, no training). Extends the consistency result
(Table 3) from one mismatch magnitude to a graded curve.

## Usage (expects `images/{train,test}`, `labels/{train,test}` of DUO at `--data`)
```
python scripts/deferred_experiments.py sensitivity --data /path/to/DUO --runs runs/detect
python scripts/deferred_experiments.py graded      --data /path/to/DUO --runs runs/detect
python scripts/deferred_experiments.py report      --results results_deferred
```
