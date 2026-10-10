# Checkpoint notes

- `runs/B0_v2/weights/best.pt` - raw baseline (B0)
- `runs/B1_v2/weights/best.pt` - UEP, white balance + adaptive saturation (B1)
- `runs/B23/weights/best.pt` - UEP + 309-parameter LCA in the neck (B2)
- `runs/B1_fixed/weights/best.pt` - UEP development run (superseded by B1_v2)
- `runs/B1_v3/weights/best.pt` - UEP development run (superseded by B1_v2)
- `runs/B2_v2/weights/best.pt` - UEP + LCA development run (superseded by B23)

All checkpoints are 3.01 M-parameter YOLOv8n models trained **from scratch** (150 epochs, AdamW lr0 = 0.00125, batch 8, 640x640, seed 0) with the exact hyper-parameters stored next to each checkpoint in `args.yaml`. Total 35.8 MB.
The three-seed results reported in the paper were produced by the multi-seed scripts in `scripts/` (`m5_full_multiseed.py`, `m6_b2ws_multiseed.py`); the corresponding seeds are seeds 0/1/2 and their per-seed values are in `results/`. Weights of the additional multi-seed runs are available from the author on request.


## Single-component ablation checkpoints (added in `v1.1-revision`)

`runs/B1_wb{,_s1,_s2}/weights/best.pt` - UEP white-balance stage only (seeds 0/1/2)
`runs/B1_sat{,_s1,_s2}/weights/best.pt` - UEP adaptive-saturation stage only (seeds 0/1/2)
`runs/B1_clahe{,_s1,_s2}/weights/best.pt` - CLAHE-only baseline (seeds 0/1/2)

Same protocol as the baseline runs (from scratch, 150 epochs, AdamW lr0 = 0.00125, batch 8,
640x640, patience = 0, identical train/test operator for each configuration); each directory
contains the exact `args.yaml` of that run. Every checkpoint is evaluated by
`scripts/p7_3_ablation.py` and `scripts/per_seed_dump.py`; the resulting per-seed, per-class
numbers are in `results_per_seed/`.

Weights of the multi-seed *baseline* runs (B0, B1 and B2 at seeds 1/2) are available from the
author on request; their per-seed, per-class evaluation values are released in
`results_per_seed/results_dump/`.
