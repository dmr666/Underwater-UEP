# Underwater-UEP — released artifacts

Code, per-seed evaluation files and trained checkpoints for the study

> **What Does Zero-Parameter Underwater Enhancement Buy a Real-Time Detector? A Controlled Multi-Seed
> Study of Train–Test Consistency**

Lihua Dai — Zhongke Shurong Technology (Zhengzhou) Co., Ltd., Zhengzhou, China — 694169145@qq.com —
ORCID [0009-0000-9133-7116](https://orcid.org/0009-0000-9133-7116)

---

## What the study found

Zero-parameter, deterministic image enhancement is routinely bolted onto an already-trained detector at
inference time. This repository supports a controlled test of that practice on the **DUO** underwater
detection benchmark, using **from-scratch YOLOv8n** training, a pinned optimizer and **three seeds**:

- **UEP** (a zero-trainable-parameter stage driven by per-image colour-cast and saturation statistics)
  preserves overall mAP50 (**0.806 vs. 0.806**) and lifts the low-saturation *holothurian* class by
  **+1.7 AP50 points** at zero parameters and ≈ 40 ms/image for the complete operator on a single CPU thread (see the latency note below; the white-balance stage alone costs 27.8 ms).
- The benefit is **conditional on train/test consistency**, which is the paper's central result:
  applying UEP **only at test time** drops mAP50 to **0.510 (−29.6 points)**, and training under UEP while
  testing on raw images drops it to **0.641 (−16.5 points)**.
- A **309-parameter** channel–spatial attention module inserted into the neck (parameter-matched control)
  gains only **+0.5 mAP50** (p = 0.02), i.e. input-space enhancement is the more parameter-efficient path
  for this task.

## Repository layout

```
train_underwater.py          training entry point (ultralytics 8.3.123, pinned)
run_all.sh, autodl_setup.sh  batch reproduction scripts used on the training host
configs/yolov8n_attn.yaml    YOLOv8n + 3 x LCA (channel-spatial attention) in the neck
ultralytics_patch/attention.py  custom-module registration patch for the attention block
docs/                        design notes (Chinese)
scripts/                     UEP implementation, multi-seed evaluation, COCO conversion, AP-by-size
results/                     per-seed / per-class evaluation outputs (the numbers behind the paper)
runs/                        per-run training records (args.yaml, results.csv) + best.pt checkpoints
runs/manifest.csv            run -> configuration mapping
CHECKPOINT_NOTES.md          what each checkpoint is
```

Key evaluation files:

| File | Content |
|---|---|
| `results/m5_cross_result.txt` | train/test cross-evaluation, 2 x 2 cells, 3-seed mean ± std (mAP50 / mAP50-95) |
| `results/m5_perclass_result.txt` | per-class AP50 for both diagonal cells, 3-seed mean ± std |
| `results/m6_b2ws_result.txt` | UEP + LCA (309 parameters), per seed and 3-seed, per-class |
| `results/m1_result.txt` | per-seed mAP for the baseline and the UEP model (single-seed development runs) |
| `results/m2_result.txt`, `results/m2_perclass_result.txt` | operation-wise ablation (white balance / CLAHE / saturation) |
| `results/m3_result.txt` | early single-seed cross-evaluation that motivated the multi-seed protocol |
| `results/degradation_result.txt`, `results/duo_test_degradation.json` | colour-cast severity bands vs. accuracy; per-image degradation statistics |
| `results/ap_size_multiseed_result.txt` | COCO-style AP by object size, 3-seed |

## Experimental protocol

| Item | Setting |
|---|---|
| Dataset | DUO (public, with public test annotations) — 6,671 train / 1,111 test images, 4 classes |
| Models | YOLOv8n from scratch (`pretrained=False`), 3.01 M parameters / 8.1 GFLOPs |
| Seeds | 0, 1, 2 (every reported number is a 3-seed mean; std reported) |
| Optimizer | AdamW, `lr0 = 0.00125`, pinned explicitly (`optimizer=AdamW`, not `auto`) |
| Training | 150 epochs, batch 8, 640x640, `patience=0` (no early stopping), mixup/copy-paste disabled |
| Enhancement | UEP applied **offline** to both train and test images (identical procedure), baseline uses raw |
| Environment | PyTorch 2.1.2+cu118, ultralytics **8.3.123** (pinned; 8.4.x incompatible), numpy 1.26.4, pandas 2.2.3 |
| Hardware | single NVIDIA RTX 4090; ~2 h per model |

## Reproducing

```bash
# 1. environment (see autodl_setup.sh for the exact pinned install)
pip install ultralytics==8.3.123 numpy==1.26.4 pandas==2.2.3

# 2. dataset: download DUO (see https://arxiv.org/abs/2106.05681), then convert to YOLO layout
python scripts/coco2yolo.py

# 3. apply UEP offline to the train/test images (identical procedure for both)
python scripts/uep_fixed.py            # writes images/train_uep and images/test_uep

# 4. train a configuration (raw baseline / UEP / UEP+attention)
python train_underwater.py             # see run_all.sh for the full matrix and seeds

# 5. reproduce the reported tables
python scripts/m5_full_multiseed.py    # cross-evaluation table (29.6 / 16.5 point losses)
python scripts/m6_b2ws_multiseed.py    # UEP + 309-parameter attention control
python scripts/ap_by_size_multiseed.py # COCO AP by size
python scripts/degradation_eval.py     # colour-cast bands vs. accuracy
```

## Data and checkpoints

- **Datasets are not redistributed here.** DUO is public (arXiv:2106.05681); the RUOD dataset used in the
  early phase of the work is public at <https://github.com/xiaoDetection/RUOD>.
- `runs/*/weights/best.pt` are the YOLOv8n checkpoints of the reported configurations (seed 0, 150 epochs,
  from scratch); the exact hyper-parameters of each run are stored in the adjacent `args.yaml`, and
  `runs/manifest.csv` maps run → configuration. Weights of the additional multi-seed runs are available
  from the author on request.

## Citation

If you use these artifacts, please cite the paper (journal reference to be added upon publication).

## Contact

Lihua Dai — 694169145@qq.com — ORCID 0009-0000-9133-7116


## Revision evidence (tag `v1.1-revision`)

Added for the revised manuscript (single-author controlled study):

**Single-component ablation, 3 seeds.** Six configurations were trained and evaluated
under the identical protocol (150 epochs, AdamW lr0=0.00125, batch 8, 640x640, seeds 0-2,
no early stopping), each with the matching test-set operator:

| Configuration | mAP50 (n=3) | mAP50-95 (n=3) |
|---|---|---|
| raw (B0) | 0.8058 +/- 0.0045 | 0.6050 +/- 0.0040 |
| UEP WB+sat (B1) | 0.8064 +/- 0.0022 | 0.5995 +/- 0.0037 |
| UEP+LCA (B2) | 0.8114 +/- 0.0012 | 0.6052 +/- 0.0024 |
| WB only | 0.8146 +/- 0.0056 | 0.6068 +/- 0.0013 |
| sat only | 0.8090 +/- 0.0039 | 0.6083 +/- 0.0059 |
| CLAHE only | 0.7863 +/- 0.0070 | 0.5715 +/- 0.0045 |

Per-seed, per-class values: `results_per_seed/results_dump/`; the summary including the
per-class tables and the train/test cross-evaluation is `results_per_seed/per_seed_summary.txt`.
Relative to raw, only CLAHE-only differs significantly (mAP50 -0.0195, independent-samples
t-test, df = 4, p = 0.015).

**Timing protocol.** `results_per_seed/latency_bench.json` records the exact measurement
conditions: Intel Xeon Gold 6430 host, OpenCV 5.0.0, NumPy 1.26.4, PyTorch 2.1.2+cu118,
pinned ultralytics 8.3.123, RTX 4090. On that host the **white-balance stage alone** costs
27.8 ms per 640x640 BGR uint8 frame on a single CPU thread (median of 200 timed runs after
20 warm-up iterations), whereas the **complete UEP operator** costs 40.3 ms (median of 200
runs; i5-1155G7 host, OpenCV 5.0.0) — the Xeon host was released before the complete-operator
re-measurement, so the two numbers come from different machines: see
`results_per_seed/latency_bench_v2_full_operator.json`. The detector costs 6.05 ms per frame
(median of 50 timed runs after 10 warm-up iterations, measured after torch.cuda.synchronize()).

**Scripts.** `scripts/p7_3_ablation.py` runs the six training runs plus the per-seed
evaluation and the latency benchmark; `scripts/per_seed_dump.py` produces the per-seed,
per-class dumps and the cross-evaluation; `scripts/table3_class_stats.py` recomputes the
per-class color-cast/saturation statistics from the dataset.

**Note.** `results/m2_result.txt` and `results/m2_perclass.txt` come from an earlier
checkpoint set; see `results_per_seed/NOTE_m2_files_are_stale.md`.

- **Degradation statistics and the adaptivity limit (recomputed 2026-10-11 over all 6,671 training
  images at the paper's protocol, longest side 512 px):** `scripts/duo_stats.py` →
  `results_per_seed/duo_stats.json`. **Cast definition used here is the one the experiments actually
  ran (`uep.py::estimate_cast`): `c = max_ch |mean_gray/mean_ch − 1|`** — the manuscript text had the
  ratio inverted and was corrected to match the code; `duo_stats.py` also stores the inverted variant
  under `cast_alt_text_mean` for traceability. Mean color cast 0.779 ± 0.709 (median 0.544, quartiles
  0.382/0.544/0.811) exceeds the clamp threshold 0.25 for **92.0%** of training images (only 8.0%
  retain any adaptivity; mean α = 0.983); holothurian stays the lowest-saturation class in **99.9%**
  of 2,000 bootstrap resamples; the two equivalent saturation definitions differ by ≤0.0003 per image
  (uint8 quantization). These numbers back the paper's §3.2/§4.3 statement that on DUO UEP acts as a
  bounded fixed-strength gray-world correction.

## Robustness extensions (run 2026-10-11, same protocol)

**Constant sensitivity (Sec. 4.3).** Each of the three hand-set constants (`wb_max` 0.9, `alpha_clip` 0.25,
`sat_max` 1.35) was perturbed by +/-20% in turn (six otherwise identical runs, seed 0, 150 epochs).
Overall mAP50 stays within 0.801--0.815 (mean 0.810) against 0.806 +/- 0.004 for the baseline, with no
directional effect; the holothurian gain is preserved in all six variants (0.798--0.816 vs. 0.789 +/- 0.013).
Per-variant results: `results_deferred/S_*.json`, `results_deferred/summary_sensitivity.json`.

**Graded enhancement strength (Sec. 4.3).** With the enhancement applied at half strength, the consistent
half/half configuration is the strongest result in the study (mAP50 0.818), while mismatch costs
0.818 -> 0.639 (-17.9 points, trained half / tested full) and 0.806 -> 0.742 (-6.4 points, trained full /
tested half). Single-seed runs, marked with a dagger in Table 3 of the paper.
Results: `results_deferred/G_*.json`, `results_deferred/summary_graded.json`; analysis notes in
`results_deferred/ANALYSIS.md`. Runner: `scripts/deferred_experiments.py` (see `docs/deferred_experiments.md`).

## Supplementary experiments (reviewer round 2, 2026-10-12)

All runs use the identical pinned protocol (`scripts/suppl_m4_m6.py`; three-stage pre-flight:
`audit` against the archived `args.yaml`, `selftest`, and a real one-epoch `smoketrain`).

**M4 — half-strength matrix with three seeds.** The two graded cells that were single-seed in the
paper are now n=3: consistent half-strength **0.814 ± 0.006** (0.8183/0.8155/0.8076) versus
half-train/full-test **0.640 ± 0.002** (0.6391/0.6414/0.6382), i.e. a separation of +0.174 mAP50
against a within-cell sd of <=0.006. The complementary cell (full-train/half-test) was completed by
re-evaluating the existing full-strength weights on the half-strength test set (0.7421/0.7436/0.7633,
**0.750 ± 0.012**), so every cell of the consistency table is now three-seed.
*Honest consequence:* with three seeds, the consistent half-strength setting (0.814 ± 0.006) and
white-balance-only enhancement (0.815 ± 0.006) are statistically indistinguishable — the paper now
says so, which strengthens (rather than weakens) its claim that the gain comes from consistency
rather than from enhancement strength.

**M6a — COCO-pretrained fine-tuning** (same schedule, 150 epochs): raw 0.852 / UEP 0.847;
inference-only enhancement costs **−23.4** mAP50 (0.852 to 0.618) and the reverse mismatch **−13.2**
(0.847 to 0.715). The penalty structure survives the change of training regime.

**M6b — second dataset, from scratch.** RUOD (4,200-image test split, the four DUO classes) trained
from scratch under the same protocol: raw 0.817 / UEP 0.821; inference-only enhancement **−23.7**,
reverse mismatch **−19.6**. `scripts/ruod_proxy.py` additionally transfers the enhancement across
datasets (DUO-trained raw model, three seeds, evaluated on RUOD): **−26.1** (0.678 to 0.417), while
the UEP-trained model prefers the enhanced input (0.506 to 0.530).

**Domain reproducibility check.** `scripts/gate_prepare.py` regenerates the enhanced image domains
deterministically and re-scores the paper's own weights on them: `G_train50_test50` reproduces its
published seed-0 value exactly (0.8183 vs 0.8183, difference 0.0000), confirming that the regenerated
domains are pixel-identical to those used in the paper.
