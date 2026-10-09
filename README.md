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
  **+1.7 AP50 points** at zero parameters and ≈ 20 ms/image on a single CPU thread.
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
