# WB+sat multiseed: drop CLAHE, keep white-balance + saturation (UEP v2)
# AutoDL run: cd /root/autodl-tmp/Underwater && python scripts/m4_wb_sat_multiseed.py
# Steps: gen train/test_wb_sat data -> train 3 seeds -> val -> mean +- std
# Output: terminal + m4_wb_sat_result.txt
# Compare against B0(raw) 3seed 0.8058/0.6050 and B1(UEPv1) 3seed 0.8099/0.5947

import os
import sys
import numpy as np
from pathlib import Path
import cv2

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")

sys.path.insert(0, str(BASE / "scripts"))
from uep import estimate_cast, adaptive_white_balance, estimate_saturation, adaptive_saturate

SEEDS = [0, 1, 2]


def wb_sat_enhance(img):
    cast = estimate_cast(img)
    sat_val = estimate_saturation(img)
    out = adaptive_white_balance(img, cast)
    out = adaptive_saturate(out, sat_val)
    return out


def worker(args):
    src, dst = args
    img = cv2.imread(str(src))
    if img is None:
        return str(src.name)
    cv2.imwrite(str(dst), wb_sat_enhance(img))
    return None


def batch_enhance(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    names = sorted([f for f in os.listdir(str(src_dir)) if f.endswith(".jpg")])
    todo = [(src_dir / f, dst_dir / f) for f in names if not (dst_dir / f).exists()]
    if not todo:
        print("skip gen %s : all exist" % dst_dir.name)
        return
    import concurrent.futures as cf
    n = max(1, (os.cpu_count() or 4) - 1)
    failed = []
    with cf.ProcessPoolExecutor(max_workers=n) as ex:
        for i, err in enumerate(ex.map(worker, todo, chunksize=16)):
            if err:
                failed.append(err)
            if (i + 1) % 2000 == 0:
                print("  %d/%d" % (i + 1, len(todo)))
    print("gen %s : done %d failed %d" % (dst_dir.name, len(todo), len(failed)))


def prepare():
    batch_enhance(DATA / "images" / "train", DATA / "images" / "train_wb_sat")
    batch_enhance(DATA / "images" / "test", DATA / "images" / "test_wb_sat")
    for split in ("train", "test"):
        src_lab = DATA / "labels" / split
        dst_lab = DATA / "labels" / (split + "_wb_sat")
        if dst_lab.exists() or dst_lab.is_symlink():
            dst_lab.unlink()
        dst_lab.symlink_to(src_lab, target_is_directory=True)
    y = DATA / "DUO_wb_sat.yaml"
    y.write_text(
        "path: " + DATA.as_posix() + "\n"
        "train: images/train_wb_sat\n"
        "val: images/test_wb_sat\n"
        "test: images/test_wb_sat\n"
        "names:\n"
        "  0: holothurian\n"
        "  1: echinus\n"
        "  2: scallop\n"
        "  3: starfish\n", encoding="utf-8")
    print("yaml ready: DUO_wb_sat.yaml")


def wpath(name):
    return BASE / "runs" / "detect" / name / "weights" / "best.pt"


def train(name, seed):
    print("train %s seed=%d ..." % (name, seed))
    from ultralytics import YOLO
    model = YOLO("yolov8n.yaml")
    model.train(data=str(DATA / "DUO_wb_sat.yaml"), name=name, seed=seed,
                project=str(BASE / "runs" / "detect"),
                epochs=150, batch=8, imgsz=640, pretrained=False, patience=0,
                device=0, workers=8, optimizer="AdamW", lr0=0.00125,
                mixup=0.0, copy_paste=0.0, exist_ok=True)
    print("train %s done" % name)


def val(name):
    from ultralytics import YOLO
    m = YOLO(str(wpath(name)))
    r = m.val(data=str(DATA / "DUO_wb_sat.yaml"), imgsz=640, verbose=False)
    return round(float(r.box.map50), 4), round(float(r.box.map), 4)


def main():
    prepare()
    lines = ["wbsat multiseed (drop CLAHE, keep wb + sat, 3 seeds)", ""]
    results = []
    for seed in SEEDS:
        name = "B1ws" if seed == 0 else ("B1ws_s%d" % seed)
        if not wpath(name).exists():
            train(name, seed)
        m50, m95 = val(name)
        results.append((seed, m50, m95))
        print("val %s seed=%d : map50=%.4f map=%.4f" % (name, seed, m50, m95))
        lines.append("seed%d : map50=%.4f  map=%.4f" % (seed, m50, m95))

    m50s = np.array([r[1] for r in results])
    m95s = np.array([r[2] for r in results])
    lines.append("")
    lines.append("B1ws 3seed : map50 = %.4f (std %.4f)   map = %.4f (std %.4f)" % (
        float(m50s.mean()), float(m50s.std(ddof=1)), float(m95s.mean()), float(m95s.std(ddof=1))))
    lines.append("")
    lines.append("compare B0 raw 3seed   : map50 0.8058 std 0.0036   map 0.6050 std 0.0033")
    lines.append("compare B1 UEPv1 3seed : map50 0.8099 std 0.0034   map 0.5947 std 0.0021")
    lines.append("")
    lines.append("target: map50 keep about +0.005, map flip to positive vs raw 0.6050")
    out = "\n".join(lines)
    print("\n" + out)
    (BASE / "m4_wb_sat_result.txt").write_text(out, encoding="utf-8")
    print("\nSaved: m4_wb_sat_result.txt")


main()
