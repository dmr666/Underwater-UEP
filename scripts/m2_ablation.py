"""
M2 逐项消融: UEP 三项操作 (白平衡/CLAHE/饱和度) 各自贡献 (AutoDL)

背景: UEP = 白平衡(WB) + CLAHE + 饱和度 三个退化感知操作串行,
      论文未逐项消融, 审稿人会问"哪个操作贡献最大、能否删掉某个"。

方案: 三个 single-op 变体 (只保留一个操作, 自适应强度, 复用 uep.py 同一函数),
      各训一个 YOLOv8n (seed 0, 与 B0/B1 一致), 与 raw (B0) 和 full-UEP (B1) 对比。

变体:
  B1_wb    = 仅白平衡
  B1_clahe = 仅 CLAHE
  B1_sat   = 仅饱和度

用法:
  cd /root/autodl-tmp/Underwater && python scripts/m2_ablation.py
输出: 终端 + m2_result.txt
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")
sys.path.insert(0, str(BASE / "scripts"))
from uep import (estimate_cast, estimate_contrast, estimate_saturation,
                 adaptive_white_balance, adaptive_clahe, adaptive_saturate)


def wb_only(img):
    return adaptive_white_balance(img, estimate_cast(img))


def clahe_only(img):
    return adaptive_clahe(img, estimate_contrast(img))


def sat_only(img):
    return adaptive_saturate(img, estimate_saturation(img))


OPS = [("wb", wb_only), ("clahe", clahe_only), ("sat", sat_only)]

CFG = dict(epochs=150, batch=8, imgsz=640, pretrained=False, patience=0,
           device=0, workers=8, optimizer="AdamW", lr0=0.00125,
           mixup=0.0, copy_paste=0.0, seed=0, exist_ok=False)


def wpath(name):
    return BASE / "runs/detect" / name / "weights" / "best.pt"


def _worker(args):
    src, dst, op = args
    img = cv2.imread(str(src))
    if img is None:
        return str(src.name)
    if op == "wb":
        out = wb_only(img)
    elif op == "clahe":
        out = clahe_only(img)
    else:
        out = sat_only(img)
    cv2.imwrite(str(dst), out)
    return None


def batch_enhance(op, func, src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.jpg"))
    todo = [(f, dst_dir / f.name, op) for f in files if not (dst_dir / f.name).exists()]
    if not todo:
        print(f"[gen] {dst_dir.name}: 已全部存在 ({len(files)} 张), 跳过", flush=True)
        return
    import concurrent.futures as cf
    n = max(1, (os.cpu_count() or 4) - 1)
    failed = []
    with cf.ProcessPoolExecutor(max_workers=n) as ex:
        for i, err in enumerate(ex.map(_worker, todo, chunksize=16)):
            if err:
                failed.append(err)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(todo)}", flush=True)
    print(f"[gen] {dst_dir.name}: 完成 {len(todo)} 张 (失败 {len(failed)})", flush=True)


def prepare(op, func):
    batch_enhance(op, func, DATA / "images/train", DATA / f"images/train_{op}")
    batch_enhance(op, func, DATA / "images/test", DATA / f"images/test_{op}")
    for split in ("train", "test"):
        src_lab = DATA / f"labels/{split}"
        dst_lab = DATA / f"labels/{split}_{op}"
        if dst_lab.exists() or dst_lab.is_symlink():
            dst_lab.unlink()
        dst_lab.symlink_to(src_lab, target_is_directory=True)
    y = DATA / f"DUO_{op}.yaml"
    y.write_text(
        f"path: {DATA.as_posix()}\n"
        f"train: images/train_{op}\n"
        f"val: images/test_{op}\n"
        f"test: images/test_{op}\n"
        "names:\n"
        "  0: holothurian\n"
        "  1: echinus\n"
        "  2: scallop\n"
        "  3: starfish\n", encoding="utf-8")
    print(f"[yaml] DUO_{op}.yaml 已生成", flush=True)


def train(name, data_yaml):
    print(f"[train] {name}  data={data_yaml} ...", flush=True)
    model = YOLO("yolov8n.yaml")
    model.train(data=str(DATA / data_yaml), name=name, **CFG)
    print(f"[train] {name} 完成", flush=True)


def val(name, data_yaml):
    m = YOLO(str(wpath(name)))
    r = m.val(data=str(DATA / data_yaml), imgsz=640, verbose=False)
    return round(float(r.box.map50), 4), round(float(r.box.map), 4)


def main():
    lines = ["M2 逐项消融: UEP 三操作贡献 (2026-08-27)", ""]
    # 基准: raw (B0) 与 full UEP (B1), 复用已有权重重新 val 保证同口径
    for tag, name, data_yaml in [
        ("raw (B0)",      "B0_v2", "DUO.yaml"),
        ("full UEP (B1)", "B1_v3", "DUO_uep.yaml"),
    ]:
        m50, m95 = val(name, data_yaml)
        print(f"[val] {tag} ({name}): mAP50={m50}  mAP50-95={m95}", flush=True)
        lines.append(f"{tag:<16}: mAP50={m50}  mAP50-95={m95}")

    for op, func in OPS:
        prepare(op, func)
        name = f"B1_{op}"
        if not wpath(name).exists():
            train(name, f"DUO_{op}.yaml")
        m50, m95 = val(name, f"DUO_{op}.yaml")
        print(f"[val] {op}-only ({name}): mAP50={m50}  mAP50-95={m95}", flush=True)
        lines.append(f"{op}-only{'':<9}: mAP50={m50}  mAP50-95={m95}")

    out = "\n".join(lines)
    print("\n" + out)
    (BASE / "m2_result.txt").write_text(out, encoding="utf-8")
    print("\n结果已存: m2_result.txt")


if __name__ == "__main__":
    main()
