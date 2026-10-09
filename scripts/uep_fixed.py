# -*- coding: utf-8 -*-
"""
UEP-Fixed: 固定强度增强 (消融对比用, 非自适应)
与 UEP 自适应版唯一区别: 三项增强强度全部固定为最大 (alpha=1, clip=4.5, sat=1.35),
不根据图像退化统计逐图调节 —— 代表传统固定参数增强 (gray-world 全强度 + 强 CLAHE + 强饱和度)。
用途: 证明"自适应调节"本身是 UEP 增益的关键 (论文 4.3 消融 + 审稿人回应)。

容器用法 (在 /root/autodl-tmp/Underwater):
  python scripts/uep_fixed.py --prepare
    # 生成 images/train_uep_fixed + images/test_uep_fixed (增强图)
    # 软链 labels/train_uep_fixed -> labels/train, labels/test_uep_fixed -> labels/test
    # 生成 DUO_uep_fixed.yaml
  然后训练:
  python train_underwater.py --model yolov8n.yaml --data DUO_uep_fixed.yaml --name B1_fixed --epochs 150 --batch 8
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path("/root/autodl-tmp/Underwater")
DUO = Path("/root/autodl-tmp/DUO")
sys.path.insert(0, str(BASE / "scripts"))


def enhance_fixed(img: np.ndarray) -> np.ndarray:
    """固定最大强度增强: 白平衡 alpha=1, CLAHE clip=4.5, 饱和度 x1.35 (全部取自适应范围的上限)。"""
    # 1) 白平衡: 全强度 (alpha=1)
    mean_bgr = np.mean(img, axis=(0, 1))
    gray = np.mean(mean_bgr)
    ratios = gray / np.maximum(mean_bgr, 1e-6)
    gains = 1.0 + (ratios - 1.0) * 0.9
    out = np.clip(img.astype(np.float32) * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)
    # 2) CLAHE: 固定 clip=4.5 (最大)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
    out = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    # 3) 饱和度: 固定 x1.35 (最大)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.35, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _worker(args):
    src, dst = args
    img = cv2.imread(str(src))
    if img is None:
        return str(src.name)
    cv2.imwrite(str(dst), enhance_fixed(img))
    return None


def batch(src_dir: Path, dst_dir: Path):
    import concurrent.futures as cf
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.jpg"))
    tasks = [(f, dst_dir / f.name) for f in files]
    n_workers = max(1, (os.cpu_count() or 4) - 1)
    failed = []
    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, err in enumerate(ex.map(_worker, tasks, chunksize=16)):
            if err:
                failed.append(err)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(files)}")
    print(f"[uep_fixed] {src_dir.name}: {len(files)} 张 -> {dst_dir} (失败 {len(failed)})")
    return failed


def prepare():
    assert DUO.exists(), f"DUO 数据目录不存在: {DUO}"
    # 1) 增强图 (train + test)
    batch(DUO / "images/train", DUO / "images/train_uep_fixed")
    batch(DUO / "images/test", DUO / "images/test_uep_fixed")
    # 2) labels 软链 (ultralytics 镜像目录名)
    for split in ("train", "test"):
        src_lab = DUO / f"labels/{split}"
        dst_lab = DUO / f"labels/{split}_uep_fixed"
        if dst_lab.exists() or dst_lab.is_symlink():
            dst_lab.unlink()
        dst_lab.symlink_to(src_lab, target_is_directory=True)
        print(f"[labels] {dst_lab.name} -> {src_lab.name}")
    # 3) yaml
    y = DUO / "DUO_uep_fixed.yaml"
    y.write_text(
        f"path: {DUO.as_posix()}\n"
        "train: images/train_uep_fixed\n"
        "val: images/test_uep_fixed\n"
        "test: images/test_uep_fixed\n"
        "names:\n"
        "  0: holothurian\n"
        "  1: echinus\n"
        "  2: scallop\n"
        "  3: starfish\n", encoding="utf-8")
    print(f"[yaml] {y.name} 已生成")
    print("\n下一步训练:")
    print("  python train_underwater.py --model yolov8n.yaml --data DUO_uep_fixed.yaml --name B1_fixed --epochs 150 --batch 8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true", help="生成固定强度增强集 + labels 软链 + yaml")
    args = ap.parse_args()
    if args.prepare:
        prepare()
    else:
        print("用法: python scripts/uep_fixed.py --prepare")
