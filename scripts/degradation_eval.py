# -*- coding: utf-8 -*-
"""
退化-精度关联分析: 按色偏程度分档, 对比 B0_v2(原始) vs B1_v3(自适应UEP) 每档 AP50
用法: python scripts/degradation_eval.py  (在 AutoDL /root/autodl-tmp/Underwater 下运行)
输出: 控制台 + degradation_result.txt (每档 mAP50 对比表)
"""
import shutil, sys
from pathlib import Path
import numpy as np
import cv2

BASE = Path("/root/autodl-tmp/Underwater")
DUO = Path("/root/autodl-tmp/DUO")
TEST_IMG = DUO / "images" / "test"
TEST_LAB = DUO / "labels" / "test"
sys.path.insert(0, str(BASE / "scripts"))   # 保证 from uep import enhance 可用


def estimate(img):
    means = img.mean(axis=(0, 1))
    gray_mean = means.mean()
    ratios = gray_mean / np.maximum(means, 1e-6)
    cast = float(np.max(np.abs(ratios - 1)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = float(np.min([gray.std() / 128, 1.0]))
    return cast, contrast


def bucket(cast):
    if cast < 0.15:
        return "light"
    if cast < 0.40:
        return "mid"
    return "severe"


# 1) 分档 + 描述统计
buckets = {"light": [], "mid": [], "severe": []}
stats = {"light": {}, "mid": {}, "severe": {}}
for f in sorted(TEST_IMG.glob("*.jpg")):
    img = cv2.imread(str(f))
    if img is None:
        continue
    cast, contrast = estimate(img)
    k = bucket(cast)
    buckets[k].append(f.stem)
    stats[k].setdefault("cast", []).append(cast)
    stats[k].setdefault("contrast", []).append(contrast)

lines = ["退化-精度关联分析 (色偏分档, DUO test 集, 2026-08-24)"]
print("[分档] 色偏分档 (cast<0.15 light / <0.40 mid / >=0.40 severe):")
for k in ("light", "mid", "severe"):
    ids = buckets[k]
    n = len(ids)
    if n == 0:
        print(f"[分档] {k}: 0 张, 跳过")
        continue
    cs = stats[k]["cast"]
    ct = stats[k]["contrast"]
    msg = f"{k}: {n} 张 | cast 均值 {sum(cs)/n:.3f} | contrast 均值 {sum(ct)/n:.3f}"
    print(f"[分档] {msg}")
    lines.append(msg)

with open(BASE / "degradation_result.txt", "w") as w:
    w.write("退化-精度关联分析 (色偏分档, DUO test 集)\n")
    w.write(f"{'档':<8}{'图数':<8}{'cast均值':<10}{'B0 mAP50':<12}{'B1 mAP50':<12}{'Delta':<8}\n")

# 2) 每档建子集 (原始 + 增强), 跑 val
from ultralytics import YOLO
m0 = YOLO(str(BASE / "runs/detect/B0_v2/weights/best.pt"))
m1 = YOLO(str(BASE / "runs/detect/B1_v3/weights/best.pt"))
print("[模型] B0_v2 + B1_v3 已加载")

for k, names in buckets.items():
    if len(names) < 5:
        print(f"[跳过] {k}: 图太少 ({len(names)})")
        continue
    # 剔除无标签图 (纯背景, 数据固有)
    names = [n for n in names if (TEST_LAB / f"{n}.txt").exists()]
    if len(names) < 5:
        print(f"[跳过] {k}: 有标签图太少 ({len(names)})")
        continue
    # 建原始子集 + UEP 子集
    sub_raw = BASE / f"viz_sub_{k}"
    sub_uep = BASE / f"viz_sub_{k}_uep"
    for sub, use_uep in [(sub_raw, False), (sub_uep, True)]:
        if sub.exists():
            shutil.rmtree(sub)
        (sub / "images").mkdir(parents=True, exist_ok=True)
        (sub / "labels").mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copy(TEST_IMG / f"{n}.jpg", sub / "images" / f"{n}.jpg")
            shutil.copy(TEST_LAB / f"{n}.txt", sub / "labels" / f"{n}.txt")
            if use_uep:
                img = cv2.imread(str(sub / "images" / f"{n}.jpg"))
                from uep import enhance
                cv2.imwrite(str(sub / "images" / f"{n}.jpg"), enhance(img))
    yaml_raw = BASE / f"sub_{k}.yaml"
    yaml_raw.write_text(f"path: {sub_raw}\ntrain: images\nval: images\nnames:\n  0: holothurian\n  1: echinus\n  2: scallop\n  3: starfish\n")
    yaml_uep = BASE / f"sub_{k}_uep.yaml"
    yaml_uep.write_text(f"path: {sub_uep}\ntrain: images\nval: images\nnames:\n  0: holothurian\n  1: echinus\n  2: scallop\n  3: starfish\n")
    r0 = m0.val(data=str(yaml_raw), imgsz=640, verbose=False)
    r1 = m1.val(data=str(yaml_uep), imgsz=640, verbose=False)
    a0, a1 = r0.box.map50, r1.box.map50
    cast_mean = sum(stats[k]["cast"]) / len(stats[k]["cast"])
    print(f"[结果] {k}: B0={a0:.4f}  B1={a1:.4f}  Delta={a1-a0:+.4f}  (cast均值 {cast_mean:.3f})")
    with open(BASE / "degradation_result.txt", "a") as w:
        w.write(f"{k:<8}{len(names):<8}{cast_mean:<10.3f}{a0:<12.4f}{a1:<12.4f}{a1-a0:<+8.4f}\n")

print("\n完成! 结果见 degradation_result.txt")
