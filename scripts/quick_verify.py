# -*- coding: utf-8 -*-
"""
验证: 全 test 用"子集格式"重建后 val, 对比官方全集数字
定位 分档结果(每档 B1<B0) vs 全集结果(B1>B0) 的矛盾
"""
import shutil
from pathlib import Path
import cv2

BASE = Path("/root/autodl-tmp/Underwater")
DUO = Path("/root/autodl-tmp/DUO")
TEST_IMG = DUO / "images" / "test"
TEST_LAB = DUO / "labels" / "test"

names = sorted(f.stem for f in TEST_IMG.glob("*.jpg") if (TEST_LAB / f"{f.stem}.txt").exists())
print(f"有标签图: {len(names)}")

for tag, use_uep in [("raw", False), ("uep", True)]:
    sub = BASE / f"viz_all_{tag}"
    (sub / "images").mkdir(parents=True, exist_ok=True)
    (sub / "labels").mkdir(parents=True, exist_ok=True)
    for n in names:
        shutil.copy(TEST_IMG / f"{n}.jpg", sub / "images" / f"{n}.jpg")
        shutil.copy(TEST_LAB / f"{n}.txt", sub / "labels" / f"{n}.txt")
        if use_uep:
            img = cv2.imread(str(sub / "images" / f"{n}.jpg"))
            from uep import enhance
            cv2.imwrite(str(sub / "images" / f"{n}.jpg"), enhance(img))
    yaml_p = BASE / f"all_{tag}.yaml"
    yaml_p.write_text(f"path: {sub}\ntrain: images\nval: images\nnames:\n  0: holothurian\n  1: echinus\n  2: scallop\n  3: starfish\n")
    print(f"[{tag}] 子集就绪: {sub}")

from ultralytics import YOLO
m0 = YOLO(str(BASE / "runs/detect/B0_v2/weights/best.pt"))
m1 = YOLO(str(BASE / "runs/detect/B1_v2/weights/best.pt"))

r0 = m0.val(data=str(BASE / "all_raw.yaml"), imgsz=640, verbose=False)
r1 = m1.val(data=str(BASE / "all_uep.yaml"), imgsz=640, verbose=False)
print(f"\n[子集格式] B0_v2: mAP50={r0.box.map50:.4f}  mAP50-95={r0.box.map:.4f}")
print(f"[子集格式] B1_v2: mAP50={r1.box.map50:.4f}  mAP50-95={r1.box.map:.4f}")
print("\n参考: 官方全集 B0_v2=0.805/0.603, B1_v2=0.811/0.598 (1111张含11背景)")
