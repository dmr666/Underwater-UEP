# -*- coding: utf-8 -*-
"""COCO 评估诊断: 检查 pycocotools 环境 + 手动 IoU 匹配 (容器端运行)"""
import json
import importlib.metadata

try:
    print("pycocotools 版本:", importlib.metadata.version("pycocotools"))
except Exception as e:
    print("版本获取失败:", e)

from pycocotools.coco import COCO

GT = "/root/autodl-tmp/DUO/annotations/instances_test.json"
PRED = "/root/autodl-tmp/Underwater/runs/detect/B06/predictions.json"  # 未反变换版

gt = COCO(GT)
pred = gt.loadRes(PRED)


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


for iid in [662, 1]:
    gts = gt.loadAnns(gt.getAnnIds(imgIds=[iid]))
    dts = pred.loadAnns(pred.getAnnIds(imgIds=[iid]))
    print(f"图 {iid}: GT {len(gts)} 个 | 预测 {len(dts)} 个")
    best = 0.0
    for g in gts[:5]:
        for d in dts[:50]:
            v = iou(g["bbox"], d["bbox"])
            if v > best:
                best = v
    print(f"  最佳手动 IoU: {best:.3f}")
    if gts and dts:
        print("  GT[0] bbox:", gts[0]["bbox"], "| cat:", gts[0]["category_id"])
        print("  DT[0] bbox:", dts[0]["bbox"], "| cat:", dts[0]["category_id"])
        print("  GT 图片尺寸:", gt.imgs[iid])
