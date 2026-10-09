# -*- coding: utf-8 -*-
"""COCOeval 深层诊断: 测 _mask.iou 扩展 + 最小 COCOeval 复现 (容器端运行)"""
import json
import tempfile
import importlib.metadata

try:
    print("pycocotools 版本:", importlib.metadata.version("pycocotools"))
except Exception as e:
    print("版本获取失败:", e)

import numpy as np
from pycocotools import _mask

# 1) _mask.iou 自测: 两个完全重叠的框, 期望输出 1.0
dt = np.array([[0, 0, 10, 10]], dtype=np.float64)
gt = np.array([[0, 0, 10, 10]], dtype=np.float64)
try:
    iou_out = _mask.iou(dt, gt, 1, 1)
    print("_mask.iou 自测 (期望 ~1.0):", iou_out)
except Exception as e:
    print("_mask.iou 调用失败:", e)

# 2) 最小 COCOeval 复现 (与 ap_by_size.py 完全相同的预处理)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_PATH = "/root/autodl-tmp/DUO/annotations/instances_test.json"
PRED_PATH = "/root/autodl-tmp/Underwater/runs/detect/B06/predictions.json"

gt_raw = json.load(open(GT_PATH, encoding="utf-8"))
for a in gt_raw["annotations"]:
    a["category_id"] -= 1
for c in gt_raw["categories"]:
    c["id"] -= 1
tmp_gt = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
json.dump(gt_raw, tmp_gt)
tmp_gt.close()

gt = COCO(tmp_gt.name)
pred = gt.loadRes(PRED_PATH)
print("GT 图数:", len(gt.imgs), "| GT 标注:", len(gt.dataset["annotations"]),
      "| 预测条数:", len(pred.dataset["annotations"]))
print("GT categories:", [(c["id"], c["name"]) for c in gt.dataset["categories"]])
print("预测 category 集合:", sorted(set(a["category_id"] for a in pred.dataset["annotations"])))

ev = COCOeval(gt, pred, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()
print("stats[:6]:", [round(float(x), 4) for x in ev.stats[:6]])

# 3) 手动用 accumulate 数据核对: 某张图的匹配数
try:
    iid = 1
    gts = gt.loadAnns(gt.getAnnIds(imgIds=[iid]))
    dts = pred.loadAnns(pred.getAnnIds(imgIds=[iid]))
    iou_mat = _mask.iou(
        np.array([d["bbox"] for d in dts], dtype=np.float64),
        np.array([g["bbox"] for g in gts], dtype=np.float64),
        len(dts), len(gts))
    print(f"图 {iid}: _mask.iou 矩阵 shape={iou_mat.shape}, 最大值={iou_mat.max() if iou_mat.size else 0}")
except Exception as e:
    print("图级 _mask.iou 失败:", e)
