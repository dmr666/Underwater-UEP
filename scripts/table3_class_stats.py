# table3_class_stats.py — 复现论文 Table 3 的逐类退化统计（color cast / saturation）
# 依据论文定义：c = max_ch |mean_gray/mean_ch - 1| (同 uep.py::estimate_cast)；s = mean over pixels of (max-min)/max；
# 图像按最长边缩到 512 像素；统计对象为"含该类目标的训练图"。
# 用法: python scripts/table3_class_stats.py --images /root/autodl-tmp/DUO/images/train \
#                                            --labels /root/autodl-tmp/DUO/labels/train \
#                                            --out table3_class_stats.json
import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

NAMES = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}


def stats_of(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    sc = 512.0 / max(h, w)
    if sc < 1:
        img = cv2.resize(img, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
    f = img.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    gray = float(means.mean())
    cast = float(np.max(np.abs(gray / means - 1.0))) if gray > 0 else 0.0
    mx, mn = f.max(2), f.min(2)
    sat = float(np.mean((mx - mn) / np.maximum(mx, 1e-6)))
    return cast, sat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="/root/autodl-tmp/DUO/images/train")
    ap.add_argument("--labels", default="/root/autodl-tmp/DUO/labels/train")
    ap.add_argument("--out", default="table3_class_stats.json")
    a = ap.parse_args()
    img_dir, lab_dir = Path(a.images), Path(a.labels)

    per_class = defaultdict(list)
    allv = []
    missing = 0
    files = sorted(lab_dir.glob("*.txt"))
    for i, lp in enumerate(files):
        cls = {int(l.split()[0]) for l in lp.read_text(errors="ignore").splitlines() if len(l.split()) >= 5}
        if not cls:
            continue
        ip = next((img_dir / (lp.stem + e) for e in (".jpg", ".JPG", ".jpeg", ".png")
                   if (img_dir / (lp.stem + e)).exists()), None)
        if ip is None:
            missing += 1
            continue
        v = stats_of(ip)
        if v is None:
            missing += 1
            continue
        allv.append(v)
        for c in cls:
            per_class[c].append(v)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    out = {"definition": "c = max_ch |mean_gray/mean_ch - 1| (同 uep.py::estimate_cast); s = mean((max-min)/max); images resized to 512 longest side",
           "images_scanned": len(files), "images_missing": missing, "classes": {}, "all": {}}
    for c in sorted(per_class):
        v = per_class[c]
        out["classes"][NAMES[c]] = {
            "images": len(v),
            "cast_mean": round(st.mean(x[0] for x in v), 3), "cast_sd": round(st.pstdev(x[0] for x in v), 3),
            "sat_mean": round(st.mean(x[1] for x in v), 3), "sat_sd": round(st.pstdev(x[1] for x in v), 3)}
    out["all"] = {"images": len(allv),
                  "cast_mean": round(st.mean(x[0] for x in allv), 3), "cast_sd": round(st.pstdev(x[0] for x in allv), 3),
                  "sat_mean": round(st.mean(x[1] for x in allv), 3), "sat_sd": round(st.pstdev(x[1] for x in allv), 3)}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


main()
