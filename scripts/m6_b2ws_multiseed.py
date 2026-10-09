# m6_b2ws_multiseed.py — B2ws (YOLOv8n+LCA on WB+sat data) 3-seed 消融, 与 B1ws 协议完全一致
# AutoDL: cd /root/autodl-tmp/Underwater && python scripts/m6_b2ws_multiseed.py
# 训练 ~2h/seed, 3 seeds 共约 6h; 若 B2ws 权重已存在则跳过训练只做 val
import sys
from pathlib import Path
import numpy as np

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")
SEEDS = [0, 1, 2]
NAMES = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}

# ---- attention 模块路径探测(在多个可能位置查找) ----
for _p in [BASE, BASE / "ultralytics_patch", BASE / "scripts",
           Path("/root/autodl-tmp"), Path("/root")]:
    if (_p / "attention.py").exists():
        sys.path.insert(0, str(_p))
        print("attention.py 位于:", _p, flush=True)
        break
else:
    print("!! 未找到 attention.py,请先上传 ultralytics_patch/attention.py 到服务器", flush=True)
    sys.exit(1)

from attention import register_attention
register_attention()


def wpath(name):
    return BASE / "runs/detect" / name / "weights" / "best.pt"


def train(name, seed):
    print("train %s seed=%d ..." % (name, seed), flush=True)
    from ultralytics import YOLO
    model = YOLO(str(BASE / "configs/yolov8n_attn.yaml"))
    model.train(data=str(DATA / "DUO_wb_sat.yaml"), name=name, seed=seed,
                project=str(BASE / "runs/detect"),
                epochs=150, batch=8, imgsz=640, pretrained=False, patience=0,
                device=0, workers=8, optimizer="AdamW", lr0=0.00125,
                mixup=0.0, copy_paste=0.0, exist_ok=True)
    print("train %s done" % name, flush=True)


def val(name):
    from ultralytics import YOLO
    m = YOLO(str(wpath(name)))
    r = m.val(data=str(DATA / "DUO_wb_sat.yaml"), imgsz=640, verbose=False)
    ap50 = list(r.box.ap50)
    idx = list(r.box.ap_class_index)
    per = {i: a for i, a in zip(idx, ap50)}
    return float(r.box.map50), float(r.box.map), per


def main():
    results = []
    per_class = {c: [] for c in range(4)}
    for seed in SEEDS:
        name = "B2ws" if seed == 0 else "B2ws_s%d" % seed
        if not wpath(name).exists():
            train(name, seed)
        m50, m95, per = val(name)
        results.append((seed, m50, m95))
        for c in range(4):
            per_class[c].append(per.get(c, 0.0))
        print("val %s seed=%d : map50=%.4f map=%.4f per=%s" % (
            name, seed, m50, m95,
            [round(per.get(c, 0.0), 4) for c in range(4)]), flush=True)

    m50s = np.array([r[1] for r in results], dtype=float)
    m95s = np.array([r[2] for r in results], dtype=float)
    L = ["B2ws (YOLOv8n+LCA, WB+sat data) 3-seed", ""]
    for (s, m50, m95) in results:
        L.append("seed%d : map50=%.4f  map=%.4f" % (s, m50, m95))
    L.append("")
    L.append("B2ws 3seed : map50 = %.4f (std %.4f)   map = %.4f (std %.4f)" % (
        m50s.mean(), m50s.std(ddof=1), m95s.mean(), m95s.std(ddof=1)))
    L.append("")
    L.append("compare B1ws 3seed : map50 0.8064 (std 0.0022)   map 0.5995 (std 0.0037)")
    L.append("compare B0 raw 3seed: map50 0.8058 (std 0.0036)   map 0.6050 (std 0.0033)")
    L.append("")
    L.append("per-class AP50 (3-seed mean +- std):")
    for c in range(4):
        a = np.array(per_class[c], dtype=float)
        L.append("  %-12s : %.4f +- %.4f" % (NAMES[c], a.mean(), a.std(ddof=1)))
    out = "\n".join(L)
    print("\n" + out, flush=True)
    (BASE / "m6_b2ws_result.txt").write_text(out, encoding="utf-8")
    print("Saved: m6_b2ws_result.txt", flush=True)


main()
