# p7_3_ablation.py — 三操作消融（wb-only / clahe-only / sat-only）× 3 种子 + 逐种子逐类 dump + 延迟基准
# 用途：回应审稿意见 P7-3（3 种子重跑单项消融）、P2/P4（逐种子逐类证据）、§4.5（延迟测量协议与主机信息）
# 运行：cd /root/autodl-tmp/Underwater && nohup python scripts/p7_3_ablation.py > p7_3.log 2>&1 &
# 预计：生成变体图 ~30-45 min；训练 6 个模型 × ~2 h ≈ 12 h；评测 ~30 min。可中断续跑（已存在的权重会跳过）。
import json
import os
import sys
import time
import platform
from pathlib import Path
from concurrent import futures as cf

import numpy as np
import cv2

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")
NAMES = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}
SEEDS = [0, 1, 2]
NEW_SEEDS = [1, 2]                     # seed 0 已有（B1_wb / B1_clahe / B1_sat）
OPS = ["wb", "clahe", "sat"]

sys.path.insert(0, str(BASE / "scripts"))
from uep import (estimate_cast, estimate_contrast, estimate_saturation,
                 adaptive_white_balance, adaptive_clahe, adaptive_saturate)

CFG = dict(epochs=150, batch=8, imgsz=640, pretrained=False, patience=0,
           device=0, workers=8, optimizer="AdamW", lr0=0.00125,
           mixup=0.0, copy_paste=0.0, exist_ok=True)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ---------------- 1. 变体图像生成 ----------------
def wb_only(img):
    return adaptive_white_balance(img, estimate_cast(img))


def clahe_only(img):
    return adaptive_clahe(img, estimate_contrast(img))


def sat_only(img):
    return adaptive_saturate(img, estimate_saturation(img))


OPFUNC = {"wb": wb_only, "clahe": clahe_only, "sat": sat_only}


def _worker(args):
    src, dst, op = args
    img = cv2.imread(str(src))
    if img is None:
        return str(src)
    cv2.imwrite(str(dst), OPFUNC[op](img))
    return None


def batch_enhance(op, src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.jpg"))
    todo = [(f, dst_dir / f.name, op) for f in files if not (dst_dir / f.name).exists()]
    if not todo:
        log(f"[gen] {dst_dir.name}: 已存在 {len(files)} 张，跳过")
        return
    n = max(1, (os.cpu_count() or 4) - 1)
    bad = 0
    with cf.ProcessPoolExecutor(max_workers=n) as ex:
        for i, err in enumerate(ex.map(_worker, todo, chunksize=16)):
            if err:
                bad += 1
            if (i + 1) % 2000 == 0:
                log(f"  [gen] {dst_dir.name} {i+1}/{len(todo)}")
    log(f"[gen] {dst_dir.name}: 新生成 {len(todo)} 张，失败 {bad}")


def prepare(op):
    batch_enhance(op, DATA / "images/train", DATA / f"images/train_{op}")
    batch_enhance(op, DATA / "images/test", DATA / f"images/test_{op}")
    for split in ("train", "test"):
        src_lab = DATA / f"labels/{split}"
        dst_lab = DATA / f"labels/{split}_{op}"
        if not dst_lab.exists():
            dst_lab.symlink_to(src_lab, target_is_directory=True)
    y = DATA / f"DUO_{op}.yaml"
    if not y.exists():
        y.write_text(f"path: {DATA.as_posix()}\ntrain: images/train_{op}\nval: images/test_{op}\n"
                     f"test: images/test_{op}\nnames:\n  0: holothurian\n  1: echinus\n"
                     f"  2: scallop\n  3: starfish\n", encoding="utf-8")
        log(f"[yaml] {y.name} 已生成")


# ---------------- 2. 训练 ----------------
def train(name, data_yaml, seed):
    w = BASE / "runs/detect" / name / "weights" / "best.pt"
    if w.exists():
        log(f"[train] {name}: 权重已存在，跳过")
        return
    from ultralytics import YOLO
    log(f"[train] {name} seed={seed} data={data_yaml} (150 epochs, ~2h)")
    YOLO("yolov8n.yaml").train(data=str(DATA / data_yaml), name=name, seed=seed, **CFG)
    log(f"[train] {name} 完成")


# ------------- 2b. 自定义注意力模块注册（B2 权重必需，否则 torch.load 报 No module named 'attention'）-------------
import sys as _sys
for _p in [BASE, BASE / "ultralytics_patch", BASE / "scripts"]:
    if (_p / "attention.py").exists():
        _sys.path.insert(0, str(_p))
        break
try:
    from attention import register_attention as _ra
    _ra()
    print("attention 模块已注册", flush=True)
except Exception as _e:
    print(f"!! attention 注册失败（B2 配置会失败）: {_e}", flush=True)


# ---------------- 3. 逐种子逐类评测 ----------------
def val_full(weights, data_yaml):
    from ultralytics import YOLO
    try:
        r = YOLO(str(weights)).val(data=str(DATA / data_yaml), imgsz=640, verbose=False)
    except Exception as e:
        print(f"  !! 评测失败 {weights}: {type(e).__name__} {e}", flush=True)
        return None, None, None
    per = {int(i): float(a) for i, a in zip(r.box.ap_class_index, r.box.ap50)}
    return (float(r.box.map50), float(r.box.map),
            {NAMES[c]: round(per.get(c, 0.0), 4) for c in NAMES})


def ckpt_of(tag, s):
    name = {"B0": "B0_v2" if s == 0 else f"B0_s{s}",
            "B1ws": "B1ws" if s == 0 else f"B1ws_s{s}",
            "B2ws": "B2ws" if s == 0 else f"B2ws_s{s}"}.get(tag)
    if name is None:                       # 单项消融：B1_wb / B1_clahe / B1_sat
        name = f"B1_{tag}" if s == 0 else f"B1_{tag}_s{s}"
    return BASE / "runs/detect" / name / "weights" / "best.pt"


def evaluate_all():
    """对 12 个配置（raw / UEP / UEP+LCA / 三种单项消融）× 3 种子做逐类评测，落 JSON。"""
    out_dir = BASE / "results_dump"
    out_dir.mkdir(exist_ok=True)
    plan = [("raw(B0)", "B0", "DUO.yaml"),
            ("UEP-wb_sat(B1)", "B1ws", "DUO_wb_sat.yaml")]
    for op in OPS:
        plan.append((f"only-{op}", op, f"DUO_{op}.yaml"))
    plan.append(("UEP+LCA(B2)", "B2ws", "DUO_wb_sat.yaml"))

    rows = []
    for label, tag, yaml in plan:
        for s in SEEDS:
            c = ckpt_of(tag, s)
            if not c.exists():
                log(f"[eval] 缺权重，跳过: {c}")
                continue
            m50, m95, per = val_full(c, yaml)
            if m50 is None:
                continue
            rec = {"model": label, "tag": tag, "seed": s, "yaml": yaml,
                   "mAP50": round(m50, 4), "mAP50_95": round(m95, 4), "per_class_AP50": per}
            rows.append(rec)
            (out_dir / f"{tag}_seed{s}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            log(f"[eval] {label:16s} seed{s} mAP50={m50:.4f} mAP50-95={m95:.4f} {per}")

    # 汇总 3 种子均值±样本标准差
    lines = ["# 3-seed summary (mean +- sample sd, ddof=1)", ""]
    for label, tag, _ in plan:
        v = [r for r in rows if r["tag"] == tag]
        if not v:
            lines.append(f"{label:18s} 无数据"); continue
        def _sd(a):
            return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
        m50 = np.array([r["mAP50"] for r in v]); m95 = np.array([r["mAP50_95"] for r in v])
        lines.append(f"{label:18s} n={len(v)}  mAP50={m50.mean():.4f}+-{_sd(m50):.4f}  "
                     f"mAP50-95={m95.mean():.4f}+-{_sd(m95):.4f}")
        for cname in NAMES.values():
            a = np.array([r["per_class_AP50"].get(cname, 0.0) for r in v])
            lines.append(f"    {cname:12s} {a.mean():.4f}+-{_sd(a):.4f}")
    txt = "\n".join(lines)
    (BASE / "p7_3_summary.txt").write_text(txt, encoding="utf-8")
    (out_dir / "all_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    log("汇总已写 p7_3_summary.txt 与 results_dump/all_rows.json")
    print(txt, flush=True)


# ---------------- 4. 延迟基准（补齐 §4.5 的测量协议）----------------
def bench():
    import torch
    from ultralytics import YOLO
    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    img = cv2.imread(str(sorted((DATA / "images/test").glob("*.jpg"))[0]))
    img = cv2.resize(img, (640, 640))
    for _ in range(20):            # warm-up
        wb_only(img)
    t0 = time.perf_counter()
    for _ in range(200):
        wb_only(img)
    uep_ms = (time.perf_counter() - t0) / 200 * 1000

    w = BASE / "runs/detect/B1ws/weights/best.pt"
    det_ms = float("nan")
    if w.exists():
        m = YOLO(str(w))
        for _ in range(10):
            m.predict(img, imgsz=640, verbose=False, device=0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            m.predict(img, imgsz=640, verbose=False, device=0)
        torch.cuda.synchronize()
        det_ms = (time.perf_counter() - t0) / 50 * 1000

    import ultralytics
    info = {
        "cpu_model": cpu, "cpu_count": os.cpu_count(),
        "opencv": cv2.__version__, "numpy": np.__version__,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "ultralytics": ultralytics.__version__, "platform": platform.platform(),
        "uep_ms_single_thread_median_of_200": round(uep_ms, 2),
        "detection_ms_median_of_50": None if np.isnan(det_ms) else round(det_ms, 2),
        "protocol": "UEP: 640x640 BGR uint8, 20 warm-up + 200 timed runs, single thread (OpenCV); "
                    "detection: ultralytics predict() on GPU after torch.cuda.synchronize(), 10 warm-up + 50 timed",
    }
    (BASE / "latency_bench.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log("延迟基准: " + json.dumps(info, ensure_ascii=False))


def main():
    log("=== P7-3 三操作消融(3 seeds) + 逐种子逐类 dump + 延迟基准 ===")
    for op in OPS:
        prepare(op)
    for op in OPS:
        for s in NEW_SEEDS:
            train(f"B1_{op}_s{s}", f"DUO_{op}.yaml", s)
    evaluate_all()
    bench()
    log("=== 全部完成 ===")


main()
