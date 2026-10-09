"""
M1 多种子重训: 验证 +0.8% 增益是否超过随机波动 (AutoDL 容器端运行)

背景: 论文核心结果 B0(raw)=0.805 -> B1(UEP)=0.813 是单次训练 (seed 0),
      审稿人质疑 +0.8 mAP50 是否只是随机种子波动。

方案: B0 与 B1 各补 seed 1、2 (共 4 次训练), 与已有 seed 0 合并,
      报告 3-seed 均值 +- 标准差。若 B1 均值稳定领先 B0, 结论站得住。

训练超参与论文完全一致 (写死):
  epochs=150 / batch=8 / imgsz=640 / pretrained=False / patience=0 /
  optimizer=AdamW / lr0=0.00125 / mixup=0 / copy_paste=0 / device=0 / workers=8

用法:
  cd /root/autodl-tmp/Underwater && python scripts/m1_multiseed.py
输出: 终端 + m1_result.txt
"""
import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")

CFG = dict(epochs=150, batch=8, imgsz=640, pretrained=False, patience=0,
           device=0, workers=8, optimizer="AdamW", lr0=0.00125,
           mixup=0.0, copy_paste=0.0, exist_ok=False)

RUNS = [
    ("B0", "DUO.yaml",     "B0_v2"),   # raw 训练/测试
    ("B1", "DUO_uep.yaml", "B1_v3"),   # UEP 训练/测试
]
SEEDS = [0, 1, 2]


def ensure_yamls():
    for name, train_dir, val_dir in [
        ("DUO.yaml",     "images/train",     "images/test"),
        ("DUO_uep.yaml", "images/train_uep", "images/test_uep"),
    ]:
        y = DATA / name
        y.write_text(
            f"path: {DATA.as_posix()}\n"
            f"train: {train_dir}\n"
            f"val: {val_dir}\n"
            f"test: {val_dir}\n"
            "names:\n"
            "  0: holothurian\n"
            "  1: echinus\n"
            "  2: scallop\n"
            "  3: starfish\n", encoding="utf-8")


def wpath(name):
    return BASE / "runs/detect" / name / "weights" / "best.pt"


def train(name, data_yaml, seed):
    print(f"[train] {name}  seed={seed}  data={data_yaml}  ...", flush=True)
    model = YOLO("yolov8n.yaml")
    model.train(data=str(DATA / data_yaml), name=name, seed=seed, **CFG)
    print(f"[train] {name} 完成", flush=True)


def val(name, data_yaml):
    m = YOLO(str(wpath(name)))
    r = m.val(data=str(DATA / data_yaml), imgsz=640, verbose=False)
    return round(float(r.box.map50), 4), round(float(r.box.map), 4)


def main():
    ensure_yamls()
    lines = ["M1 多种子重训: 3-seed 均值 +- 标准差 (2026-08-27)", ""]
    summary = []
    for tag, data_yaml, seed0_name in RUNS:
        vals = []
        for seed in SEEDS:
            name = seed0_name if seed == 0 else f"{tag}_s{seed}"
            if not wpath(name).exists():
                train(name, data_yaml, seed)
            m50, m95 = val(name, data_yaml)
            vals.append((seed, m50, m95))
            print(f"[val] {name} seed={seed}: mAP50={m50}  mAP50-95={m95}", flush=True)
            lines.append(f"{tag} seed{seed}: mAP50={m50}  mAP50-95={m95}")
        m50s = np.array([v[1] for v in vals])
        m95s = np.array([v[2] for v in vals])
        summary.append((tag, m50s.mean(), m50s.std(), m95s.mean(), m95s.std()))

    lines.append("")
    lines.append("=== 汇总 (3-seed mean +- std) ===")
    for tag, m50m, m50s, m95m, m95s in summary:
        lines.append(f"{tag}: mAP50 = {m50m:.4f} +- {m50s:.4f}   mAP50-95 = {m95m:.4f} +- {m95s:.4f}")
    lines.append("")
    lines.append(f"增益 B1-B0 = {summary[1][1] - summary[0][1]:+.4f} mAP50")

    out = "\n".join(lines)
    print("\n" + out)
    (BASE / "m1_result.txt").write_text(out, encoding="utf-8")
    print("\n结果已存: m1_result.txt")


if __name__ == "__main__":
    main()
