# -*- coding: utf-8 -*-
"""suppl_m4_m6.py — 审稿人点名的三组补实验（M4 补种子 / M6a 预训练微调 / M6b RUOD 交叉评估）

设计原则：与论文协议逐项一致；复用 deferred_experiments.py 里已跑通的算子与数据生成；
每个实验独立落盘（断线重跑自动跳过已完成的 run），跑完直接产出可写进回复信的数字。

用法（实例上，工作目录 /root/autodl-tmp/Underwater）：
  python scripts/suppl_m4_m6.py audit                       # 只需 CPU：审计配置是否与论文一致
  python scripts/suppl_m4_m6.py selftest --workers 8        # 只需 CPU：造图+算子扰动自检
  python scripts/suppl_m4_m6.py smoketrain                  # 需 GPU：真跑 1 epoch 抓调用点错误
  python scripts/suppl_m4_m6.py m4   --workers 32           # 4 次训练 ≈7.5h：half×half / half×full 各补 2 种子
  python scripts/suppl_m4_m6.py m6a  --workers 32           # 2 次训练 ≈4h：COCO 预训练 + 微调，四条一致性臂
  python scripts/suppl_m4_m6.py m6b  --ruod /root/autodl-tmp/RUOD --workers 32   # 2 次训练 ≈4h：RUOD 迁移
  python scripts/suppl_m4_m6.py report                      # 汇总成回复信可用的一页数字

注意：本脚本的 UEP 算子直接 import deferred_experiments（其 estimate_cast 与论文实验代码
scripts/uep.py::estimate_cast 完全一致：c = max_ch |mean_gray/mean_ch - 1|），不要另写一份。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import deferred_experiments as de          # 该模块有 __main__ 守卫，import 不会触发 main()

HALF = de.variants_graded()[0][1][0]      # {'wb_max':0.45,'alpha_clip':0.25,'sat_max':1.175}
FULL = de.variants_graded()[1][1][1]      # {'wb_max':0.90,'alpha_clip':0.25,'sat_max':1.35}  ← 取第二变体的“测试端常数”
assert FULL == {'wb_max': 0.9, 'alpha_clip': 0.25, 'sat_max': 1.35}, f'FULL 取错: {FULL}'
assert HALF == {'wb_max': 0.45, 'alpha_clip': 0.25, 'sat_max': 1.175}, f'HALF 取错: {HALF}'
ARMS = ["raw", "uep"]                     # 训练域
TESTS = ["raw", "uep"]                    # 测试域


# ---------------------------------------------------------------- 审计（CPU）
PAPER_CFG = dict(epochs=150, batch=8, imgsz=640, optimizer="AdamW", lr0=0.00125, patience=0,
                 deterministic=True, pretrained=False)
PAPER_HALF = dict(wb_max=0.45, alpha_clip=0.25, sat_max=1.175)


def audit(runs: Path):
    """把本脚本将要使用的配置，与论文已归档的 args.yaml 逐字段比对（不一致就打出来）。"""
    import yaml
    bad = 0
    print("=== 1) 训练超参 vs 论文归档 args.yaml ===")
    refs = sorted(runs.glob("B1ws*/args.yaml")) + sorted(runs.glob("B0_s1/args.yaml"))
    if not refs:
        print("  !! 未找到归档 args.yaml，跳过该项（请确认 --runs 指向论文的 runs/detect）")
    for ref in refs[:4]:
        a = yaml.safe_load(ref.read_text(encoding="utf-8"))
        print(f"  [{ref.parent.name}]  ← 归档 args.yaml")
        for k, v in PAPER_CFG.items():
            got = a.get(k)
            same = (got == v) or (k == "pretrained" and got in (False, None, "False"))
            print(f"     {'✓' if same else '✗'} {k:14s} 本脚本={v!r:12} 归档={got!r}")
            bad += (not same)
        for k in ("seed", "device", "workers"):
            if k in a:
                print(f"       · {k} = {a[k]!r}（允许与论文不同：seed 由变体决定、device/workers 是本机设置）")
    print("\n=== 2) UEP 算子常数 vs 论文 ===")
    for k, v in PAPER_HALF.items():
        bad_c = 0
    for k, v in PAPER_HALF.items():
        ok = abs(HALF[k] - v) < 1e-9
        bad_c += (not ok)
        print(f"  half  {k:11s} = {HALF[k]}  (期望 {v}) → {'✓' if ok else '✗'}")
    for k, v in {"wb_max": 0.9, "alpha_clip": 0.25, "sat_max": 1.35}.items():
        ok = abs(FULL[k] - v) < 1e-9
        bad_c += (not ok)
        print(f"  full  {k:11s} = {FULL[k]}  (期望 {v}) → {'✓' if ok else '✗'}")
    assert bad_c == 0, "强度常数与论文不一致，禁止开跑"
    bad += bad_c
    print("\n=== 3) 色偏定义核对（必须与论文实验代码一致）===")
    rng = np.random.default_rng(0)
    im = (rng.random((64, 64, 3)) * 180 + 40).astype(np.uint8)
    im[..., 0] = np.clip(im[..., 0].astype(np.float32) * 0.72, 0, 255).astype(np.uint8)   # 明确偏色
    m = np.mean(im, axis=(0, 1)); g = np.mean(m)
    mine = float(np.max(np.abs(g / np.maximum(m, 1e-6) - 1.0)))
    theirs = float(np.max(np.abs(m / np.maximum(g, 1e-6) - 1.0)))
    uep_val = de.estimate_cast(im)
    print(f"  偏色测试图: 通道均值={np.round(m, 1)}")
    print(f"  deferred_experiments.estimate_cast = {uep_val:.6f}")
    print(f"  代码定义 |mean_gray/mean_ch - 1|    = {mine:.6f}  → {'✓ 一致' if abs(uep_val - mine) < 1e-6 else '✗ 不一致'}")
    print(f"  正文曾写反的 |mean_ch/mean_gray-1|  = {theirs:.6f}  （两者应明显不同；正文已改齐）")
    assert abs(uep_val - mine) < 1e-6 and abs(mine - theirs) > 0.05, "色偏定义核对未通过"
    print(f"\n审计结果：{'通过' if bad == 0 else f'有 {bad} 个字段不一致，请先核对再开跑'}")


# ---------------------------------------------------------------- 训练 / 评测（扩展版）
def train2(data: Path, runs: Path, name: str, yaml_name: str, seed=0, model="yolov8n.yaml",
           pretrained=False, epochs=None, project=None):
    """比 de.train 多两项：model 可选 .pt（预训练微调）、pretrained 可控。落盘即返回。"""
    from ultralytics import YOLO
    w = runs / name / "weights" / "best.pt"
    if w.exists():
        de.log(f"[train] {name} 已存在，跳过")
        return w
    cfg = {**de.CFG, "pretrained": pretrained}
    if seed is not None:
        cfg["seed"] = seed
    if epochs is not None:
        cfg["epochs"] = epochs
    de.log(f"[train] {name} model={model} seed={seed} data={yaml_name} epochs={cfg['epochs']} pretrained={pretrained}")
    YOLO(model).train(data=str(data / yaml_name), name=name, project=str(project or runs), **cfg)
    de.log(f"[train] {name} 完成 → {w}")
    return w


def arms_table(data: Path, outdir: Path, name: str, w_raw: Path, w_uep: Path):
    """四条一致性臂：训练域 {raw,uep} × 测试域 {raw,uep}。"""
    rows = []
    for tr in ARMS:
        w = w_raw if tr == "raw" else w_uep
        for te in TESTS:
            yaml_name = "DUO.yaml" if te == "raw" else "DUO_uep.yaml"
            tag = f"{name}_train{tr}_test{te}"
            jf = outdir / (tag + ".json")
            if jf.exists():
                rec = json.loads(jf.read_text(encoding="utf-8"))
            else:
                rec = de.evaluate(data, w, yaml_name)
                rec.update(arm=tag, train_domain=tr, test_domain=te, weights=str(w), yaml=yaml_name)
                jf.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            rows.append(rec)
            de.log(f"[eval] {tag:34s} mAP50={rec['mAP50']:.4f} mAP50-95={rec['mAP50_95']:.4f}")
    return rows


# ---------------------------------------------------------------- M4：half-strength 2×2 补 3 种子
def m4(data: Path, runs: Path, outdir: Path, workers: int):
    """补 seed1/2：half→half（域内一致）与 half→full（失配）；其余格复用已有权重。"""
    # 数据目录名与论文 seed0 的两次 run 完全同名 → prepare 检测到已生成会直接跳过，
    # 从而保证补种子的训练图/测试图与 seed0 逐图一致（不重新生成=无漂移风险）
    plan = [("G_train50_test50", HALF, HALF), ("G_train50_test100", HALF, FULL)]
    rows = []
    for name, tr, te in plan:
        yaml_name = de.prepare(data, outdir, name, tr, te, workers=workers)
        for seed in (1, 2):
            run_name = f"{name}_s{seed}"
            w = runs / run_name / "weights" / "best.pt"
            if not w.exists():
                w = train2(data, runs, run_name, yaml_name, seed=seed)
            rec = de.evaluate(data, w, yaml_name)
            rec.update(arm=run_name, seed=seed, train_constants=tr, test_constants=te, yaml=yaml_name)
            (outdir / (run_name + ".json")).write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            rows.append(rec)
            de.log(f"[m4] {run_name:26s} mAP50={rec['mAP50']:.4f} holothurian={rec['per_class_AP50']['holothurian']:.4f}")
    (outdir / "summary_m4.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    de.log(f"m4 完成 {len(rows)} 格 → {outdir}/summary_m4.json")


# ---------------------------------------------------------------- M6a：COCO 预训练 + 微调
def m6a(data: Path, runs: Path, outdir: Path, workers: int, weights_pt: str):
    """与论文同协议，仅把 from-scratch 换成 COCO 预训练权重做微调；跑四条一致性臂。"""
    w_raw = runs / "P_raw" / "weights" / "best.pt"
    w_uep = runs / "P_uep" / "weights" / "best.pt"
    # 复用论文现成的两个域（不生成新图）：raw 域 = DUO.yaml，增强域 = DUO_uep.yaml
    if not w_raw.exists():
        w_raw = train2(data, runs, "P_raw", "DUO.yaml", seed=0, model=weights_pt, pretrained=True)
    if not w_uep.exists():
        w_uep = train2(data, runs, "P_uep", "DUO_uep.yaml", seed=0, model=weights_pt, pretrained=True)
    rows = arms_table(data, outdir, "P", w_raw, w_uep)
    (outdir / "summary_m6a.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    de.log(f"m6a 完成 {len(rows)} 臂 → {outdir}/summary_m6a.json")


# ---------------------------------------------------------------- M6b：RUOD 迁移
def ruod_prepare(src: Path, data_out: Path, workers: int):
    """RUOD COCO(4 类与 DUO 同名) → YOLO 目录；再按 UEP 生成增强域。"""
    import shutil
    import cv2
    tr_j, te_j = src / "train.json", src / "test.json"
    if not tr_j.exists():
        cand = sorted(src.rglob("*.json"))                     # 递归：RUOD 的标注在 RUOD_ANN/ 子目录里
        tr_j = next((c for c in cand if "train" in c.name.lower()), None)
        te_j = next((c for c in cand if "test" in c.name.lower()), None)
    print(f"[ruod] 标注: train={tr_j}  test={te_j}", flush=True)
    assert tr_j and tr_j.exists(), f"未找到 RUOD 的 COCO 标注（{src}）"
    pics = next((p for p in [src, src / "RUOD_pic", src / "images"] if p.exists()), src)
    for jf, split in ((tr_j, "train"), (te_j, "test")):
        if jf is None:
            continue
        d = json.loads(jf.read_text(encoding="utf-8"))
        id2img = {im["id"]: im for im in d["images"]}
        img_dir = data_out / "images" / split; lab_dir = data_out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True); lab_dir.mkdir(parents=True, exist_ok=True)
        missing = 0
        for im in d["images"]:
            (lab_dir / (Path(im["file_name"]).stem + ".txt")).write_text("", encoding="utf-8")
        for an in d["annotations"]:
            im = id2img.get(an["image_id"]);  cat = int(an["category_id"])
            if im is None or cat not in (1, 2, 3, 4):
                continue
            x, y, w, h = an["bbox"]; W, H = im["width"], im["height"]
            with open(lab_dir / (Path(im["file_name"]).stem + ".txt"), "a", encoding="utf-8") as f:
                f.write(f"{cat-1} {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}\n")
        for im in d["images"]:
            t = img_dir / im["file_name"]
            if t.exists():
                continue                      # 目标已存在（如测试图已在实例上硬链接）→ 不动
            # 源图可能在多个层次：RUOD_pic/*.jpg、RUOD_pic/{split}/*.jpg、RUOD_pic/RUOD_pic/{split}/*.jpg
            for cand in (pics / im["file_name"], pics / split / im["file_name"],
                         pics / "RUOD_pic" / split / im["file_name"], pics / "RUOD_pic" / im["file_name"]):
                if cand.exists():
                    shutil.copy2(cand, t); break
            else:
                missing += 1
        de.log(f"RUOD {split}: {len(d['images'])} 图 / {len(d['annotations'])} 标注 / 缺图 {missing}")
    (data_out / "RUOD.yaml").write_text(
        f"path: {data_out}\ntrain: images/train\nval: images/test\ntest: images/test\n"
        "names:\n  0: holothurian\n  1: echinus\n  2: scallop\n  3: starfish\n", encoding="utf-8")
    return "RUOD.yaml"


def m6b(data: Path, runs: Path, outdir: Path, workers: int, ruod: Path):
    """在 RUOD 上从零训 raw / UEP 两条，再跑四条一致性臂（最小可用版：单种子，回复信里注明）。"""
    rd = data.parent / "RUOD_yolo"
    yaml_raw = ruod_prepare(ruod, rd, workers) if not (rd / "RUOD.yaml").exists() else "RUOD.yaml"
    y_uep = de.prepare(rd, outdir, "uep", FULL, FULL, workers=workers)
    w_raw = runs / "R_raw" / "weights" / "best.pt"
    if not w_raw.exists():
        w_raw = train2(data=rd, runs=runs, name="R_raw", yaml_name=yaml_raw, seed=0)
    w_uep = runs / "R_uep" / "weights" / "best.pt"
    if not w_uep.exists():
        w_uep = train2(data=rd, runs=runs, name="R_uep", yaml_name=y_uep, seed=0)
    rows = []
    for tr, w in (("raw", w_raw), ("uep", w_uep)):
        for te, y in (("raw", "RUOD.yaml"), ("uep", y_uep)):
            rec = de.evaluate(rd, w, y)
            rec.update(arm=f"RUOD_train{tr}_test{te}", train_domain=tr, test_domain=te, yaml=y)
            (outdir / (rec["arm"] + ".json")).write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            rows.append(rec)
            de.log(f"[m6b] {rec['arm']:28s} mAP50={rec['mAP50']:.4f}")
    (outdir / "summary_m6b.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    de.log(f"m6b 完成 {len(rows)} 臂 → {outdir}/summary_m6b.json")


# ---------------------------------------------------------------- 汇总
def report(outdir: Path):
    import statistics as st
    def load(f):
        p = outdir / f
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    print("\n================ 补实验汇总（回复信用） ================")
    m4 = load("summary_m4.json")
    if m4:
        print("\n[M4] half-strength 2×2 的 3 种子（★=本次新跑）")
        base = load("G_train50_test50.json") or {}
        for cell, rows in (("half→half", [r for r in m4 if "test50" in r["arm"]]),
                           ("half→full", [r for r in m4 if "test100" in r["arm"]])):
            vals = [(r["seed"], r["mAP50"]) for r in rows] + ([(0, base.get("mAP50"))] if base else [])
            vals = [v for v in vals if v[1] is not None]
            if vals:
                ms = [v[1] for v in vals]
                print(f"  {cell}: n={len(ms)} mean={st.mean(ms):.4f} sd={st.stdev(ms) if len(ms) > 1 else float('nan'):.4f}  " +
                      "  ".join(f"s{s}={v:.4f}" for s, v in vals))
        print("  判读：若半强度一致格的 3 种子均值仍显著高于失配格（且 sd 远小于差值），M4 的“★单种子”可去掉。")
    for tag, label in (("summary_m6a.json", "[M6a] COCO 预训练 + 微调"),
                       ("summary_m6b.json", "[M6b] RUOD 迁移")):
        rows = load(tag)
        if not rows:
            continue
        print(f"\n{label}（单种子，回复信需注明）")
        for r in rows:
            print(f"  {r['arm']:28s} mAP50={r['mAP50']:.4f} mAP50-95={r['mAP50_95']:.4f} "
                  f"holothurian={r['per_class_AP50']['holothurian']:.4f}")
        g = {(r['train_domain'], r['test_domain']): r['mAP50'] for r in rows}
        if ("uep", "uep") in g and ("uep", "raw") in g:
            print(f"  一致性惩罚（训 UEP→测原图）：{g[('uep','raw')]-g[('uep','uep')]:+.4f} mAP50"
                  f"（论文 from-scratch 版本为 −0.165）")
        if ("raw", "raw") in g and ("raw", "uep") in g:
            print(f"  纯测试端施加 UEP：{g[('raw','uep')]-g[('raw','raw')]:+.4f} mAP50"
                  f"（论文 from-scratch 版本为 −0.296）")
    print("\n======================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit", "selftest", "smoketrain", "m4", "m6a", "m6b", "report"])
    ap.add_argument("--data", default="/root/autodl-tmp/DUO")
    ap.add_argument("--runs", default="/root/autodl-tmp/Underwater/runs/detect")
    ap.add_argument("--results", default="results_deferred")
    ap.add_argument("--ruod", default="/root/autodl-tmp/RUOD")
    ap.add_argument("--weights-pt", default="yolov8n.pt", help="M6a 用的 COCO 预训练权重")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    data, runs, out, ruod = Path(a.data), Path(a.runs), Path(a.results), Path(a.ruod)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "audit":
        audit(runs)
    elif a.mode == "selftest":
        de.selftest(a.workers if a.workers > 1 else 4)
    elif a.mode == "smoketrain":
        de.smoketrain(data, runs)
    elif a.mode == "m4":
        m4(data, runs, out, a.workers)
    elif a.mode == "m6a":
        m6a(data, runs, out, a.workers, a.weights_pt)
    elif a.mode == "m6b":
        m6b(data, runs, out, a.workers, ruod)
    else:
        report(out)


if __name__ == "__main__":
    main()
