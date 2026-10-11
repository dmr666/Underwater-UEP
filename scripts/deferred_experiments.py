# deferred_experiments.py — GPU 待办实验（常数敏感性 M4 / 分级强度 M2），与论文协议完全一致
#
# 用途：租到 GPU 后一条命令跑完，产出可直接写进 §4.3 的数据与统计文本。
# 用法：
#   python scripts/deferred_experiments.py sensitivity --data /root/autodl-tmp/DUO --runs /root/autodl-tmp/Underwater/runs/detect
#   python scripts/deferred_experiments.py graded      --data /root/autodl-tmp/DUO --runs /root/autodl-tmp/Underwater/runs/detect
#   python scripts/deferred_experiments.py report      --results results_deferred
#
# 设计（都是 seed 0 + 与已有 3 种子基线比，噪声带取 B0/B1ws 的 3 种子样本标准差）：
#   sensitivity: 三个手设常数 (wb_max=0.9, alpha_clip=0.25, sat_max=1.35) 各自 ±20%，共 6 个变体。
#   graded:      增强强度 50% 训练 / 100% 测试，及其反向，共 2 个配置（检验"分级失配"）。
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ------------------ UEP 算子（与论文 m4_wb_sat_multiseed.py 一致，常数可调） ------------------
def estimate_cast(img):
    m = np.mean(img, axis=(0, 1)); g = np.mean(m)
    return float(np.max(np.abs(g / np.maximum(m, 1e-6) - 1.0)))


def estimate_saturation(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[..., 1]) / 255.0)


def adaptive_white_balance(img, cast, wb_max=0.9, alpha_clip=0.25):
    m = np.mean(img, axis=(0, 1)); g = np.mean(m)
    ratios = g / np.maximum(m, 1e-6)
    alpha = min(cast / alpha_clip, 1.0)
    gains = 1.0 + (ratios - 1.0) * wb_max * alpha
    return np.clip(img.astype(np.float32) * gains.reshape(1, 1, 3), 0, 255).astype(np.uint8)


def adaptive_saturate(img, sat, sat_min=1.0, sat_max=1.35):
    factor = sat_min + (1.0 - sat) * (sat_max - sat_min)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def uep(img, wb_max=0.9, alpha_clip=0.25, sat_max=1.35):
    out = adaptive_white_balance(img, estimate_cast(img), wb_max, alpha_clip)
    return adaptive_saturate(out, estimate_saturation(img), 1.0, sat_max)


# ------------------ 变体定义 ------------------
def variants_sensitivity():
    """6 个变体：每个常数 ±20%，其余保持论文值。"""
    v = []
    for name, key, base in [("wbmax", "wb_max", 0.9), ("clip", "alpha_clip", 0.25), ("satmax", "sat_max", 1.35)]:
        for sign, tag in ((+1, "plus20"), (-1, "minus20")):
            kw = {"wb_max": 0.9, "alpha_clip": 0.25, "sat_max": 1.35}
            kw[key] = round(base * (1 + 0.2 * sign), 4)
            v.append((f"S_{name}_{tag}", kw))
    return v


def variants_graded():
    """分级强度：(训练常数, 测试常数)。50% = 各强度系数乘 0.5。"""
    full = {"wb_max": 0.9, "alpha_clip": 0.25, "sat_max": 1.35}
    half = {"wb_max": 0.45, "alpha_clip": 0.25, "sat_max": 1.175}
    return [("G_train50_test50", (half, half)),        # 域内一致（50% 强度）
            ("G_train50_test100", (half, full))]       # 失配：训练 50% / 测试 100%


def variant_kwargs(name):
    """敏感性变体：返回 (train_kw, test_kw) 中与 name 匹配的一对常数。"""
    for n, kw in variants_sensitivity():
        if n == name:
            return kw
    raise KeyError(name)


# ------------------ 数据准备 ------------------
def _enhance_one(job):
    """单图增强（worker 用；Windows 下需模块级函数才可 pickle）"""
    src, dst_path, kw = job
    cv2.setNumThreads(1)
    im = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)   # 非 ASCII 路径安全
    if im is None:
        return 0
    ok, buf = cv2.imencode(".jpg", uep(im, **kw))
    if not ok:
        return 0
    buf.tofile(dst_path)
    return 1


def prepare(data: Path, out: Path, name: str, train_kw: dict, test_kw: dict = None, workers: int = 1,
            splits=("train", "test")):
    """生成 images/{train,test}_{name}（可选多进程）+ labels 链接 + DUO_{name}.yaml；splits 可只给 ("test",)"""
    test_kw = test_kw or train_kw
    for split in splits:
        kw = train_kw if split == "train" else test_kw
        src = data / "images" / split
        dst = data / "images" / f"{split}_{name}"
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.jpg"))
        have = len(list(dst.glob("*.jpg"))) if dst.exists() else 0
        if have == len(files) and have > 0:
            log(f"  {name} {split} 已生成 {have} 张，跳过（重跑省时）")
            continue
        jobs = [(str(f), str(dst / f.name), kw) for f in files]
        done = failed = 0
        t0 = time.time()
        if workers > 1:
            from multiprocessing import Pool
            with Pool(workers) as pool:
                for r in pool.imap_unordered(_enhance_one, jobs, chunksize=16):
                    done += r
                    failed += (1 - r)
                    if (done + failed) % 2000 == 0:
                        rate = (done + failed) / max(time.time() - t0, 1e-6)
                        log(f"  {name} {split} {done + failed}/{len(files)}  {rate:.0f} img/s  {time.time() - t0:.0f}s")
        else:
            for j in jobs:
                r = _enhance_one(j)
                done += r
                failed += (1 - r)
                if (done + failed) % 2000 == 0:
                    log(f"  {name} {split} {done + failed}/{len(files)}  {time.time() - t0:.0f}s")
        log(f"  {name} {split} 完成: 成功 {done} / 失败 {failed} / 共 {len(files)}  ({time.time() - t0:.0f}s)")
        lab = data / "labels" / f"{split}_{name}"
        if not lab.exists():
            try:
                os.symlink(data / "labels" / split, lab)
            except (OSError, NotImplementedError):        # Windows / 无权限时退回复制
                shutil.copytree(data / "labels" / split, lab)
    y = data / f"DUO_{name}.yaml"
    y.write_text(f"path: {data}\ntrain: images/train_{name}\nval: images/test_{name}\ntest: images/test_{name}\n"
                 "names:\n  0: holothurian\n  1: echinus\n  2: scallop\n  3: starfish\n", encoding="utf-8")
    return y.name


# ------------------ 训练 / 评测（与论文协议逐项一致） ------------------
CFG = dict(epochs=150, batch=8, imgsz=640, optimizer="AdamW", lr0=0.00125, patience=0,
           deterministic=True, pretrained=False, seed=0, exist_ok=True, plots=True, val=True)


def train(data: Path, runs: Path, name: str, yaml_name: str, seed=0, epochs=None):
    from ultralytics import YOLO
    w = runs / name / "weights" / "best.pt"
    if w.exists():
        log(f"[train] {name} 已存在，跳过")
        return w
    cfg = {**CFG, "seed": seed}
    if epochs is not None:
        cfg["epochs"] = epochs
    log(f"[train] {name} seed={seed} data={yaml_name} ({cfg['epochs']} epochs)")
    YOLO("yolov8n.yaml").train(data=str(data / yaml_name), name=name,
                               project=str(runs), **cfg)   # seed 只在 CFG 展开里传一次
    log(f"[train] {name} 完成")
    return w


def evaluate(data: Path, w: Path, yaml_name: str):
    from ultralytics import YOLO
    r = YOLO(str(w)).val(data=str(data / yaml_name), imgsz=640, verbose=False)
    per = {int(i): float(a) for i, a in zip(r.box.ap_class_index, r.box.ap50)}
    names = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}
    return dict(mAP50=round(float(r.box.map50), 4), mAP50_95=round(float(r.box.map), 4),
                per_class_AP50={names[c]: round(per.get(c, 0.0), 4) for c in names})


def run_experiments(kind: str, data: Path, runs: Path, outdir: Path, workers: int = 1):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    if kind == "sensitivity":
        plan = [(n, kw, kw) for n, kw in variants_sensitivity()]
    else:
        plan = [(n, tr, te) for n, (tr, te) in variants_graded()]
    for name, tr_kw, te_kw in plan:
        w = runs / name / "weights" / "best.pt"
        if w.exists():
            yaml_name = prepare(data, outdir, name, tr_kw, te_kw, workers=workers, splits=("test",))
            log(f"[skip-train] {name} 已有权重，仅重建测试集后评测")
        else:
            yaml_name = prepare(data, outdir, name, tr_kw, te_kw, workers=workers)
            w = train(data, runs, name, yaml_name, seed=0)
        rec = evaluate(data, w, yaml_name)
        rec.update(model=name, train_constants=tr_kw, test_constants=te_kw, yaml=yaml_name)
        (outdir / (name + ".json")).write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        rows.append(rec)
        log(f"[eval] {name} mAP50={rec['mAP50']} mAP50-95={rec['mAP50_95']} {rec['per_class_AP50']}")
    if kind == "graded":
        # 反向失配：复用已有的 100% 论文权重（B1ws seed 0）评 50% 强度的测试集，无需训练
        half = {"wb_max": 0.45, "alpha_clip": 0.25, "sat_max": 1.175}
        full = {"wb_max": 0.9, "alpha_clip": 0.25, "sat_max": 1.35}
        yaml50 = prepare(data, outdir, "test50", half, half, workers=workers)
        w100 = runs / "B1ws" / "weights" / "best.pt"
        if w100.exists():
            rec = evaluate(data, w100, yaml50)
            rec.update(model="G_train100_test50", train_constants=full, test_constants=half,
                       reused_b1ws=True, yaml=yaml50)
            (outdir / "G_train100_test50.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            rows.append(rec)
            log(f"[eval] G_train100_test50 (复用 B1ws) mAP50={rec['mAP50']} {rec['per_class_AP50']}")
        else:
            log("!! 未找到 B1ws 权重，跳过 train100/test50 一格")
    (outdir / ("summary_" + kind + ".json")).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"汇总已写 {outdir}/summary_{kind}.json（{len(rows)} 格）")
    return rows


# ------------------ 报告（含噪声带与结论是否成立的判断） ------------------
def report(resdir: Path):
    import statistics as st
    base_json = resdir / "baseline_per_seed.json"       # 由发布仓的 results_per_seed/results_dump 汇总而来
    band = None
    if base_json.exists():
        d = json.loads(base_json.read_text(encoding="utf-8"))
        m = [x["mAP50"] for x in d["B0"]]
        band = st.stdev(m)
    print("\n=== 敏感性 / 分级强度结果 ===")
    for f in sorted(resdir.glob("summary_*.json")):
        for r in json.loads(f.read_text(encoding="utf-8")):
            print(f"  {r['model']:22s} mAP50={r['mAP50']:.4f}  mAP50-95={r['mAP50_95']:.4f}  "
                  f"holothurian={r['per_class_AP50']['holothurian']:.4f}  常数={r['constants']}")
    if band:
        print(f"\n参考：raw(B0) 三种子样本标准差 = {band:.4f}（论文里的‘种子噪声带’）")
        print("判读：若各变体的 mAP50 与基线之差都在该带内，则可写“结论对三个手设常数的 ±20% 扰动不敏感”。")


def selftest(workers: int):
    """自检：造几张合成图，走 prepare()（含多进程分支），确认图与 yaml 都生成。"""
    import tempfile
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for split, n in (("train", 6), ("test", 3)):
            (d / "images" / split).mkdir(parents=True, exist_ok=True)
            (d / "labels" / split).mkdir(parents=True, exist_ok=True)
            for i in range(n):
                im = (rng.random((640, 640, 3)) * 200 + 20).astype(np.uint8)
                cv2.imencode(".jpg", im)[1].tofile(str(d / "images" / split / f"{split}{i}.jpg"))
                (d / "labels" / split / f"{split}{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        y = prepare(d, d, "SELFTEST", {"wb_max": 0.9, "alpha_clip": 0.25, "sat_max": 1.35}, workers=workers)
        nt = len(list((d / "images" / "train_SELFTEST").glob("*.jpg")))
        nv = len(list((d / "images" / "test_SELFTEST").glob("*.jpg")))
        lab = (d / "labels" / "train_SELFTEST")
        print(f"SELFTEST workers={workers}: 训练图 {nt}/6, 测试图 {nv}/3, yaml {y}, 标签{'OK' if lab.exists() else 'MISSING'}")
        assert (nt, nv) == (6, 3), "图像生成数量不对"
        assert lab.exists(), "标签目录未建立"
        # 单图算子检查（有区分度的输入）
        base = cv2.imdecode(np.fromfile(str(d / "images" / "train" / "train0.jpg"), dtype=np.uint8), cv2.IMREAD_COLOR)
        cast_im = base.copy()
        cast_im[..., 0] = np.clip(cast_im[..., 0].astype(np.float32) * 0.82, 0, 255).astype(np.uint8)
        a1 = uep(cast_im, wb_max=0.9, alpha_clip=0.25, sat_max=1.35)
        a2 = uep(cast_im, wb_max=0.9, alpha_clip=0.20, sat_max=1.35)
        assert not np.array_equal(a1, a2), "alpha_clip 扰动未生效"
        print("SELFTEST 通过：多进程生成 + 算子扰动均可用")


def smoketrain(data: Path, runs: Path):
    """真·预检：用 1 个 epoch 跑通 train()+evaluate() 全链路（约 2 分钟），抓调用点/API 错误。"""
    import shutil as sh
    name = "SMOKE_1EP"
    w = train(data, runs, name, "DUO.yaml", seed=0, epochs=1)
    r = evaluate(data, w, "DUO.yaml")
    print(f"SMOKETRAIN 通过：1 epoch 训练 + 评测可用，mAP50={r['mAP50']}（数值无意义，只看通路）")
    for d in (runs / name,):
        if d.exists():
            sh.rmtree(d, ignore_errors=True)
    log("SMOKETRAIN 已清理临时 run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sensitivity", "graded", "report", "selftest", "smoketrain"])
    ap.add_argument("--data", default="/root/autodl-tmp/DUO")
    ap.add_argument("--runs", default="/root/autodl-tmp/Underwater/runs/detect")
    ap.add_argument("--results", default="results_deferred")
    ap.add_argument("--workers", type=int, default=1, help="图像生成的并行进程数")
    a = ap.parse_args()
    data, runs, out = Path(a.data), Path(a.runs), Path(a.results)
    if a.mode == "smoketrain":
        smoketrain(data, runs)
    elif a.mode == "selftest":
        selftest(a.workers if a.workers > 1 else 4)
    elif a.mode == "report":
        report(out)
    else:
        run_experiments(a.mode, data, runs, out, workers=a.workers)


if __name__ == "__main__":
    main()
