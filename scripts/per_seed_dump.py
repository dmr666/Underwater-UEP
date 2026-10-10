# per_seed_dump.py — 逐种子逐类评测 + 复核论文中的 t 检验 + 生成 P2/P4 证据
# 在老实例运行：cd /root/autodl-tmp/Underwater && nohup python scripts/per_seed_dump.py > per_seed.log 2>&1 &
import json
import statistics as st
from pathlib import Path

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")
NAMES = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}

# 配置 -> (权重名模板, 评测用 yaml)
CONFIGS = {
    "raw(B0)":        ("B0_v2", "B0_s1", "B0_s2", ["DUO.yaml"] * 3),
    "UEP-wb_sat(B1)": ("B1ws", "B1ws_s1", "B1ws_s2", ["DUO_wb_sat.yaml"] * 3),
    "UEP+LCA(B2)":    ("B2ws", "B2ws_s1", "B2ws_s2", ["DUO_wb_sat.yaml"] * 3),
    "only-wb":        ("B1_wb", None, None, ["DUO_wb.yaml"]),
    "only-clahe":     ("B1_clahe", None, None, ["DUO_clahe.yaml"]),
    "only-sat":       ("B1_sat", None, None, ["DUO_sat.yaml"]),
}
# 交叉评测：B0 在 wb_sat 测试集上、B1ws 在 raw 测试集上（Table 4 的非对角格）
CROSS = [("raw training (B0) on WB+sat test", ("B0_v2", "B0_s1", "B0_s2"), "DUO_wb_sat.yaml"),
         ("WB+sat training (B1) on raw test", ("B1ws", "B1ws_s1", "B1ws_s2"), "DUO.yaml")]

OUT = BASE / "results_dump"

# B2 权重含自定义注意力模块，评测前必须注册（与 m6_b2ws_multiseed.py 同法）
import sys
for _p in [BASE, BASE / "ultralytics_patch", BASE / "scripts", Path("/root/autodl-tmp")]:
    if (_p / "attention.py").exists():
        sys.path.insert(0, str(_p))
        print("attention.py 位于:", _p, flush=True)
        break
try:
    from attention import register_attention
    register_attention()
    print("attention 模块已注册", flush=True)
except Exception as _e:
    print("!! attention 注册失败（B2 配置将被跳过）:", _e, flush=True)


def val_one(name, yaml):
    from ultralytics import YOLO
    w = BASE / "runs/detect" / name / "weights" / "best.pt"
    if not w.exists():
        return None
    try:
        r = YOLO(str(w)).val(data=str(DATA / yaml), imgsz=640, verbose=False)
    except Exception as e:
        print(f"  !! {name} 评测失败: {type(e).__name__} {e}", flush=True)
        return None
    per = {int(i): float(a) for i, a in zip(r.box.ap_class_index, r.box.ap50)}
    return {"name": name, "yaml": yaml, "mAP50": round(float(r.box.map50), 6),
            "mAP50_95": round(float(r.box.map), 6),
            "per_class_AP50": {NAMES[c]: round(per.get(c, 0.0), 6) for c in NAMES}}


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = st.mean(xs)
    s = st.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def welch(a, b):
    """独立样本 t 检验（Welch），返回 t 与双尾 p（用 scipy）。"""
    from scipy import stats
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def holm(pvals):
    """Holm-Bonferroni 校正。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [None] * m
    prev = 0.0
    for k, i in enumerate(order):
        val = min(1.0, (m - k) * pvals[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


def main():
    OUT.mkdir(exist_ok=True)
    rows = {}
    print("=== 逐种子逐类评测 ===", flush=True)
    for label, (n0, n1, n2, yamls) in CONFIGS.items():
        recs = []
        for i, n in enumerate([n0, n1, n2]):
            if n is None:
                continue
            r = val_one(n, yamls[i] if i < len(yamls) else yamls[-1])
            if r:
                r["seed"] = i
                r["config"] = label
                recs.append(r)
                print(f"{label:16s} seed{i} {n:12s} mAP50={r['mAP50']:.4f} mAP50-95={r['mAP50_95']:.4f} "
                      f"{r['per_class_AP50']}", flush=True)
        rows[label] = recs
        (OUT / (label.replace("/", "_").replace("(", "").replace(")", "") + ".json")).write_text(
            json.dumps(recs, indent=1), encoding="utf-8")

    for label, names, yaml in CROSS:
        recs = []
        for i, n in enumerate(names):
            r = val_one(n, yaml)
            if r:
                r["seed"] = i
                r["config"] = label
                recs.append(r)
                print(f"{label:34s} seed{i} mAP50={r['mAP50']:.4f} mAP50-95={r['mAP50_95']:.4f}", flush=True)
        rows[label] = recs

    (OUT / "all_rows.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    # ---- 汇总表 ----
    L = ["# 逐种子 / 3 种子汇总（样本标准差 ddof=1）", ""]
    for label, recs in rows.items():
        if not recs:
            continue
        m50, s50 = mean_sd([r["mAP50"] for r in recs])
        m95, s95 = mean_sd([r["mAP50_95"] for r in recs])
        L.append(f"{label:34s} n={len(recs)}  mAP50={m50:.4f}±{s50:.4f}  mAP50-95={m95:.4f}±{s95:.4f}")
        per = {}
        for c in NAMES.values():
            mm, ss = mean_sd([r["per_class_AP50"][c] for r in recs])
            per[c] = (mm, ss)
            L.append(f"      {c:12s} {mm:.4f}±{ss:.4f}")
        L.append("")

    # ---- 复核论文中的 t 检验 ----
    L.append("# 论文关键统计量的复算（Welch，独立样本，n=3）")
    try:
        b0 = rows["raw(B0)"]
        b1 = rows["UEP-wb_sat(B1)"]
        b2 = rows["UEP+LCA(B2)"]
        for metric in ("mAP50", "mAP50_95"):
            a = [r[metric] for r in b0]
            b = [r[metric] for r in b1]
            t, p = welch(a, b)
            L.append(f"  B1 vs B0 {metric:9s}: t={t:.2f} p={p:.3f} (论文: mAP50 t=0.21 p=0.84; mAP50-95 t=1.73 p=0.16)")
        a = [r["mAP50"] for r in b1]
        b = [r["mAP50"] for r in b2]
        t, p = welch(a, b)
        L.append(f"  B2 vs B1 mAP50    : t={t:.2f} p={p:.3f} (论文: t=3.53 p=0.024)")
        t, p = welch([r["mAP50_95"] for r in b1], [r["mAP50_95"] for r in b2])
        L.append(f"  B2 vs B1 mAP50-95 : t={t:.2f} p={p:.3f} (论文: p=0.088)")
        ps = []
        for c in NAMES.values():
            t, p = welch([r["per_class_AP50"][c] for r in b1], [r["per_class_AP50"][c] for r in b2])
            ps.append(p)
            L.append(f"  B2 vs B1 {c:12s}: diff={st.mean([r['per_class_AP50'][c] for r in b2]) - st.mean([r['per_class_AP50'][c] for r in b1]):+.4f} t={t:.2f} p={p:.4f}")
        adj = holm(ps)
        L.append("  Holm 校正后逐类 p: " + ", ".join(f"{c}={a:.4f}" for c, a in zip(NAMES.values(), adj)))
        L.append("  逐种子增益 B2-B1 (mAP50): " +
                 ", ".join(f"{b2[i]['mAP50'] - b1[i]['mAP50']:+.4f}" for i in range(min(len(b1), len(b2)))))
    except Exception as e:
        L.append("  [复算失败] " + repr(e))

    txt = "\n".join(L)
    (BASE / "per_seed_summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt, flush=True)
    print("\n结果目录:", OUT, flush=True)


main()
