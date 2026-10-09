# 小目标多种子尺度分析: 验证 UEP 的 AP50_s 增益是否 3-seed 稳定
# AutoDL 容器端运行: cd /root/autodl-tmp/Underwater && python scripts/ap_by_size_multiseed.py
# 对 B0/B1 各 3 个 seed 权重各跑一次 COCOeval 尺度分桶, 汇总均值 +- 标准差
# 输出: 终端 + ap_size_multiseed_result.txt
# 目的: scale_analysis_result.txt 的 +10.8% 是 seed 0 单次, 需多种子确认稳健性

import sys
import numpy as np
from pathlib import Path

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")
GT = DATA / "annotations" / "instances_test.json"

# (标签, 数据yaml, 权重目录名) —— seed 0 复用 B0_v2/B1_v3, seed 1/2 为 M1 新训
RUNS = [
    ("B0_s0", "DUO.yaml",     "B0_v2"),
    ("B0_s1", "DUO.yaml",     "B0_s1"),
    ("B0_s2", "DUO.yaml",     "B0_s2"),
    ("B1_s0", "DUO_uep.yaml", "B1_v3"),
    ("B1_s1", "DUO_uep.yaml", "B1_s1"),
    ("B1_s2", "DUO_uep.yaml", "B1_s2"),
]

METRICS = ["AP50_s", "AP50_m", "AP50_l", "APs", "APm", "APl"]


def wpath(name):
    return BASE / "runs" / "detect" / name / "weights" / "best.pt"


def scale_eval(tag, yaml_name, wname):
    # 注册 LCA(对 B0/B1 权重无害, 与 ap_by_size.py 保持一致)
    patch_dir = BASE / "ultralytics_patch"
    if patch_dir.exists() and str(patch_dir) not in sys.path:
        sys.path.insert(0, str(patch_dir))
    try:
        from attention import register_attention
        register_attention()
    except Exception as e:
        print("[warn] LCA 注册跳过:", e)

    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    m = YOLO(str(wpath(wname)))
    r = m.val(data=str(DATA / yaml_name), imgsz=640, save_json=True, name=tag, verbose=False)
    pred_path = Path(r.save_dir) / "predictions.json"
    assert pred_path.exists(), "predictions.json 未生成: %s" % pred_path

    gt = COCO(str(GT))
    pred = gt.loadRes(str(pred_path))

    # 标准 COCO (IoU 0.5:0.95) -> APs/APm/APl
    ev = COCOeval(gt, pred, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats

    # 仅 IoU=0.5 的尺度分桶 -> AP50_s/m/l
    ev2 = COCOeval(gt, pred, "bbox")
    ev2.params.iouThrs = [0.5]
    ev2.evaluate()
    ev2.accumulate()
    prec = ev2.eval["precision"]

    def ap_area(aidx):
        p = prec[0, :, :, aidx, 2]
        p = p[p != -1]
        return float(np.mean(p)) if p.size else 0.0

    s50 = (ap_area(1), ap_area(2), ap_area(3))
    return dict(AP=float(s[0]), AP50=float(s[1]),
                APs=float(s[3]), APm=float(s[4]), APl=float(s[5]),
                AP50_s=s50[0], AP50_m=s50[1], AP50_l=s50[2])


def main():
    rows = {}
    for tag, yaml_name, wname in RUNS:
        wp = wpath(wname)
        if not wp.exists():
            print("[skip] %s 缺权重 %s" % (tag, wp))
            continue
        rr = scale_eval(tag, yaml_name, wname)
        rows[tag] = rr
        print("[done] %s: AP50_s=%.3f AP50_m=%.3f AP50_l=%.3f | APs=%.3f APm=%.3f APl=%.3f" % (
            tag, rr["AP50_s"], rr["AP50_m"], rr["AP50_l"], rr["APs"], rr["APm"], rr["APl"]))

    lines = ["小目标多种子尺度分析 (3-seed, COCO 口径)", ""]
    lines.append("单模型明细:")
    for tag, _, _ in RUNS:
        if tag in rows:
            r = rows[tag]
            lines.append("  %-6s AP50_s=%.3f AP50_m=%.3f AP50_l=%.3f | APs=%.3f APm=%.3f APl=%.3f" % (
                tag, r["AP50_s"], r["AP50_m"], r["AP50_l"], r["APs"], r["APm"], r["APl"]))

    def grp(prefix):
        keys = [k for k in rows if k.startswith(prefix)]
        out = []
        for metric in METRICS:
            vals = [rows[k][metric] for k in keys]
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) >= 2 else 0.0
            out.append("%s=%.3f+-%.3f" % (metric, m, sd))
        return keys, "  ".join(out)

    lines.append("")
    b0k, b0s = grp("B0")
    b1k, b1s = grp("B1")
    lines.append("B0 3-seed: " + b0s)
    lines.append("B1 3-seed: " + b1s)
    lines.append("")
    lines.append("增益 B1-B0 (3-seed 均值差):")
    for metric in METRICS:
        d = float(np.mean([rows[k][metric] for k in b1k]) - np.mean([rows[k][metric] for k in b0k]))
        lines.append("  %s = %+.3f" % (metric, d))
    lines.append("")
    lines.append("判断: AP50_s 增益若稳定在 +0.08 以上, 小目标增益稳健; 若缩到 +-0.02 内, 为单 seed 偶然。")

    out = "\n".join(lines)
    print("\n" + out)
    (BASE / "ap_size_multiseed_result.txt").write_text(out, encoding="utf-8")
    print("\n结果已存: ap_size_multiseed_result.txt")


main()
