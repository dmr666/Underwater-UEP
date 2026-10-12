# -*- coding: utf-8 -*-
"""gate_prepare.py — 长跑前的最后一道闸：生成缺失的图像域，并用论文已有权重复核已知数值。

原理：增强是逐图确定性算子，只要源图相同，重新生成的域与论文当时用的逐位一致。
      因此用「论文的权重 + 重新生成的域」应当复现出论文的数值；复现不出就说明域生成
      与论文不同，此时**禁止**开长跑（否则 16 GPU-h 全废）。

期望值（论文单种子 seed0 数字）：
      B1ws/best.pt            on DUO_uep.yaml              → mAP50 ≈ 0.8064
      G_train50_test50/best.pt on DUO_G_train50_test50.yaml → mAP50 ≈ 0.8183
"""
import json
import sys
from pathlib import Path

import deferred_experiments as de
import suppl_m4_m6 as S

DATA = Path("/root/autodl-tmp/DUO")
RUNS = Path("/root/autodl-tmp/Underwater/runs/detect")
OUT = Path("/root/autodl-tmp/results_deferred")
TOL = 0.004          # 允许的偏差（同域同权重应几乎完全相同；cuDNN 非确定性给一点余量）


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) 生成缺失域（已存在会自动跳过）
    de.log("生成全强度增强域 uep（train_uep / test_uep）…")
    de.prepare(DATA, OUT, "uep", S.FULL, S.FULL, workers=32)
    for name, kw in (("G_train50_test50", S.HALF), ("G_train50_test100", S.HALF)):
        de.log(f"生成半强度域 {name} …")
        te = S.HALF if name.endswith("test50") else S.FULL
        de.prepare(DATA, OUT, name, kw, te, workers=32)

    # 2) 端到端复核（用论文权重 + 新生成的域）
    checks = [
        ("B1ws", "DUO_uep.yaml", "results_per_seed/per_seed_dump/B1ws_seed0.json", 0.8064),
        ("G_train50_test50", "DUO_G_train50_test50.yaml",
         "results_deferred/G_train50_test50.json", 0.8183),
    ]
    print("\n================ 闸门：域生成复核 ================", flush=True)
    ok = True
    for run, yaml_name, ref_json, ref_default in checks:
        w = RUNS / run / "weights" / "best.pt"
        if not w.exists():
            print(f"  ✗ 缺少权重 {w}", flush=True)
            ok = False
            continue
        r = de.evaluate(DATA, w, yaml_name)
        ref = ref_default
        p = Path("/root/autodl-tmp") / ref_json
        if p.exists():
            try:
                ref = float(json.loads(p.read_text(encoding="utf-8"))["mAP50"])
            except Exception:
                pass
        d = abs(r["mAP50"] - ref)
        flag = "✓" if d <= TOL else "✗"
        if d > TOL:
            ok = False
        print(f"  {flag} {run:20s} on {yaml_name:26s} mAP50={r['mAP50']:.4f}  参考={ref:.4f}  差={d:.4f}", flush=True)
        print(f"      逐类: {r['per_class_AP50']}", flush=True)

    print("=================================================", flush=True)
    if ok:
        print("闸门通过：新生成的图像域与论文当时使用的一致，可以放长跑。", flush=True)
        sys.exit(0)
    print("!! 闸门未通过：域生成与论文不一致，先排查，禁止开长跑。", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
