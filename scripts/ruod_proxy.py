# -*- coding: utf-8 -*-
"""ruod_proxy.py — M6b 的可用版本：跨数据集（RUOD）检验"只在测试端加增强"的惩罚是否同样存在。

背景：本实例上 RUOD 只转换了测试划分（labels/test 4200 张，无 train），
      因此无法做"在 RUOD 上从零训练"的完整版；这里做**代理实验**：
      用 DUO 上训练的模型（raw 域 B0 三种子 / 增强域 B1ws）在 RUOD 测试集上，
      分别按 raw 与 UEP 增强两种输入评估 —— 直接检验"即插即用增强"的失效是否跨数据集复现。

  raw 域模型：回原图 vs 回增强图  → 纯测试端施加增强的惩罚（论文 DUO 上为 −29.6 点）
  增强域模型：回增强图 vs 回原图  → 反向失配惩罚（论文 DUO 上为 −16.5 点）

局限（必须在回复信写明）：训练仍在 DUO 上，属**代理证据**；完整版需 RUOD 训练划分。

用法: cd /root/autodl-tmp/Underwater && python scripts/ruod_proxy.py
"""
import json
import statistics as st
from pathlib import Path

import deferred_experiments as de

RUOD = Path("/root/autodl-tmp/RUOD")
RUNS = Path("/root/autodl-tmp/Underwater/runs/detect")
OUT = Path("/root/autodl-tmp/results_deferred")
RAW_YAML, UEP_YAML = "RUOD_test.yaml", "RUOD_test_uep.yaml"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = [("B0_v2", "raw"), ("B0_s1", "raw"), ("B0_s2", "raw"), ("B1ws", "uep")]
    rows = []
    for run, train_domain in plan:
        w = RUNS / run / "weights" / "best.pt"
        if not w.exists():
            de.log(f"[skip] 缺权重 {run}")
            continue
        for yaml_name in (RAW_YAML, UEP_YAML):
            test_domain = "raw" if "uep" not in yaml_name else "uep"
            jf = OUT / f"RUOD_{run}_test{test_domain}.json"
            if jf.exists():
                rec = json.loads(jf.read_text(encoding="utf-8"))
            else:
                rec = de.evaluate(RUOD, w, yaml_name)
                rec.update(model=run, train_domain=train_domain, test_domain=test_domain,
                           yaml=yaml_name, dataset="RUOD-test4200", note="DUO 训练的跨域代理")
                jf.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
            rows.append(rec)
            de.log(f"[RUOD] {run:7s} train={train_domain:3s} test={test_domain:3s} mAP50={rec['mAP50']:.4f} "
                   f"holothurian={rec['per_class_AP50']['holothurian']:.4f}")
    (OUT / "summary_ruod_proxy.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n================ RUOD 跨域代理结果 ================")
    for tag, label in (("raw", "raw 域模型（B0，三种子）"), ("uep", "增强域模型（B1ws，种子0）")):
        sub = [r for r in rows if r["train_domain"] == tag]
        raw = [r["mAP50"] for r in sub if r["test_domain"] == "raw"]
        uep = [r["mAP50"] for r in sub if r["test_domain"] == "uep"]
        if not raw or not uep:
            continue
        mr, mu = st.mean(raw), st.mean(uep)
        print(f"\n{label}")
        print(f"  回原图   mean mAP50 = {mr:.4f}   (n={len(raw)}: {[round(x,4) for x in raw]})")
        print(f"  回增强图 mean mAP50 = {mu:.4f}   (n={len(uep)}: {[round(x,4) for x in uep]})")
        print(f"  差 = {mu - mr:+.4f} mAP50")
    print("\n对照：论文 DUO 上 纯测试端施加 UEP = -0.296；训 UEP 测原图 = -0.165")
    print("==================================================")


if __name__ == "__main__":
    main()
