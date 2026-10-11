# repair_dump.py — 补数：只重建各变体的测试集并复评已有 best.pt，落盘 per-variant JSON
# 不写 summary_*.json（留给看门狗的完整重跑，避免第 6 组还没训完就把汇总写死）
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deferred_experiments as de   # 该模块有 __main__ 守卫，import 不会触发 main()

DATA = Path("/root/autodl-tmp/DUO")
RUNS = Path("/root/autodl-tmp/Underwater/runs/detect")
OUT = Path("/root/autodl-tmp/results_deferred")
OUT.mkdir(parents=True, exist_ok=True)

for name, kw in de.variants_sensitivity():
    w = RUNS / name / "weights" / "best.pt"
    if not w.exists():
        print("[skip] %s 尚无权重（训练未完成）" % name, flush=True)
        continue
    if (OUT / (name + ".json")).exists():
        print("[skip] %s JSON 已存在" % name, flush=True)
        continue
    yaml_name = de.prepare(DATA, OUT, name, kw, kw, workers=32, splits=("test",))
    rec = de.evaluate(DATA, w, yaml_name)
    rec.update(model=name, train_constants=kw, test_constants=kw, yaml=yaml_name, repaired=True)
    (OUT / (name + ".json")).write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    print("[eval] %s mAP50=%.4f mAP50-95=%.4f %s" % (name, rec["mAP50"], rec["mAP50_95"], rec["per_class_AP50"]), flush=True)

print("REPAIR DONE", flush=True)
