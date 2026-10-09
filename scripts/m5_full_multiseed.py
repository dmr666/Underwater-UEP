# m5_full_multiseed.py — 3-seed per-class AP50 + 3-seed 2x2 cross-eval (final, 2026-08-30)
# AutoDL: cd /root/autodl-tmp/Underwater && python scripts/m5_full_multiseed.py
# 只做 val,不重训。输出: m5_perclass_result.txt + m5_cross_result.txt
import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")
NAMES = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}
SEEDS = [0, 1, 2]


def ckpt(tag, s):
    if tag == "B0":
        name = "B0_v2" if s == 0 else "B0_s%d" % s
    else:
        name = "B1ws" if s == 0 else "B1ws_s%d" % s
    return BASE / "runs/detect" / name / "weights" / "best.pt"


def val_full(c, yaml):
    m = YOLO(str(c))
    r = m.val(data=str(DATA / yaml), imgsz=640, verbose=False)
    ap50 = list(r.box.ap50)
    idx = list(r.box.ap_class_index)
    per = {i: a for i, a in zip(idx, ap50)}
    return float(r.box.map50), float(r.box.map), per


def ms(a):
    x = np.array(a, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def main():
    rows = []
    for s in SEEDS:
        for tag in ("B0", "B1ws"):
            c = ckpt(tag, s)
            if not c.exists():
                print("MISSING %s seed %d: %s" % (tag, s, c), flush=True)
                continue
            for yaml, dtag in (("DUO.yaml", "raw"), ("DUO_wb_sat.yaml", "wb_sat")):
                m50, m95, per = val_full(c, yaml)
                rows.append((tag, s, dtag, m50, m95, per))
                print("%s seed%d on %-6s : mAP50=%.4f mAP50-95=%.4f" % (tag, s, dtag, m50, m95), flush=True)

    # ---- per-class 3-seed (对角协议: B0 on raw, B1ws on wb_sat) ----
    L = ["m5 per-class AP50 (3-seed mean +- std, diagonal protocol)", ""]
    L.append("  class            B0(raw)                B1ws(wb_sat)             diff(mean)")
    for c in range(4):
        b0 = [r[5][c] for r in rows if r[0] == "B0" and r[2] == "raw"]
        b1 = [r[5][c] for r in rows if r[0] == "B1ws" and r[2] == "wb_sat"]
        if b0 and b1:
            m0, s0 = ms(b0); m1, s1 = ms(b1)
            L.append("  %-12s  %.4f +- %.4f    %.4f +- %.4f    %+.4f" % (NAMES[c], m0, s0, m1, s1, m1 - m0))
    out = "\n".join(L)
    print("\n" + out, flush=True)
    (BASE / "m5_perclass_result.txt").write_text(out, encoding="utf-8")

    # ---- 2x2 交叉 3-seed 均值 ----
    L = ["m5 cross-eval 2x2 (3-seed mean +- std)  mAP50 / mAP50-95", ""]
    L.append("training             raw test                 wb_sat test")
    for tag, dtrain in (("B0", "raw"), ("B1ws", "wb_sat")):
        cells = []
        for dtest in ("raw", "wb_sat"):
            m50 = [r[3] for r in rows if r[0] == tag and r[2] == dtest]
            m95 = [r[4] for r in rows if r[0] == tag and r[2] == dtest]
            if m50:
                a, b = ms(m50); c, d = ms(m95)
                cells.append("%.4f +- %.4f / %.4f +- %.4f" % (a, b, c, d))
        L.append("%-18s  %s   %s" % (tag + "(" + dtrain + ")", cells[0], cells[1]))
    out = "\n".join(L)
    print("\n" + out, flush=True)
    (BASE / "m5_cross_result.txt").write_text(out, encoding="utf-8")
    print("Saved: m5_perclass_result.txt  m5_cross_result.txt", flush=True)


main()
