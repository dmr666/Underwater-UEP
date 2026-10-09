# B1ws (WB+sat) per-class AP50, compare against raw (B0_v2)
# AutoDL: cd /root/autodl-tmp/Underwater && python scripts/m4_perclass.py
import sys
from pathlib import Path
from ultralytics import YOLO

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")
NAMES = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}


def perclass(weights, data_yaml, tag):
    m = YOLO(str(weights))
    r = m.val(data=str(DATA / data_yaml), imgsz=640, verbose=False)
    ap50 = list(r.box.ap50)
    idx = list(r.box.ap_class_index)
    d = {i: a for i, a in zip(idx, ap50)}
    print(tag + "  mAP50=%.4f" % float(r.box.map50), flush=True)
    for c in range(4):
        print("  %s : %.4f" % (NAMES[c], d.get(c, 0.0)), flush=True)
    return d


def main():
    d_ws = perclass(BASE / "runs/detect/B1ws/weights/best.pt", "DUO_wb_sat.yaml", "B1ws WB+sat")
    d_raw = perclass(BASE / "runs/detect/B0_v2/weights/best.pt", "DUO.yaml", "B0 raw")
    lines = ["per-class AP50 (raw vs B1ws WB+sat)", ""]
    lines.append("  class           raw       B1ws      diff")
    for c in range(4):
        a = d_raw.get(c, 0.0)
        b = d_ws.get(c, 0.0)
        lines.append("  %s   %.4f   %.4f   %+.4f" % (NAMES[c].ljust(12), a, b, b - a))
    out = "\n".join(lines)
    print("\n" + out, flush=True)
    (BASE / "m4_perclass_result.txt").write_text(out, encoding="utf-8")
    print("\nSaved: m4_perclass_result.txt", flush=True)


main()
