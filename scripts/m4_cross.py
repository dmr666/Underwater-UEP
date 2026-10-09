import sys
from pathlib import Path
from ultralytics import YOLO

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")


def val(weights, data_yaml):
    m = YOLO(str(weights))
    r = m.val(data=str(DATA / data_yaml), imgsz=640, verbose=False)
    return round(float(r.box.map50), 4), round(float(r.box.map), 4)


def main():
    b0_raw = val(BASE / "runs/detect/B0_v2/weights/best.pt", "DUO.yaml")
    b0_ws = val(BASE / "runs/detect/B0_v2/weights/best.pt", "DUO_wb_sat.yaml")
    b1_raw = val(BASE / "runs/detect/B1ws/weights/best.pt", "DUO.yaml")
    b1_ws = val(BASE / "runs/detect/B1ws/weights/best.pt", "DUO_wb_sat.yaml")

    lines = ["WB+sat cross-eval (2x2)  mAP50 / mAP50-95", ""]
    lines.append("training        raw test            wb_sat test")
    lines.append("B0 raw          %.4f / %.4f   %.4f / %.4f" % (b0_raw[0], b0_raw[1], b0_ws[0], b0_ws[1]))
    lines.append("B1ws wb_sat     %.4f / %.4f   %.4f / %.4f" % (b1_raw[0], b1_raw[1], b1_ws[0], b1_ws[1]))

    out = "\n".join(lines)
    print("\n" + out, flush=True)
    (BASE / "m4_cross_result.txt").write_text(out, encoding="utf-8")
    print("\nSaved: m4_cross_result.txt", flush=True)


main()
