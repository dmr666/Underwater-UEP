import sys
import cv2
from pathlib import Path
from ultralytics import YOLO

BASE = Path("/root/autodl-tmp/Underwater")
DATA = Path("/root/autodl-tmp/DUO")
sys.path.insert(0, str(BASE / "scripts"))
from uep import estimate_cast, adaptive_white_balance, estimate_saturation, adaptive_saturate


def wb_sat(img):
    cast = estimate_cast(img)
    sat = estimate_saturation(img)
    out = adaptive_white_balance(img, cast)
    out = adaptive_saturate(out, sat)
    return out


def main():
    names = ["29", "132", "1025"]
    # Fig 2: WB+sat 增强图
    for n in names:
        src = DATA / "images/test" / (n + ".jpg")
        if not src.exists():
            print("missing", src, flush=True)
            continue
        img = cv2.imread(str(src))
        out = wb_sat(img)
        dst = BASE / ("wb_sat_" + n + ".jpg")
        cv2.imwrite(str(dst), out)
        print("saved", dst.name, flush=True)

    # Fig 3: B1ws 检测结果图(在 wb_sat 增强图上推理)
    model = YOLO(str(BASE / "runs/detect/B1ws/weights/best.pt"))
    for n in names:
        enh = BASE / ("wb_sat_" + n + ".jpg")
        if not enh.exists():
            continue
        model.predict(str(enh), imgsz=640, conf=0.25, save=True,
                      project=str(BASE / "viz_ws"), name=n, exist_ok=True)
        print("detected", n, flush=True)

    print("all done", flush=True)


main()
