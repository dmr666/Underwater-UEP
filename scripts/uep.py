# -*- coding: utf-8 -*-
"""
UEP: Underwater Enhancement Pipeline — 自适应版本 (创新点 C1)

退化感知的自适应增强, 全部基于 OpenCV, 零可训练参数:
  1) 自适应白平衡: 色偏越重 (通道均值比偏离 1 越大), 白平衡增益越强
  2) 自适应 CLAHE:  对比度越低, clip 参数越大
  3) 自适应饱和度:  饱和度越低, 增强系数越大

论文叙事: 三项增强的强度均由图像统计特征 (色偏/对比度/饱和度) 动态驱动,
即"退化感知"的增强管线, 而不是固定参数的经典方法堆叠。

用法:
  单张:  python uep.py in.jpg out.jpg
  批量:  python uep.py --dir in_dir --out out_dir --ext jpg
  训练/推理中 import:  from uep import enhance
"""
import argparse
import os
import cv2
import numpy as np
from pathlib import Path


def estimate_cast(img: np.ndarray) -> float:
    """色偏程度: 通道均值比偏离 1 的最大幅度 (0=无偏, 越大越偏色)。"""
    mean_bgr = np.mean(img, axis=(0, 1))
    gray = np.mean(mean_bgr)
    ratios = gray / np.maximum(mean_bgr, 1e-6)
    return float(np.max(np.abs(ratios - 1.0)))


def estimate_contrast(img: np.ndarray) -> float:
    """对比度: 灰度标准差归一化到 [0,1] (0=极低对比, 1=高对比)。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(min(np.std(g) / 128.0, 1.0))


def estimate_saturation(img: np.ndarray) -> float:
    """饱和度: HSV S 通道均值归一化 [0,1]。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[..., 1]) / 255.0)


def adaptive_white_balance(img: np.ndarray, cast: float, wb_max: float = 0.9) -> np.ndarray:
    """色偏驱动白平衡: 无偏图像不动 (增益=1), 色偏越重增益越接近 wb_max 强度。"""
    mean_bgr = np.mean(img, axis=(0, 1))
    gray = np.mean(mean_bgr)
    ratios = gray / np.maximum(mean_bgr, 1e-6)
    alpha = min(cast / 0.25, 1.0)          # 色偏 0.25 视为"重偏色", 全量校正
    gains = 1.0 + (ratios - 1.0) * wb_max * alpha
    out = img.astype(np.float32) * gains.reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def adaptive_clahe(img: np.ndarray, contrast: float, clip_min: float = 1.5, clip_max: float = 4.5) -> np.ndarray:
    """对比度驱动 CLAHE: 对比度越低 clip 越大。"""
    clip = clip_min + (1.0 - contrast) * (clip_max - clip_min)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    lab2 = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def adaptive_saturate(img: np.ndarray, sat: float, sat_min: float = 1.0, sat_max: float = 1.35) -> np.ndarray:
    """饱和度驱动: 饱和度越低增强越强。"""
    factor = sat_min + (1.0 - sat) * (sat_max - sat_min)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def enhance(img: np.ndarray, sat: float = 0.0) -> np.ndarray:
    """UEP 自适应主入口: 估计退化 -> 三项自适应增强。"""
    cast = estimate_cast(img)
    contrast = estimate_contrast(img)
    sat_val = estimate_saturation(img)
    out = adaptive_white_balance(img, cast)
    out = adaptive_clahe(out, contrast)
    out = adaptive_saturate(out, sat_val)
    return out


def _worker(args):
    """多进程 worker: (src_path, dst_path)"""
    src, dst = args
    img = cv2.imread(str(src))
    if img is None:
        return str(src.name)
    cv2.imwrite(str(dst), enhance(img))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="UEP: Underwater Enhancement Pipeline (adaptive)")
    ap.add_argument("input", nargs="?", help="单张输入图")
    ap.add_argument("output", nargs="?", help="单张输出图")
    ap.add_argument("--dir", help="批量: 输入目录")
    ap.add_argument("--out", help="批量: 输出目录")
    ap.add_argument("--ext", default="jpg", help="批量: 扩展名 (默认 jpg)")
    ap.add_argument("--workers", type=int, default=0, help="并行进程数 (默认 0=CPU核数)")
    args = ap.parse_args()

    if args.dir:
        import concurrent.futures as cf
        src = Path(args.dir)
        dst = Path(args.out)
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob(f"*.{args.ext}"))
        tasks = [(f, dst / f.name) for f in files]
        n_workers = args.workers or max(1, (os.cpu_count() or 4) - 1)
        failed = []
        with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
            for i, err in enumerate(ex.map(_worker, tasks, chunksize=16)):
                if err:
                    failed.append(err)
                if (i + 1) % 1000 == 0:
                    print(f"  {i+1}/{len(files)}")
        print(f"UEP 自适应批量完成: {len(files)} 张 -> {dst} (失败 {len(failed)})")
        if failed:
            print("失败:", failed[:10])
    else:
        img = cv2.imread(args.input)
        cv2.imwrite(args.output, enhance(img))
        print(f"UEP 完成: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
