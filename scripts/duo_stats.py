# duo_stats.py — 从 DUO 底图 zip 直接复算：色偏 c、饱和度 s、钳位比例 alpha、逐类饱和度与 bootstrap
# 协议与论文一致：最长边缩到 512 像素后再统计（论文 Sec.4.3 表注）
# 同时验证 Minor 1：HSV 的 S = (max-min)/max，两种写法应逐图完全一致
import json
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ZIP = r"D:\docter\DUO_data.zip"
CLIP = 0.25
OUT = r"D:\docter\Underwater_SIViP_投稿_20261009\_server_results\duo_stats.json"


def one(args):
    name, data = args
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        return None
    h, w = im.shape[:2]
    sc = 512.0 / max(h, w)
    if sc < 1.0:
        im = cv2.resize(im, (int(round(w * sc)), int(round(h * sc))), interpolation=cv2.INTER_AREA)
    m = im.reshape(-1, 3).mean(axis=0)              # BGR 通道均值
    g = m.mean()
    # 论文/原始脚本定义（正确）: c = max_ch |mean_ch/mean_gray - 1|
    c = float(np.max(np.abs(m / np.maximum(g, 1e-6) - 1.0)))
    # 对照（我上一版写反的比值，仅用于说明差异来源）
    c_inv = float(np.max(np.abs(g / np.maximum(m, 1e-6) - 1.0)))
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    s_hsv = float(hsv[..., 1].mean() / 255.0)                    # 论文 Sec.3.2 写法：mean(I_HSV,S)/255
    mx = im.max(axis=2).astype(np.float32)
    mn = im.min(axis=2).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    s_alt = float(sat.mean())                                    # 论文 Sec.4.3 写法：mean((max-min)/max)
    return name, c, s_hsv, s_alt, c_inv


def classes_of(labels: dict, name: str):
    txt = labels.get(name)
    if txt is None:
        return set()
    out = set()
    for line in txt.splitlines():
        p = line.split()
        if p:
            out.add(int(float(p[0])))
    return out


def main():
    zf = zipfile.ZipFile(ZIP)
    imgs = [n for n in zf.namelist() if n.startswith("DUO/images/train/") and n.lower().endswith((".jpg", ".png"))]
    imgs.sort()
    labels = {}
    for n in zf.namelist():
        if n.startswith("DUO/labels/train/") and n.lower().endswith(".txt"):
            key = "DUO/images/train/" + Path(n).stem + Path(imgs[0]).suffix
            labels[key] = zf.read(n).decode("utf-8", "ignore")

    jobs = [(n, zf.read(n)) for n in imgs]
    print(f"图像 {len(jobs)} 张，开始（8 进程）…", flush=True)
    with Pool(8) as pool:
        res = [r for r in pool.imap_unordered(one, jobs, chunksize=32) if r]
    print(f"完成 {len(res)} 张", flush=True)

    c = np.array([r[1] for r in res]); s = np.array([r[2] for r in res]); salt = np.array([r[3] for r in res])
    cinv = np.array([r[4] for r in res])          # 反比值版本（仅用于说明差异来源）
    names = {0: "holothurian", 1: "echinus", 2: "scallop", 3: "starfish"}
    per_class = {v: [] for v in names.values()}
    for r in res:
        for cl in classes_of(labels, r[0]):
            if cl in names:
                per_class[names[cl]].append(r[2])

    # bootstrap：holothurian 的饱和度是否稳定最低
    rng = np.random.default_rng(0)
    lowest = 0
    B = 2000
    arrs = {k: np.array(v) for k, v in per_class.items() if len(v) > 10}
    keys = list(arrs)
    for _ in range(B):
        means = {k: arrs[k][rng.integers(0, len(arrs[k]), len(arrs[k]))].mean() for k in keys}
        if min(means, key=means.get) == "holothurian":
            lowest += 1

    out = dict(
        n_images=len(res),
        clip_threshold=CLIP,
        cast=dict(mean=float(c.mean()), std=float(c.std(ddof=1)), q10=float(np.percentile(c, 10)),
                  q25=float(np.percentile(c, 25)), q50=float(np.percentile(c, 50)),
                  q75=float(np.percentile(c, 75)), q90=float(np.percentile(c, 90))),
        sat=dict(mean=float(s.mean()), std=float(s.std(ddof=1)), q10=float(np.percentile(s, 10)),
                 q50=float(np.percentile(s, 50)), q90=float(np.percentile(s, 90))),
        alpha_clamped_fraction=float((c >= CLIP).mean()),
        alpha_mean=float(np.minimum(c / CLIP, 1.0).mean()),
        alpha_hist=[float((np.minimum(c / CLIP, 1.0) < t).mean()) for t in (0.25, 0.5, 0.75, 0.9, 1.0)],
        two_sat_definitions_max_abs_diff=float(np.max(np.abs(s - salt))),
        cast_inverted_mean=float(cinv.mean()), cast_inverted_std=float(cinv.std(ddof=1)),
        per_class_sat={k: dict(n=len(v), mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
                       for k, v in per_class.items()},
        bootstrap_holothurian_lowest_fraction=lowest / B,
        bootstrap_B=B,
    )
    Path(OUT).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
