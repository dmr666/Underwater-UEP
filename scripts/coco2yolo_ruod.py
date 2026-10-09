# -*- coding: utf-8 -*-
"""
RUOD COCO -> YOLO 格式转换 (泛化评估用)
只保留与 DUO 共享的 4 类: holothurian(1) echinus(2) scallop(3) starfish(4) -> 0..3
生成: labels/test(全量 4200 图) + labels/{blur,color,light}(环境子集各 100 图)
     + RUOD_test.yaml / RUOD_test_uep.yaml / RUOD_{env}.yaml / RUOD_{env}_uep.yaml
用法: python scripts/coco2yolo_ruod.py
"""
import json
from pathlib import Path

BASE = Path(r"D:/docter/Underwater_YOLO/data")
SRC = BASE / "RUOD"
OUT = BASE / "yolo_ruod"

# RUOD 类别 id -> YOLO 类别 (只保留共享 4 类)
CAT_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
NAMES = ["holothurian", "echinus", "scallop", "starfish"]


def convert(coco_path: Path, out_label_dir: Path, tag: str):
    d = json.load(open(coco_path, encoding="utf-8"))
    img_info = {im["id"]: im for im in d["images"]}
    out_label_dir.mkdir(parents=True, exist_ok=True)

    # 为所有图预建空 txt (无共享类目标的图 = 背景图, 同样参与评估)
    for im in d["images"]:
        stem = im["file_name"].rsplit(".", 1)[0]
        (out_label_dir / f"{stem}.txt").touch()

    n_keep = 0
    n_drop_cat = 0      # 非共享类
    n_drop_invalid = 0  # 非法框
    per_class = {i: 0 for i in range(4)}
    for ann in d["annotations"]:
        cid = ann["category_id"]
        if cid not in CAT_MAP:
            n_drop_cat += 1
            continue
        im = img_info.get(ann["image_id"])
        if im is None:
            continue
        w, h = im["width"], im["height"]
        x, y, bw, bh = ann["bbox"]
        if bw <= 0 or bh <= 0 or x < 0 or y < 0 or x + bw > w + 1 or y + bh > h + 1:
            n_drop_invalid += 1
            continue
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        # 钳制到 [0,1] (避免浮点越界导致 ultralytics 警告)
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        nw, nh = min(max(nw, 0.0), 1.0), min(max(nh, 0.0), 1.0)
        cls = CAT_MAP[cid]
        stem = im["file_name"].rsplit(".", 1)[0]
        with open(out_label_dir / f"{stem}.txt", "a") as f:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        n_keep += 1
        per_class[cls] += 1

    n_txt = len(list(out_label_dir.glob("*.txt")))
    print(f"[{tag}] 图 {len(img_info)} | 实例(共享4类) {n_keep} {per_class} | "
          f"txt {n_txt} | 丢非共享 {n_drop_cat} | 丢非法 {n_drop_invalid}")
    return per_class


def write_yaml(path: Path, val_images: str, tag: str):
    path.write_text(
        f"path: /root/autodl-tmp/RUOD\n"
        f"train: images/test\n"
        f"val: {val_images}\n"
        f"names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8")
    print(f"[yaml] {tag} -> {path.name}")


def main():
    # 1) 全量 test 标注
    convert(SRC / "RUOD_ANN/instances_test.json", OUT / "labels/test", "test")
    # 2) 环境子集标注
    for env in ["blur", "color", "light"]:
        convert(SRC / f"Environmet_ANN/instances_{env}.json",
                OUT / f"labels/{env}", env)
    # 3) yaml (容器布局: /root/autodl-tmp/RUOD, images/ 与 labels/ 同根)
    write_yaml(OUT / "RUOD_test.yaml", "images/test", "原始全量")
    write_yaml(OUT / "RUOD_test_uep.yaml", "images/test_uep", "UEP全量")
    for env in ["blur", "color", "light"]:
        write_yaml(OUT / f"RUOD_{env}.yaml", f"images/{env}", f"原始{env}")
        write_yaml(OUT / f"RUOD_{env}_uep.yaml", f"images/{env}_uep", f"UEP{env}")


if __name__ == "__main__":
    main()
