# -*- coding: utf-8 -*-
"""
DUO COCO -> YOLO 格式转换 + data.yaml 生成
用法: python coco2yolo.py --data D:/docter/Underwater_YOLO/data/DUO --out D:/docter/Underwater_YOLO/data/yolo
"""
import argparse
import json
from pathlib import Path


def convert(coco_path: Path, img_dir: Path, out_dir: Path, split: str) -> None:
    with open(coco_path, encoding="utf-8") as f:
        d = json.load(f)
    cat_map = {c["id"]: i for i, c in enumerate(sorted(d["categories"], key=lambda x: x["id"]))}
    img_info = {im["id"]: im for im in d["images"]}

    label_dir = out_dir / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)

    n_skip_iscrowd = 0
    n_skip_invalid = 0
    n_empty = 0
    for ann in d["annotations"]:
        if ann.get("iscrowd", 0):
            n_skip_iscrowd += 1
            continue
        im = img_info.get(ann["image_id"])
        if im is None:
            continue
        w, h = im["width"], im["height"]
        x, y, bw, bh = ann["bbox"]
        if bw <= 0 or bh <= 0 or x < 0 or y < 0 or x + bw > w + 1 or y + bh > h + 1:
            n_skip_invalid += 1
            continue
        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        cls = cat_map[ann["category_id"]]
        with open(label_dir / (im["file_name"].rsplit(".", 1)[0] + ".txt"), "a") as f:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    n_txt = len(list(label_dir.glob("*.txt")))
    print(f"[{split}] 图像 {len(img_info)} | txt {n_txt} | 跳过 iscrowd {n_skip_iscrowd} | 跳过非法框 {n_skip_invalid}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"D:/docter/Underwater_YOLO/data/DUO")
    ap.add_argument("--out", default=r"D:/docter/Underwater_YOLO/data/yolo")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)

    for split in ["train", "test"]:
        convert(data / f"annotations/instances_{split}.json",
                data / f"images/{split}", out, split)

    # data.yaml (test 集即验证集, 官方划分)
    names = ["holothurian", "echinus", "scallop", "starfish"]
    yaml_text = (
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/test\n"
        "test: images/test\n"
        "names:\n"
        + "\n".join(f"  {i}: {n}" for i, n in enumerate(names)) + "\n"
    )
    (out / "DUO.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"data.yaml 已生成: {out / 'DUO.yaml'}")
    print(f"类别: {names}")


if __name__ == "__main__":
    main()
