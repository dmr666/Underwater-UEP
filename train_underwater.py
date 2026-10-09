# -*- coding: utf-8 -*-
"""
水下目标检测统一训练入口 (DUO + YOLOv8n)
用法:
  B0 基线:      python train_underwater.py --model yolov8n.yaml --data DUO.yaml --name B0
  B1 +UEP:      python train_underwater.py --model yolov8n.yaml --data DUO_uep.yaml --name B1
  B2 完整:      python train_underwater.py --model configs/yolov8n_attn.yaml --data DUO_uep.yaml --name B2 --attn
协议写死: pretrained=False / patience=0 / seed=0 / batch=8 / imgsz=640 / epochs=150
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA = Path("/root/autodl-tmp/DUO")   # DUO_data.zip 解压后 DUO 即数据根 (images/labels 直接在其下)


def build_data_yaml(use_uep: bool) -> Path:
    """生成 data yaml (原始 or UEP 增强版) — 注意: UEP 是端到端管线, 训练/验证都用增强集"""
    train_dir = "images/train_uep" if use_uep else "images/train"
    val_dir = "images/test_uep" if use_uep else "images/test"   # UEP 协议: 测试集也要增强
    y = DATA / ("DUO_uep.yaml" if use_uep else "DUO.yaml")
    y.write_text(
        f"path: {DATA.as_posix()}\n"
        f"train: {train_dir}\n"
        f"val: {val_dir}\n"
        f"test: {val_dir}\n"
        "names:\n"
        "  0: holothurian\n"
        "  1: echinus\n"
        "  2: scallop\n"
        "  3: starfish\n", encoding="utf-8")
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.yaml")
    ap.add_argument("--data", default=None, help="yaml 名 (默认自动: DUO.yaml / DUO_uep.yaml)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--uep", action="store_true", help="训练集用 UEP 增强版")
    ap.add_argument("--attn", action="store_true", help="B2: 启用注意力模型")
    args = ap.parse_args()

    data = args.data or ("DUO_uep.yaml" if args.uep else "DUO.yaml")
    build_data_yaml(args.uep)   # 总是生成对应 yaml (B0 生成 DUO.yaml, B1/B2 生成 DUO_uep.yaml)

    if args.attn:
        # 注册注意力模块 (见 ultralytics_patch/attention.py)
        import sys
        sys.path.insert(0, str(BASE / "ultralytics_patch"))
        from attention import register_attention
        register_attention()

    model = YOLO(args.model)
    model.train(
        data=str(DATA / data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        pretrained=False,
        patience=0,
        seed=0,
        device=0,
        workers=8,
        optimizer="AdamW",   # 消融公平: 三轮统一优化器+lr (auto 会按模型结构改优化器和 lr, 必须写死)
        lr0=0.00125,         # 匹配 auto-AdamW 的默认 lr (B0/B1 日志实测 lr=0.00125)
        mixup=0.0,           # 显式禁用 mixup(密集小目标场景会崩 recall)
        copy_paste=0.0,      # 显式禁用 copy-paste(同上)
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
