# -*- coding: utf-8 -*-
"""
按目标尺度报告 AP (COCO 标准 AP_s / AP_m / AP_l), 用于论文尺度分析。
流程: ultralytics val (save_json) -> pycocotools COCOeval (按面积分桶)。
注意: ultralytics save_json 的 category_id 是 COCO 原始编号 (1-4), 与 DUO gt 一致, 直接匹配。

容器用法 (在 /root/autodl-tmp/Underwater):
  B0 (原始图):   python scripts/ap_by_size.py --weights runs/detect/B0_v2/weights/best.pt --name B0 --data /root/autodl-tmp/DUO/DUO.yaml --gt /root/autodl-tmp/DUO/annotations/instances_test.json
  B1 (自适应):   python scripts/ap_by_size.py --weights runs/detect/B1_v3/weights/best.pt --name B1 --data /root/autodl-tmp/DUO/DUO_uep.yaml --gt /root/autodl-tmp/DUO/annotations/instances_test.json
  B2 (完整):     python scripts/ap_by_size.py --weights runs/detect/B2_v2/weights/best.pt --name B2 --data /root/autodl-tmp/DUO/DUO_uep.yaml --gt /root/autodl-tmp/DUO/annotations/instances_test.json
  B1_fixed:      python scripts/ap_by_size.py --weights runs/detect/B1_fixed/weights/best.pt --name B1f --data /root/autodl-tmp/DUO/DUO_uep_fixed.yaml --gt /root/autodl-tmp/DUO/annotations/instances_test.json
输出: AP / AP50 / AP75 / APs / APm / APl (0.5:0.95) + AP50 分桶
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()

    # 注册自定义模块 (LCA) — B2_v2 权重需要; 对 B0/B1 无害
    BASE = Path(__file__).resolve().parent.parent
    patch_dir = BASE / "ultralytics_patch"
    if patch_dir.exists():
        sys.path.insert(0, str(patch_dir))
        try:
            from attention import register_attention
            register_attention()
            print("[ap_by_size] LCA 自定义模块已注册")
        except Exception as e:
            print(f"[ap_by_size] 自定义模块注册跳过: {e}")

    # 1) val + save_json
    from ultralytics import YOLO
    m = YOLO(args.weights)
    r = m.val(data=args.data, imgsz=640, save_json=True, name=args.name, verbose=False)
    pred_path = Path(r.save_dir) / "predictions.json"
    assert pred_path.exists(), f"predictions.json 未生成: {pred_path}"
    print(f"[{args.name}] 预测文件: {pred_path}")

    # 2) pycocotools (按需安装)
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("安装 pycocotools ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pycocotools", "-q"], check=True)
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

    # 3) 直接用原始 GT: ultralytics save_json 的 category_id 已是 COCO 原始编号 (1-4),
    #    与 DUO gt 一致, 无需映射。
    gt = COCO(args.gt)
    pred = gt.loadRes(str(pred_path))

    # eval1: 标准 COCO (IoU 0.5:0.95), summarize() 填充 stats
    ev = COCOeval(gt, pred, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats  # [AP, AP50, AP75, APs, APm, APl, ...]

    # eval2: 仅 IoU=0.5 的尺度分桶 (summarize 会因单 IoU 越界, 手动从 precision 计算)
    import numpy as np
    ev2 = COCOeval(gt, pred, "bbox")
    ev2.params.iouThrs = [0.5]
    ev2.evaluate()
    ev2.accumulate()
    prec = ev2.eval["precision"]  # (T=1, R, K, A=4, M)

    def ap_area(aidx):
        # 最后一维是 maxDets=[1,10,100] 的索引, 必须用 2 (=100) 与 COCO 标准一致
        p = prec[0, :, :, aidx, 2]
        p = p[p != -1]  # 去掉无 GT 的面积组
        return float(np.mean(p)) if p.size else 0.0

    s50 = (ap_area(1), ap_area(2), ap_area(3))  # small / medium / large @ IoU=0.5
    print(f"\n[{args.name}] COCO 标准 (IoU 0.5:0.95):")
    print(f"  AP={s[0]:.3f}  AP50={s[1]:.3f}  AP75={s[2]:.3f}")
    print(f"  APs={s[3]:.3f}  APm={s[4]:.3f}  APl={s[5]:.3f}")
    print(f"[{args.name}] IoU=0.5 分桶 (论文 mAP50 口径):")
    print(f"  AP50_s={s50[0]:.3f}  AP50_m={s50[1]:.3f}  AP50_l={s50[2]:.3f}")
    print(f"\n结果行: {args.name} AP={s[0]:.3f} AP50={s[1]:.3f} APs={s[3]:.3f} APm={s[4]:.3f} APl={s[5]:.3f} | AP50_s={s50[0]:.3f} AP50_m={s50[1]:.3f} AP50_l={s50[2]:.3f}")


if __name__ == "__main__":
    main()
