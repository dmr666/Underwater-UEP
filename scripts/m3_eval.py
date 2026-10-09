# -*- coding: utf-8 -*-
"""
M3 对照评估: 分解 UEP 增益来源 (AutoDL 容器端运行)

回答评审核心质疑: +0.8% 增益来自
  (a) "推理时增强" —— 在已训练检测器前直接插入 UEP (论文的部署定位), 还是
  (b) "训练域一致化" —— 训练+测试都用 UEP 带来的训练分布对齐?

2×2 对照:
  B0 (raw 训练) × {raw 测试, UEP 测试}
  B1 (UEP 训练) × {raw 测试, UEP 测试}

关键解读 (核心是第 2 行 B0+UEP测试 的 mAP50):
  · 若 B0+UEP测试 ≈ B1+UEP测试 (0.813) → 增益主要来自"推理时增强", 论文定位成立
  · 若 B0+UEP测试 ≈ B0+raw测试 (0.805) → 增益主要来自"训练域一致化", 叙事需调整
  · 中间值 → 两者兼有, 可按数值拆分贡献

前置:
  1) DUO 数据已在 /root/autodl-tmp/DUO (images/train, images/test,
     images/test_uep, labels/ 均已就绪; test_uep 由 B1 训练时生成)
  2) B0_v2 / B1_v3 权重已在 /root/autodl-tmp/Underwater/runs/detect/

用法:
  cd /root/autodl-tmp/Underwater && python scripts/m3_eval.py
输出: 终端 + m3_result.txt
"""
import sys
from pathlib import Path
from ultralytics import YOLO

BASE = Path("/root/autodl-tmp/DUO")
UWD = Path("/root/autodl-tmp/Underwater")

# 0) 确保 labels/test_uep 复用 labels/test (UEP 不改标注)
lab_test = BASE / "labels" / "test"
lab_uep = BASE / "labels" / "test_uep"
assert lab_test.exists(), f"缺少 {lab_test}, 请先准备好 DUO 数据"
if not lab_uep.exists() and not lab_uep.is_symlink():
    lab_uep.symlink_to(lab_test, target_is_directory=True)
    print("[setup] 已软链 labels/test_uep -> labels/test")
else:
    print("[setup] labels/test_uep 已存在, 跳过")

# 1) 生成 M3 专用 yaml (仅 val/test + names, 无需 train)
def write_yaml(name, val_dir):
    y = BASE / name
    y.write_text(
        f"path: {BASE.as_posix()}\n"
        f"val: {val_dir}\n"
        f"test: {val_dir}\n"
        "names:\n"
        "  0: holothurian\n"
        "  1: echinus\n"
        "  2: scallop\n"
        "  3: starfish\n",
        encoding="utf-8")
    return y

yaml_raw = write_yaml("DUO_m3_raw_test.yaml", "images/test")
yaml_uep = write_yaml("DUO_m3_uep_test.yaml", "images/test_uep")

# 2) 权重
b0_path = UWD / "runs/detect/B0_v2/weights/best.pt"
b1_path = UWD / "runs/detect/B1_v3/weights/best.pt"
assert b0_path.exists(), f"缺少 B0 权重 {b0_path}"
assert b1_path.exists(), f"缺少 B1 权重 {b1_path}"
b0 = YOLO(str(b0_path))
b1 = YOLO(str(b1_path))

# 3) 2×2 评估
def val(model, tag, yaml):
    r = model.val(data=str(yaml), imgsz=640, verbose=False)
    ap50 = [round(float(x), 3) for x in r.box.ap50]
    row = (tag, round(float(r.box.map50), 4), round(float(r.box.map), 4), ap50)
    print(f"[val] {tag}: mAP50={row[1]:.4f}  mAP50-95={row[2]:.4f}  每类AP50={ap50}")
    return row

print("\n=== M3 2×2 对照评估开始 ===")
res = []
res.append(val(b0, "B0(raw训练) + raw测试  [基线]", yaml_raw))
res.append(val(b0, "B0(raw训练) + UEP测试  [M3核心]", yaml_uep))
res.append(val(b1, "B1(UEP训练) + raw测试  [反向]", yaml_raw))
res.append(val(b1, "B1(UEP训练) + UEP测试  [参考]", yaml_uep))

# 4) 解读
b0_raw = res[0][1]
b0_uep = res[1][1]
b1_raw = res[2][1]
b1_uep = res[3][1]
d_infer = b0_uep - b0_raw        # 纯推理时增强贡献
d_align = b1_uep - b0_uep        # 训练域一致化的额外贡献

lines = ["M3 对照: UEP 增益来源分解 (2026-08-27)", "",
         "类序: 0=holothurian 1=echinus 2=scallop 3=starfish", "",
         f"{'评估':<34}{'mAP50':<10}{'mAP50-95':<12}{'每类AP50'}"]
for tag, m50, m95, ap50 in res:
    lines.append(f"{tag:<34}{m50:<10.4f}{m95:<12.4f}{ap50}")
lines += ["",
          f"纯推理时增强贡献  (B0+UEP - B0+raw):   {d_infer:+.4f}",
          f"训练域一致化额外贡献 (B1+UEP - B0+UEP): {d_align:+.4f}",
          f"总计              (B1+UEP - B0+raw):   {b1_uep - b0_raw:+.4f}",
          "",
          "解读:",
          "· 第2行(B0+UEP)接近第4行(B1+UEP)≈0.813 → 增益主要来自推理时增强, 部署定位成立",
          "· 第2行(B0+UEP)接近第1行(B0+raw)≈0.805 → 增益主要来自训练域一致化, 叙事需调整",
          "· 第3行(B1+raw)越低, 说明 UEP 训练的模型越依赖增强输入"]
out = "\n".join(lines)
print("\n" + out)
(UWD / "m3_result.txt").write_text(out, encoding="utf-8")
print("\n结果已存: m3_result.txt")
