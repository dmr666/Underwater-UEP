# -*- coding: utf-8 -*-
"""
RUOD 跨数据集泛化评估 (AutoDL 容器端运行)
前置: 1) 图片已上传到 /root/autodl-tmp/RUOD/images/test/ (4200 张 jpg)
      2) labels/ 与 yaml 已从 Underwater_RUOD评估.zip 解压到 /root/autodl-tmp/RUOD/
      3) B0_v2/B1_v3 权重在 /root/autodl-tmp/Underwater/runs/detect/
流程: 生成增强图 -> 软链环境子集 -> 8 次 val -> 汇总 ruod_result.txt
用法: cd /root/autodl-tmp/Underwater && python scripts/ruod_eval.py
"""
import shutil, sys
from pathlib import Path

BASE = Path("/root/autodl-tmp/RUOD")
UWD = Path("/root/autodl-tmp/Underwater")
TEST_IMG = BASE / "images" / "test"
sys.path.insert(0, str(UWD / "scripts"))

assert TEST_IMG.exists(), f"缺少图片目录 {TEST_IMG}, 请先上传 RUOD test 图片"
assert (BASE / "labels/test").exists(), "缺少 labels/, 请先解压 Underwater_RUOD评估.zip 到 /root/autodl-tmp/RUOD"

# 1) 生成全量增强图 (与 DUO 同一自适应 uep.enhance)
from uep import enhance
import cv2

uep_dir = BASE / "images" / "test_uep"
if not uep_dir.exists():
    uep_dir.mkdir(parents=True)
    jpgs = sorted(TEST_IMG.glob("*.jpg"))
    for i, f in enumerate(jpgs, 1):
        img = cv2.imread(str(f))
        cv2.imwrite(str(uep_dir / f.name), enhance(img))
        if i % 1000 == 0:
            print(f"[enhance] {i}/{len(jpgs)}")
    print(f"[enhance] 全量增强完成: {len(jpgs)} 张 -> {uep_dir}")
else:
    print("[enhance] test_uep 已存在, 跳过")

# 2) 软链环境子集 (图片从 test / test_uep 链接, 标注复用 labels/{env})
for env in ["blur", "color", "light"]:
    lab = BASE / "labels" / env
    assert lab.exists(), f"缺少 labels/{env}"
    stems = [f.stem for f in lab.glob("*.txt")]
    for sub, src_img in [(BASE / f"images/{env}", TEST_IMG),
                         (BASE / f"images/{env}_uep", uep_dir)]:
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir(parents=True)
        for s in stems:
            (sub / f"{s}.jpg").symlink_to(src_img / f"{s}.jpg")
    # labels/{env}_uep 复用 labels/{env}
    lab_uep = BASE / "labels" / f"{env}_uep"
    if lab_uep.exists() or lab_uep.is_symlink():
        lab_uep.unlink()
    lab_uep.symlink_to(lab, target_is_directory=True)
    print(f"[subset] {env}: {len(stems)} 图 (软链)")

# 3) labels/test_uep 复用 labels/test
lab_test_uep = BASE / "labels" / "test_uep"
if lab_test_uep.exists() or lab_test_uep.is_symlink():
    lab_test_uep.unlink()
lab_test_uep.symlink_to(BASE / "labels" / "test", target_is_directory=True)

# 4) 8 次 val
from ultralytics import YOLO
m0 = YOLO(str(UWD / "runs/detect/B0_v2/weights/best.pt"))
m1 = YOLO(str(UWD / "runs/detect/B1_v3/weights/best.pt"))

results = []
def run_val(model, tag, yaml_name):
    r = model.val(data=str(BASE / yaml_name), imgsz=640, verbose=False)
    ap50 = [round(float(x), 3) for x in r.box.ap50]
    row = (tag, round(float(r.box.map50), 4), round(float(r.box.map), 4), ap50)
    results.append(row)
    print(f"[val] {tag}: mAP50={row[1]:.4f} mAP50-95={row[2]:.4f} 每类AP50={ap50}")

print("\n=== 评估开始 ===")
run_val(m0, "test-raw     ", "RUOD_test.yaml")
run_val(m1, "test-uep     ", "RUOD_test_uep.yaml")
for env in ["blur", "color", "light"]:
    run_val(m0, f"{env}-raw      ", f"RUOD_{env}.yaml")
    run_val(m1, f"{env}-uep      ", f"RUOD_{env}_uep.yaml")

# 5) 汇总
lines = ["RUOD 泛化评估 (DUO 训练模型, 共享4类, 2026-08-24)",
         "类序: 0=holothurian 1=echinus 2=scallop 3=starfish",
         f"{'评估':<14}{'mAP50':<10}{'mAP50-95':<12}{'每类AP50'}",]
for tag, m50, m95, ap50 in results:
    lines.append(f"{tag:<14}{m50:<10.4f}{m95:<12.4f}{ap50}")
out = "\n".join(lines)
print("\n" + out)
(UWD / "ruod_result.txt").write_text(out, encoding="utf-8")
print("\n结果已存: ruod_result.txt")
