# 水下目标检测 (DUO) AutoDL 部署手册

## 一、文件清单
- 代码包 Underwater_代码包.zip: train_underwater.py / configs/ / ultralytics_patch/ / scripts/uep.py / autodl_setup.sh
- 数据包 DUO_data.zip: data/DUO/ (images train+test+train_uep, labels, yaml) 约 6GB

## 二、AutoDL 上传后 (按顺序)

### 1. 环境
    cd /root/autodl-tmp
    unzip Underwater_代码包.zip
    unzip DUO_data.zip
    bash autodl_setup.sh

   目录结构预期:
   /root/autodl-tmp/DUO/            <- 数据 (DUO_data.zip 解压: images/train images/test labels)
   /root/autodl-tmp/Underwater/     <- 代码 (代码包解压)

### 1.5 在 AutoDL 上生成 UEP 增强训练集 (约 5-10 分钟)
    cd /root/autodl-tmp/Underwater
    python scripts/uep.py --dir /root/autodl-tmp/DUO/images/train --out /root/autodl-tmp/DUO/images/train_uep --ext jpg
    (多进程自动, 完成后 6671 张)

### 2. 冒烟测试 (1 epoch, 约 3 分钟)
    cd /root/autodl-tmp/Underwater
    python train_underwater.py --model yolov8n.yaml --uep --epochs 1 --batch 8 --name smoke

   (注: train_underwater.py 里 DATA 路径默认 /root/autodl-tmp/DUO/data, 如目录不同需改)

### 3. 正式训练 (依次, 一卡一任务)
   B0 基线 (原始图像训练):
    nohup python train_underwater.py --model yolov8n.yaml --epochs 150 --batch 8 --name B0 > logs_B0.txt 2>&1 &

   B1 (+UEP 增强训练集):
    nohup python train_underwater.py --model yolov8n.yaml --uep --epochs 150 --batch 8 --name B1 > logs_B1.txt 2>&1 &

   B2 (完整: UEP + LCA 注意力):
    nohup python train_underwater.py --model configs/yolov8n_attn.yaml --uep --epochs 150 --batch 8 --name B2 --attn > logs_B2.txt 2>&1 &

   每轮预计 2-4 小时 (6671 图, batch=8, 150 epochs)。

### 4. 验证
   tail -n 20 logs_B0.txt   # 看 "150 epochs completed" 和 mAP
   每轮结束自动在 test 集 (1111 图) 验证, 输出 per-class AP。

### 5. 下载结果
   打包 runs/detect/ 下 B0/B1/B2 到本地, 然后关机 (关机不扣钱):
    tar czf B0_B1_B2.tar.gz runs/detect/B0 runs/detect/B1 runs/detect/B2

## 三、注意
- 数据 6GB 建议用百度网盘 -> AutoDL 或 AutoDL 自带的数据传输
- 禁止 pip 升级 ultralytics (8.4.x 不兼容)
- 测试集 (images/test + labels/test) 是官方划分, 评估直接用, 不要再划分
