#!/bin/bash
# 串行训练: B0 -> B1 -> B2 (每轮成功才跑下一轮, 失败即停)
# 用法: nohup bash run_all.sh > run_all.log 2>&1 &
set -u
cd /root/autodl-tmp/Underwater || exit 1

run() {
    local name=$1; shift
    echo "[$(date +%H:%M)] ==== 开始 $name: $* ===="
    python train_underwater.py --epochs 150 --batch 8 --name "$name" "$@" > "logs_${name}.txt" 2>&1
    local code=$?
    if [ $code -ne 0 ]; then
        echo "[$(date +%H:%M)] !!!! $name 失败 (exit=$code), 停止后续训练 !!!!"
        echo "---- logs_${name}.txt 尾部 ----"
        tail -n 25 "logs_${name}.txt"
        exit 1
    fi
    # 打印该轮关键结果 (最后的 all 行)
    echo "[$(date +%H:%M)] ---- $name 完成, 最终结果 ----"
    grep -A1 "^                   all " "logs_${name}.txt" | tail -2 || true
    echo ""
}

run B0 --model yolov8n.yaml
run B1 --model yolov8n.yaml --uep
run B2 --model configs/yolov8n_attn.yaml --uep --attn

echo "[$(date +%H:%M)] ========================================"
echo "[$(date +%H:%M)] 全部三轮完成! 结果见 logs_B0/B1/B2.txt 与 runs/detect/"
echo "[$(date +%H:%M)] ========================================"
