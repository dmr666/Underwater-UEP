#!/bin/bash
# restart_sens.sh — 停旧 → 自检 → 真训练预检(1 epoch) → 通过才启动 6 组敏感性 + 看门狗
cd /root/autodl-tmp || exit 1
PY=/root/miniconda3/bin/python

echo "=== 1) 停掉旧进程（含崩溃循环的看门狗）==="
pkill -f "deferred_experiments.py" 2>/dev/null
pkill -f "keep_deferred_v2.sh" 2>/dev/null
pkill -f "keep_deferred\.sh" 2>/dev/null
sleep 3
pgrep -af "deferred_experiments.py" | grep -v "bash -c" | head -3

echo "=== 2) 生成 & 算子自检 ==="
$PY deferred_experiments.py selftest --workers 16 2>&1 | tail -2 || { echo "SELFTEST 失败，中止"; exit 1; }

echo "=== 3) 真训练预检（1 epoch，含评测；只跑一次）==="
$PY deferred_experiments.py smoketrain > /root/autodl-tmp/smoketrain.log 2>&1
tail -8 /root/autodl-tmp/smoketrain.log
if ! grep -q "SMOKETRAIN 通过" /root/autodl-tmp/smoketrain.log; then
  echo "SMOKETRAIN 未通过，不启动长跑"; exit 1
fi

echo "=== 4) 启动 6 组敏感性（32 进程生成）+ 看门狗 ==="
: > /root/autodl-tmp/sens.log
nohup $PY deferred_experiments.py sensitivity --data /root/autodl-tmp/DUO \
  --runs /root/autodl-tmp/Underwater/runs/detect --workers 32 >> /root/autodl-tmp/sens.log 2>&1 &
sleep 5
nohup bash keep_deferred_v2.sh > /dev/null 2>&1 &
sleep 3
echo "=== 进程 ==="
pgrep -af "deferred_experiments.py" | grep -v "bash -c" | head -2
cat /root/autodl-tmp/watchdog.pid 2>/dev/null | sed 's/^/watchdog pid: /'
echo "=== 磁盘 ==="; df -h /root/autodl-tmp | tail -1
