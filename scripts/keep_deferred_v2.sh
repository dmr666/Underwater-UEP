#!/bin/bash
# keep_deferred_v2.sh — 看门狗 v2（多进程版）
#  1) 敏感性实验掉了就续跑（32 进程生成变体图）；
#  2) 敏感性完成后自动接分级强度（M2），无需人工；
#  3) 清理已评测过的变体图（结果已落 JSON，图可由脚本重生成）。
cd /root/autodl-tmp || exit 1
PY=/root/miniconda3/bin/python
W=/root/autodl-tmp/deferred_watchdog.log
SENS=/root/autodl-tmp/sens.log
GRAD=/root/autodl-tmp/graded.log
echo $$ > /root/autodl-tmp/watchdog.pid
echo "[watchdog v2] 启动 $(date) pid=$$" >> $W
while true; do
  for n in $(grep -ho '\[eval\] [A-Za-z0-9_]*' $SENS $GRAD 2>/dev/null | awk '{print $2}' | sort -u); do
    if [ -d "/root/autodl-tmp/DUO/images/train_$n" ]; then
      rm -rf "/root/autodl-tmp/DUO/images/train_$n" "/root/autodl-tmp/DUO/images/test_$n" \
             "/root/autodl-tmp/DUO/labels/train_$n" "/root/autodl-tmp/DUO/labels/test_$n"
      echo "[watchdog] $(date) 已清理已评测变体图 $n" >> $W
    fi
  done
  if [ ! -f /root/autodl-tmp/results_deferred/summary_sensitivity.json ]; then
    if ! pgrep -f "deferred_experiments.py sensitivity" > /dev/null; then
      echo "[watchdog] $(date) 敏感性进程不在，续跑" >> $W
      nohup $PY deferred_experiments.py sensitivity --data /root/autodl-tmp/DUO \
        --runs /root/autodl-tmp/Underwater/runs/detect --workers 32 >> $SENS 2>&1 &
    fi
  elif [ ! -f /root/autodl-tmp/results_deferred/summary_graded.json ]; then
    if ! pgrep -f "deferred_experiments.py graded" > /dev/null; then
      echo "[watchdog] $(date) 启动分级强度（M2）" >> $W
      nohup $PY deferred_experiments.py graded --data /root/autodl-tmp/DUO \
        --runs /root/autodl-tmp/Underwater/runs/detect --workers 32 >> $GRAD 2>&1 &
    fi
  else
    echo "[watchdog] $(date) 两组实验均完成，退出" >> $W
    break
  fi
  sleep 600
done
