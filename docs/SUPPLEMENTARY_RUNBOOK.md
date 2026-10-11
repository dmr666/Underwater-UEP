# Supplementary experiments (reviewer-requested)

## 3. 三组实验（可同时开三个 ssh 会话；显存各约 8 GB，24G 卡可并行）

```bash
python scripts/suppl_m4_m6.py m4  --workers 32          # 4 次训练 ≈ 7.5 h
python scripts/suppl_m4_m6.py m6a --workers 32          # 2 次训练 ≈ 4 h（需 yolov8n.pt，见下）
python scripts/suppl_m4_m6.py m6b --ruod /root/autodl-tmp/RUOD --workers 32   # 2 次训练 ≈ 4 h
```
- 每个 run 完成即落盘 JSON；**断线/关机后重跑会自动跳过已完成的 run**，不必从头来。
- `--workers 32` 是图像增强的并行度（实测 2.7 → 29.5 张/秒），别用 1。
- 长跑建议挂看门狗（仓库 `scripts/keep_deferred_v2.sh` 可直接复用）。
- `yolov8n.pt`（M6a 用）：若实例拉不到 GitHub，用
  `export HF_ENDPOINT=https://hf-mirror.com` 后 `huggingface-cli download Ultralytics/YOLOv8 yolov8n.pt --local-dir .`，
  或直接从你本机/网盘传到 `/root/autodl-tmp/Underwater/`。

## 4. 取回与读结果

```bash
# 实例 → 本机
scp -P $P -r $H:/root/autodl-tmp/Underwater/results_deferred/  "D:/docter/Underwater_SIViP_投稿_20261009/_server_results/suppl/"
# 本机汇总
cd _gpu_deferred && python suppl_m4_m6.py report --results "D:/docter/.../_server_results/suppl"
```
`report` 直接打印回复信要用的三块数字：M4 的 2×2 三种子均值/标准差、M6a/M6b 的四条一致性臂
以及两个关键惩罚（训 UEP 测原图、纯测试端施加 UEP），并自动与论文 from-scratch 版本对照。

## 5. 抢不到 4090 怎么办（这几组实验对卡要求不高）

| 方案 | 说明 |
|---|---|
| **3090 / 3080Ti / A5000（≥12 GB）** | 首选替代：论文的精度数字与 GPU 型号无关（只有延迟相关），150 epochs/640/batch8 在这些卡上都能跑；3090 约 ¥0.88/h，总价更低 |
| **切换 AutoDL 地区** | 4090 常在某些地区紧张；可试西北/西南节点，或"弹性部署" |
| **恒源云 / 蓝耘 / 共绩算力** | 与 AutoDL 同为国内按小时计费，镜像与目录结构类似；把 `/root/autodl-tmp` 的布局照搬即可 |
| **不建议** | Colab/Kaggle 免费额度：单次会话 12 h 上限 + 数据上传慢，8 次训练会被反复打断 |

**总价估算**：8 次训练 ≈ 16 GPU-h（4090）≈ ¥35；3090 上 ≈ 22 h ≈ ¥20。加实例开机与上传约 1 h。

## 6. 跑完写什么（回复信口径）

- **M4**：把 Table 3 里两个 `†`（half→half 0.818、half→full 0.639）从单种子升级为 3 种子
  mean ± sd，并据此把 §4.3"strongest single-seed result"的措辞恢复为不含 "single-seed"。
- **M6a/M6b**：作为"效应可移植性"的证据放进回复信（或 §4.7 新小节）：报告四条臂 + 两个惩罚，
  并注明单种子。若两处惩罚的方向和量级与 from-scratch 版本同向，即能正面回应"泛化性仅限自训场景"。
- 未做的那一组要在回复信里如实写明"留待后续"，不要含糊。

## 7. 已知坑（都是这套项目踩过的，脚本里已规避，改脚本时别改回去）

1. **色偏定义**：`|mean_gray/mean_ch − 1|`（= 实验代码 `uep.py::estimate_cast`）。正文曾写成反比值，已改齐；审计会把这条钉住。
2. **中文路径下 `cv2.imread/imwrite` 静默失败** → 必须 `imdecode(np.fromfile)` / `imencode(...)[1].tofile`。
3. **`seed` 不能既显式传又 `**CFG` 展开**（`TypeError: got multiple values`），会在真训练那刻才炸 → 所以必须有 `smoketrain`。
4. **每轮结束就落盘 JSON**（曾因重写循环丢了 `write_text` 白跑 12 h）。
5. **`.sh` 必须是 LF**，CRLF 会让服务器 bash 报 `$'\r'`。
6. **长跑前确认没有别人在跑同目录**（`pgrep -fa python`），避免 `runs/` 互相覆盖。
