#!/bin/bash
# AutoDL 环境一键配置 (水下课题)
# 用法: bash autodl_setup.sh
set -e
echo "[1/3] 固定依赖版本 (ultralytics 8.3.123 + numpy 1.26.4 + pandas 2.2.3)..."
pip install "ultralytics==8.3.123" "numpy==1.26.4" "pandas==2.2.3" -i https://pypi.tuna.tsinghua.edu.cn/simple -q

echo "[2/3] 验证..."
python -c "import ultralytics, numpy, pandas; print('ultralytics', ultralytics.__version__, '| numpy', numpy.__version__, '| pandas', pandas.__version__)"

echo "[3/3] 完成。注意: 禁止 pip 升级 ultralytics (8.4.x 不兼容)。"
