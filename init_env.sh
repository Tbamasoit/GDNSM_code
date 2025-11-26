#!/bin/bash

BACKUP_PATH="/home/ma-user/work/envs_backup/gdnsm_env.tar.gz"
ENV_DIR="/cache/envs"
ENV_NAME="gdnsm_py37"
TARGET_PATH="$ENV_DIR/$ENV_NAME"

echo "========================================"
echo "      环境一键复活脚本 (Snapshot Mode)   "
echo "========================================"

# 1. 检查是否需要解压
if [ -d "$TARGET_PATH" ]; then
    echo "环境 $TARGET_PATH 已存在，跳过解压。"
else
    if [ -f "$BACKUP_PATH" ]; then
        echo "正在从 $BACKUP_PATH 解压环境..."
        mkdir -p "$ENV_DIR"
        # 解压
        tar -xzf "$BACKUP_PATH" -C "$ENV_DIR"
        echo "解压完成！"
    else
        echo "错误：找不到备份文件 $BACKUP_PATH"
        exit 1
    fi
fi

# 2. 注册 Kernel (因为重启后 Kernel 注册表会清空，必须重做)
echo "正在注册 Jupyter Kernel..."
source activate "$TARGET_PATH"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "GDNSM (Snapshot)"

echo "========================================"
echo "   环境准备就绪！请刷新页面选择 Kernel   "
echo "========================================"