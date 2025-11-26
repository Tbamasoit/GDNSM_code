#!/bin/bash

# === 配置区域 ===
# 你的 OBS 下载链接
DATA_URL="https://gdnsm.obs.cn-east-3.myhuaweicloud.com/datasets.zip"

# 缓存目录 (临时盘，空间大)
CACHE_DIR="/cache"

# 项目中的数据目录位置 (持久化盘)
# 注意：请确认你的项目文件夹名字是 GDNSM
PROJECT_DATA_PATH="/home/ma-user/work/GDNSM/datasets"

echo "========================================"
echo "   开始初始化数据集 (OBS -> Cache)   "
echo "========================================"

# 1. 检查 /cache 下是否已经有数据 (防止重复下载)
if [ -d "$CACHE_DIR/datasets" ]; then
    echo "检测到 /cache/datasets 已存在，跳过下载。"
else
    echo "正在下载数据集到 /cache ..."
    # -O 指定下载文件的保存路径和文件名
    wget -O "$CACHE_DIR/datasets.zip" "$DATA_URL"

    echo "正在解压..."
    # -d 指定解压的目标目录
    unzip -q "$CACHE_DIR/datasets.zip" -d "$CACHE_DIR/"
    
    # [新增] 检查是否解压出了大写的 Datasets，如果是，改成小写
    if [ -d "$CACHE_DIR/Datasets" ]; then
        echo "检测到大写 Datasets 文件夹，正在重命名为 datasets..."
        mv "$CACHE_DIR/Datasets" "$CACHE_DIR/datasets"
    fi

    # 删除压缩包释放空间
    rm "$CACHE_DIR/datasets.zip"
    echo "下载并解压完成！"
fi

# 2. 建立软链接 (关键步骤！)
# 这一步是为了让代码能找到数据，同时不占用 work 的空间

echo "正在配置软链接..."

# 先删除 work 下可能存在的空文件夹 (否则软链接会建立在文件夹里面)
if [ -d "$PROJECT_DATA_PATH" ] && [ ! -L "$PROJECT_DATA_PATH" ]; then
    echo "清理旧的 datasets 文件夹..."
    rm -rf "$PROJECT_DATA_PATH"
fi

# 如果软链接不存在，则创建
if [ ! -L "$PROJECT_DATA_PATH" ]; then
    ln -s "$CACHE_DIR/datasets" "$PROJECT_DATA_PATH"
    echo "软链接创建成功：$PROJECT_DATA_PATH -> $CACHE_DIR/datasets"
else
    echo "软链接已存在，无需操作。"
fi

echo "========================================"
echo "          数据初始化完毕！              "
echo "========================================"
