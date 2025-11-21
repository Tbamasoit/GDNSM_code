# !/bin/bash

echo "Downloading datasets from OBS..."

# 使用 wget 或 curl 命令，并输入你在第一步中获取的 URL
wget -O datasets.zip https://gdnsm.obs.cn-east-3.myhuaweicloud.com/datasets.zip

echo "Unzipping datasets..."
unzip datasets.zip

# (可选) 删除压缩包
rm datasets.zip

echo "Data preparation complete. The 'datasets' folder is ready."