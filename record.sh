#!/bin/bash

# 设置GPU索引
GPU_INDEX=3

# 设置日志文件路径
LOG_FILE="gpu_memory_usage.log"

# 开始监控
echo "Monitoring GPU memory usage on GPU $GPU_INDEX. Logging to $LOG_FILE."
echo "Time,Used Memory (MiB)" > $LOG_FILE

while true; do
    # 使用nvidia-smi获取特定GPU的显存使用情况
    MEMORY_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_INDEX | tr -d '[:space:]')
    
    # 获取当前时间戳
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    
    # 将时间戳和显存使用情况记录到日志文件
    echo "$TIMESTAMP,$MEMORY_USAGE" >> $LOG_FILE
    
    # 每秒更新一次
    sleep 5
done
