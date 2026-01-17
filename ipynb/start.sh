#!/bin/bash

REAL_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$REAL_PATH")

echo "Starting to run $SCRIPT_DIR..."
echo "Log file: /root/GLM-Image-diffusers/log.txt"

source "$SCRIPT_DIR/venv/bin/activate"

export HF_ENDPOINT="https://hf-mirror.com"

cd /root/GLM-Image-diffusers

# 使用 tee 同时输出到屏幕和文件
python glut.py --server_name "0.0.0.0" --server_port 8111 2>&1 | tee /root/GLM-Image-diffusers/log.txt