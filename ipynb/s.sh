#!/bin/bash
export PATH="/usr/local/miniconda3/envs/glut/bin:/usr/local/miniconda3/condabin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ffmpeg/bin:/usr/local/cuda/bin"
export HF_ENDPOINT="https://hf-mirror.com"
export MPLBACKEND="agg"
cd /workspace/Tongbi
python glut.py --server_name "0.0.0.0" --server_port 7860 > log.txt 2>&1