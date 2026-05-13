#!/bin/bash

# CUDA
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=6,7
# notice: set CUDA_VISIBLE_DEVICES may cause Vulkan device ordering to differ from CUDA, 
#         which can lead to crashes in Omniverse. 
#         Unset CUDA_VISIBLE_DEVICES if you encounter such issues.

# fix Vulkan reload issue
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# conda and pip
export CONDA_ENVS_PATH=/mnt/data/shaolong/BEHAVIOR-1K/.conda/envs
export CONDA_PKGS_DIRS=/mnt/data/shaolong/BEHAVIOR-1K/.conda/pkgs
export PIP_CACHE_DIR=/mnt/data/shaolong/BEHAVIOR-1K/.cache/pip
export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL

# local caches and temp files
export XDG_CACHE_HOME=/mnt/data/shaolong/BEHAVIOR-1K/.cache
export HF_HOME=/mnt/data/shaolong/BEHAVIOR-1K/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/mnt/data/shaolong/BEHAVIOR-1K/.cache/huggingface/hub
export HF_DATASETS_CACHE=/mnt/data/shaolong/BEHAVIOR-1K/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/mnt/data/shaolong/BEHAVIOR-1K/.cache/huggingface/transformers
export TORCH_HOME=/mnt/data/shaolong/BEHAVIOR-1K/.cache/torch
export TMPDIR=/mnt/data/shaolong/BEHAVIOR-1K/.tmp
export TEMP=/mnt/data/shaolong/BEHAVIOR-1K/.tmp
export TMP=/mnt/data/shaolong/BEHAVIOR-1K/.tmp

conda activate behavior

# export OMNIGIBSON_HEADLESS=1
# export OMNIGIBSON_REMOTE_STREAMING=webrtc
export OMNIGIBSON_REMOTE_STREAMING=websocket
# WebSocket-only tuning knobs (effective when OMNIGIBSON_REMOTE_STREAMING=websocket)
# export OMNIGIBSON_WS_FPS=40
# export OMNIGIBSON_WS_JPEG_QUALITY=92
export PUBLIC_IP=120.48.128.17

# mkdir -p \
# 	/mnt/data/shaolong/BEHAVIOR-1K/.conda/envs \
# 	/mnt/data/shaolong/BEHAVIOR-1K/.conda/pkgs \
# 	/mnt/data/shaolong/BEHAVIOR-1K/.cache/pip \
# 	/mnt/data/shaolong/BEHAVIOR-1K/.cache/huggingface/hub \
# 	/mnt/data/shaolong/BEHAVIOR-1K/.cache/huggingface/datasets \
# 	/mnt/data/shaolong/BEHAVIOR-1K/.cache/huggingface/transformers \
# 	/mnt/data/shaolong/BEHAVIOR-1K/.cache/torch \
# 	/mnt/data/shaolong/BEHAVIOR-1K/.tmp