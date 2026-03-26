#!/usr/bin/env bash
cd /mnt/ai4s/zhouhaojie/zhangxinrui/OpenAvatarChat
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate oac
source .venv/bin/activate
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="/home/zhouhaojie/.conda/envs/qwen3vl_zxy/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
python src/demo.py --config config/chat_with_openai_compatible_edge_tts.yaml

#bash /mnt/ai4s/zhouhaojie/zhangxinrui/OpenAvatarChat/run.sh