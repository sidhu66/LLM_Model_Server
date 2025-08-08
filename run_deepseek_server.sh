#!/bin/bash

cd /home/sidhu66/LADy/LLM_Model_Server/LLM_Model_Server

source /home/sidhu66/miniconda3/etc/profile.d/conda.sh
# Name of the conda environment
ENV_NAME="lady_llm_env"  # Replace with the actual environment name

# Activate the environment
conda activate "$ENV_NAME"

# To keep it running in background
tmux
# Run the command
CUDA_VISIBLE_DEVICES=0 uvicorn deepseek_server:app --host 0.0.0.0 --port 8000
