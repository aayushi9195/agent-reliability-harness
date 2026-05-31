#!/bin/bash
#SBATCH --job-name=pred-early
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

# ========================
# ENV SETUP
# ========================
module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM

export HF_HOME=/gpfs/home/anjutakh/.cache/huggingface
export HF_HUB_CACHE=/gpfs/projects/imt526a/group2/agent-reliability/models/huggingface/hub
export TRANSFORMERS_CACHE=$HF_HOME

export TRITON_CACHE_DIR=/gpfs/projects/imt526a/group2/.triton_cache
export TORCH_HOME=/gpfs/projects/imt526a/group2/.torch_cache
export UV_CACHE_DIR=/gpfs/projects/imt526a/group2/uv-cache
export PIP_CACHE_DIR=/gpfs/projects/imt526a/group2/pip-cache

mkdir -p $TRITON_CACHE_DIR $TORCH_HOME $HF_HOME /gpfs/projects/imt526a/group2/agent-reliability/logs

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
cd $PROJECT_DIR

# ========================
# JOB INFO
# ========================
echo "==== JOB CONFIG ===="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Start    : $(date)"
echo "GPU(s)   : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================="

# ========================
# STEP 1: Prepare early-trajectory data (first 3 tool-call rounds)
# Truncates full_trajectory in episode_results.json to first 3 tool calls.
# Output: data/predictor_early_3.json  (same schema as episode_results.json)
# ========================
echo "Step 1: Preparing early-trajectory data (prefix=3 tool calls)..."
python prepare_predictor_early_data.py \
    --input  metrics/episode_results.json \
    --output data/predictor_early_3.json \
    --prefix-tool-calls 3

echo "============================="

# ========================
# STEP 2: Train binary predictor on early-trajectory data
# Reads from data/predictor_early_3.json.
# Splits 70-15-15 (train/val/test), selects checkpoints by val AUROC,
# reports held-out test AUROC at the end.
# ========================
echo "Step 2: Training early-trajectory binary predictor..."
python train_predictor_binary.py \
    --input   data/predictor_early_3.json \
    --output-dir results/predictor-early-3 \
    --base-model /gpfs/projects/imt526a/group2/agent-reliability/models/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 8 \
    --learning-rate 2e-4 \
    --lora-r 16 \
    --lora-alpha 32 \
    --max-length 2048 \
    --save-steps 100 \
    --seed 42

echo "============================="
echo "Job finished at: $(date)"
