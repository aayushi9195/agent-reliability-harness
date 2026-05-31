#!/bin/bash
#SBATCH --job-name=pred-binary
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
# TRAIN BINARY PREDICTOR
# Reads from metrics/episode_results.json (all models, full trajectories).
# Splits 70-15-15 (train/val/test), selects checkpoints by val AUROC,
# reports held-out test AUROC at the end.
# ========================
echo "Starting binary predictor training..."
python train_predictor_binary.py \
    --input   metrics/episode_results.json \
    --output-dir results/predictor-binary \
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
