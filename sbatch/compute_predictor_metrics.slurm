#!/bin/bash
#SBATCH --job-name=pred-metrics
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
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

mkdir -p $TRITON_CACHE_DIR $TORCH_HOME /gpfs/projects/imt526a/group2/agent-reliability/logs

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
cd $PROJECT_DIR

echo "==== JOB CONFIG ===="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Start    : $(date)"
echo "GPU(s)   : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================="

# MODEL_DIR can be overridden at submission time:
#   sbatch --export=ALL,MODEL_DIR=results/predictor-early-3 compute_predictor_metrics.slurm
MODEL_DIR=${MODEL_DIR:-results/predictor-binary}
OUTPUT=${OUTPUT:-results/predictor_metrics_full.json}
PREFIX_TOOL_CALLS=${PREFIX_TOOL_CALLS:-3}

echo "Model dir         : $MODEL_DIR"
echo "Output            : $OUTPUT"
echo "Prefix tool calls : $PREFIX_TOOL_CALLS"
echo "============================="

# ========================
# Compute per-model AUROC, F1, and compute-saved %
# ========================
echo "Running predictor metrics..."
python compute_predictor_metrics.py \
    --input              metrics/episode_results.json \
    --model-dir          $MODEL_DIR \
    --output             $OUTPUT \
    --prefix-tool-calls  $PREFIX_TOOL_CALLS \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --max-length 2048 \
    --batch-size 4

echo "============================="
echo "Job finished at: $(date)"
