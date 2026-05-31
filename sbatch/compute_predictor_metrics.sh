#!/bin/bash
#SBATCH --job-name=pred-metrics
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability

mkdir -p "${PROJECT_DIR}/logs" \
         /gpfs/projects/imt526a/group2/cache/{triton,torchinductor,cuda} \
         /gpfs/projects/imt526a/group2/tmp

export HF_HOME=${PROJECT_DIR}/models/huggingface
export HF_HUB_CACHE=${PROJECT_DIR}/models/huggingface/hub
export XDG_CACHE_HOME=/gpfs/projects/imt526a/group2/cache
export TRITON_CACHE_DIR=/gpfs/projects/imt526a/group2/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/gpfs/projects/imt526a/group2/cache/torchinductor
export CUDA_CACHE_PATH=/gpfs/projects/imt526a/group2/cache/cuda
export TMPDIR=/gpfs/projects/imt526a/group2/tmp

cd "${PROJECT_DIR}"

echo "Started: $(date)"
nvidia-smi

python compute_predictor_metrics.py \
  --input    metrics/episode_results.json \
  --model-dir results/predictor-binary \
  --output   results/predictor_metrics_full.json \
  --prefix-tool-calls 3 \
  --seed 42

echo "Finished: $(date)"
