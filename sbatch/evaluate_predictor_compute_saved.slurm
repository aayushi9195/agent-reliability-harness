#!/bin/bash
#SBATCH --job-name=pred-compute-save
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
cd "${PROJECT_DIR}"

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

# Defaults can be overridden at submission time, for example:
# sbatch --export=ALL,MAX_FALSE_STOP_RATE=0.10 sbatch/evaluate_predictor_compute_saved.slurm
FULL_INPUT=${FULL_INPUT:-metrics/episode_results.json}
EARLY_INPUT=${EARLY_INPUT:-data/predictor_early_3.json}
MODEL_DIR=${MODEL_DIR:-results/predictor-early-3}
OUTPUT_DIR=${OUTPUT_DIR:-results/predictor-early-3/compute_saved_eval}
PREFIX_TOOL_CALLS=${PREFIX_TOOL_CALLS:-3}
MAX_FALSE_STOP_RATE=${MAX_FALSE_STOP_RATE:-0.05}
MAX_LENGTH=${MAX_LENGTH:-2048}
BATCH_SIZE=${BATCH_SIZE:-4}

echo "==== JOB CONFIG ===="
echo "Job ID              : ${SLURM_JOB_ID}"
echo "Node                : ${SLURMD_NODENAME}"
echo "Start               : $(date)"
echo "Full input          : ${FULL_INPUT}"
echo "Early input         : ${EARLY_INPUT}"
echo "Model dir           : ${MODEL_DIR}"
echo "Output dir          : ${OUTPUT_DIR}"
echo "Prefix tool calls   : ${PREFIX_TOOL_CALLS}"
echo "Max false-stop rate : ${MAX_FALSE_STOP_RATE}"
echo "===================="
nvidia-smi

python sbatch/evaluate_predictor_compute_saved.py \
  --full-input "${FULL_INPUT}" \
  --early-input "${EARLY_INPUT}" \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --prefix-tool-calls "${PREFIX_TOOL_CALLS}" \
  --max-false-stop-rate "${MAX_FALSE_STOP_RATE}" \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42 \
  --max-length "${MAX_LENGTH}" \
  --batch-size "${BATCH_SIZE}"

echo "Finished: $(date)"
