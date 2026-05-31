#!/bin/bash
#SBATCH --job-name=rollout-qwen
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

# ========================
# ENV SETUP
# ========================
module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM

source ~/.secrets/tau_env.sh

# Disable vLLM usage reporting (to avoid disk quota issues)

export HF_HOME=/gpfs/projects/imt526a/group2/.hf_cache
export TRITON_CACHE_DIR=/gpfs/projects/imt526a/group2/.triton_cache
export TORCH_HOME=/gpfs/projects/imt526a/group2/.torch_cache
mkdir -p $HF_HOME $TRITON_CACHE_DIR $TORCH_HOME

export HOSTED_VLLM_API_BASE="http://127.0.0.1:8000/v1"
export HOSTED_VLLM_API_KEY="local-token"

# ========================
# PARAMS
# ========================
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_PATH="/gpfs/projects/imt526a/group2/agent-reliability/models/qwen2.5-7b-instruct"
SEED=10
START_IDX=0
END_IDX=10
FAULT_MODE="clean"

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
RESULT_DIR=$PROJECT_DIR/results/qwen/${FAULT_MODE}/seed_${SEED}

mkdir -p $RESULT_DIR
cd $PROJECT_DIR

echo "==== JOB CONFIG ===="
echo "Model: $MODEL_NAME"
echo "Seed: $SEED"
echo "Range: $START_IDX -> $END_IDX"

# ========================
# START VLLM
# ========================
echo "Starting vLLM..."

vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local-token \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > $RESULT_DIR/vllm.log 2>&1 &

VLLM_PID=$!

# Wait until ready
echo "Waiting for vLLM..."
READY=0
for i in {1..180}; do
  RESPONSE=$(curl -s -H "Authorization: Bearer local-token" \
    http://127.0.0.1:8000/v1/models 2>/dev/null)

  if echo "$RESPONSE" | grep -q "Qwen/Qwen2.5-7B-Instruct"; then
    echo "✓ vLLM is ready"
    READY=1
    break
  fi
  echo "  Attempt $i/180..."
  sleep 2  # Reduced from 5s to speed up detection
done

if [ $READY -eq 0 ]; then
  echo "ERROR: vLLM never became ready"
  echo "Last response: $RESPONSE"
  kill $VLLM_PID || true
  exit 1
fi

echo "Starting run.py at $(date)..."
echo "API Key check: ${OPENAI_API_KEY:0:20}..."

cd /gpfs/projects/imt526a/group2/agent-reliability/repos/tau-bench

python -u run.py \
  --num-trials 1 \
  --agent-strategy tool-calling \
  --env retail \
  --model hosted_vllm/$MODEL_NAME \
  --model-provider hosted_vllm \
  --user-model gpt-4o-mini \
  --user-model-provider openai \
  --user-strategy llm \
  --temperature 0.0 \
  --task-split test \
  --start-index $START_IDX \
  --end-index $END_IDX \
  --seed $SEED \
  --max-concurrency 1 \
  --log-dir $RESULT_DIR 2>&1 | tee -a $RESULT_DIR/run.log

echo "Rollout complete at $(date)"
