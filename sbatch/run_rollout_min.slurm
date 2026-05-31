#!/bin/bash
#SBATCH --job-name=rollout-minstral
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

# ========================
# ENV SETUP
# ========================
module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM

source ~/.secrets/tau_env.sh

export HF_HOME=/gpfs/projects/imt526a/group2/agent-reliability/models/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export UV_CACHE_DIR=/gpfs/projects/imt526a/group2/uv-cache
export PIP_CACHE_DIR=/gpfs/projects/imt526a/group2/pip-cache

# Fix cache locations (critical for Triton)
export TRITON_CACHE_DIR=/gpfs/projects/imt526a/group2/.triton_cache
export TORCH_HOME=/gpfs/projects/imt526a/group2/.torch_cache
export HF_HOME=/gpfs/projects/imt526a/group2/.hf_cache

mkdir -p $TRITON_CACHE_DIR $TORCH_HOME $HF_HOME

export HOSTED_VLLM_API_BASE="http://127.0.0.1:8000/v1"
export HOSTED_VLLM_API_KEY="local-token"

# ========================
# PARAMS (OVERRIDE VIA SBATCH)
# ========================
MODEL_NAME=${MODEL_NAME:-"mistralai/Mistral-7B-Instruct-v0.1"}
MODEL_PATH=${MODEL_PATH:-"/gpfs/projects/imt526a/group2/agent-reliability/models/Mistral-7B-Instruct-v0.1"}
SEED=${SEED:-10}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-165}
FAULT_MODE=${FAULT_MODE:-"clean"}   # clean | light | heavy | schema

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
RESULT_DIR=$PROJECT_DIR/results/${MODEL_NAME}/${FAULT_MODE}/seed_${SEED}

mkdir -p $RESULT_DIR
cd $PROJECT_DIR

echo "==== JOB CONFIG ===="
echo "Model: $MODEL_NAME"
echo "Fault: $FAULT_MODE"
echo "Seed: $SEED"
echo "Range: $START_IDX -> $END_IDX"
echo "Output: $RESULT_DIR"

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
for i in {1..180}; do
  if curl -s http://127.0.0.1:8000/v1/models \
    -H "Authorization: Bearer local-token" \
    | grep -q "$MODEL_NAME"; then
    echo "vLLM ready"
    break
  fi
  sleep 5
done

# ========================
# RUN TAU-BENCH ROLLOUT
# ========================
echo "Starting rollout..."
cd /gpfs/projects/imt526a/group2/agent-reliability/repos/tau-bench

python run.py \
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
  --log-dir $RESULT_DIR

echo "Rollout complete"

# ========================
# CLEANUP
# ========================
kill $VLLM_PID || true

echo "Job finished"
