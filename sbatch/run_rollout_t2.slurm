#!/bin/bash
#SBATCH --job-name=rollout-qwen-tau2
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

# ========================
# ENV SETUP
# ========================
module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM

source ~/.secrets/tau_env.sh
export LITELLM_DISABLE_COST_CALC=true

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
MODEL_NAME=${MODEL_NAME:-"Qwen3-8B"}
MODEL_PATH=${MODEL_PATH:-"/gpfs/projects/imt526a/group2/agent-reliability/models/Qwen3-8B"}
SEED=${SEED:-10}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-165}
FAULT_MODE=${FAULT_MODE:-"clean"}   # clean | light | heavy | schema

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
RESULT_DIR=$PROJECT_DIR/results/${MODEL_NAME}/${FAULT_MODE}/seed_${SEED}

mkdir -p $RESULT_DIR
cd $PROJECT_DIR
mkdir -p $HOME/tau_runs


echo "==== JOB CONFIG ===="
echo "Model: $MODEL_NAME"
echo "Fault: $FAULT_MODE"
echo "Seed: $SEED"
echo "Range: $START_IDX -> $END_IDX"
echo "Output: $RESULT_DIR"
echo "$HOME/tau_runs"

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

echo "Starting tau2-bench 50-task Qwen calibration..."
START_TS=$(date +%s)

tau2 run \
  --domain retail \
  --agent-llm hosted_vllm/$MODEL_NAME \
  --user-llm o4-mini \
  --agent-llm-args "{\"api_base\": \"$HOSTED_VLLM_API_BASE\", \"api_key\": \"$HOSTED_VLLM_API_KEY\", \"temperature\": 0.0}"
  --output-dir $HOME/tau_runs
  --save-to qwen_retail_${SEED}_${START_IDX}_${END_IDX}
  --num-trials 1 \
  --num-tasks 50

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "tau2-bench ended at: $(date)"
echo "Elapsed seconds: ${ELAPSED}"
echo "Elapsed minutes: $((ELAPSED / 60))"

python - <<PY
elapsed = ${ELAPSED}
episodes = 50
print(f"Average seconds per episode: {elapsed / episodes:.2f}")
PY

echo "Stopping vLLM..."
kill "${VLLM_PID}" || true

echo "Job finished at: $(date)"
