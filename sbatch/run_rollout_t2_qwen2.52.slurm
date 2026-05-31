#!/bin/bash
#SBATCH --job-name=Rollout-Qwen2.5-tau2
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM

source ~/.secrets/tau_env.sh
export LITELLM_DISABLE_COST_CALC=true
export LITELLM_LOCAL_MODEL_COST_MAP=True

# Cache locations
export TRITON_CACHE_DIR=/gpfs/projects/imt526a/group2/.triton_cache
export TORCH_HOME=/gpfs/projects/imt526a/group2/.torch_cache
export HF_HOME=/gpfs/projects/imt526a/group2/.hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME
export UV_CACHE_DIR=/gpfs/projects/imt526a/group2/uv-cache
export PIP_CACHE_DIR=/gpfs/projects/imt526a/group2/pip-cache
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_HOME" "$HF_HOME" "$HF_HUB_CACHE"

# vLLM endpoint (PLAIN STRING, no HTML)
export HOSTED_VLLM_API_BASE="http://127.0.0.1:8000/v1"
export HOSTED_VLLM_API_KEY="local-token"

# Params (override via sbatch --export=...)
MODEL_NAME=${MODEL_NAME:-"Qwen2.5b"}
MODEL_PATH=${MODEL_PATH:-"/gpfs/projects/imt526a/group2/agent-reliability/models/qwen2.5-7b-instruct"}
SEED=${SEED:-10}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-165}
FAULT_MODE=${FAULT_MODE:-"clean"}   # clean | light | heavy | schema
export FAULT_MODE

JOB_ID=${SLURM_JOB_ID}
PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
RESULT_DIR=$PROJECT_DIR/results/${MODEL_NAME}/${FAULT_MODE}/seed_${SEED}
RESULT_TAU=${MODEL_NAME}/${FAULT_MODE}/seed_${SEED}_${START_IDX}_${END_IDX}_${JOB_ID}_

mkdir -p "$RESULT_DIR"
cd "$PROJECT_DIR"

echo "==== JOB CONFIG ===="
echo "Model: $MODEL_NAME"
echo "Fault: $FAULT_MODE"
echo "Seed: $SEED"
echo "Range: $START_IDX -> $END_IDX"
echo "Output: $RESULT_DIR"

echo "Starting vLLM..."
vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local-token \
  --generation-config vllm \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "$RESULT_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM..."
READY=0
for i in {1..180}; do
  if curl -s http://127.0.0.1:8000/v1/models \
    -H "Authorization: Bearer local-token" \
    | grep -q "\"id\":\"$MODEL_NAME\""; then
    echo "vLLM ready"
    READY=1
    break
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "vLLM failed to start"
  cat "$RESULT_DIR/vllm.log"
  exit 1
fi

echo "Available models:"
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer local-token" || true

NUM_TASKS=$((END_IDX - START_IDX + 1))
echo "Computed task count: $NUM_TASKS"

echo "Starting tau2-bench Qwen 2.5 calibration..."
START_TS=$(date +%s)

tau2 run \
  --domain retail \
  --agent-llm "hosted_vllm/$MODEL_NAME" \
  --user-llm gpt-4o-mini \
  --agent-llm-args "{\"api_base\": \"$HOSTED_VLLM_API_BASE\", \"api_key\": \"$HOSTED_VLLM_API_KEY\", \"temperature\": 0.0, \"top_p\": 1.0, \"seed\": $SEED, \"tool_choice\": \"none\"}" \
  --save-to "$RESULT_TAU" \
  --num-trials 1 \
  --num-tasks "$NUM_TASKS" \
  --max-concurrency 1 \
  --max-steps 300

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

python - <<PY
elapsed = ${ELAPSED}
episodes = ${NUM_TASKS}
print(f"Average seconds per episode: {elapsed / episodes:.2f}")
PY

echo "Stopping vLLM..."
kill "$VLLM_PID" || true
echo "Job finished at: $(date)"
