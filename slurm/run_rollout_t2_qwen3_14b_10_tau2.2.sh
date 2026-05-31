#!/bin/bash
#SBATCH --job-name=qwen3-14b-10
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=05:00:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM
source ~/.secrets/tau_env.sh

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
TAU_DIR=${PROJECT_DIR}/repos/tau-bench
LOG_DIR=${PROJECT_DIR}/logs
VLLM_LOG=${LOG_DIR}/vllm-qwen3-14b-${SLURM_JOB_ID}.log

mkdir -p "${LOG_DIR}" /gpfs/projects/imt526a/group2/cache/{triton,torchinductor,cuda,vllm} /gpfs/projects/imt526a/group2/tmp

export HF_HOME=${PROJECT_DIR}/models/huggingface
export HF_HUB_CACHE=${PROJECT_DIR}/models/huggingface/hub
export XDG_CACHE_HOME=/gpfs/projects/imt526a/group2/cache
export TRITON_CACHE_DIR=/gpfs/projects/imt526a/group2/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/gpfs/projects/imt526a/group2/cache/torchinductor
export CUDA_CACHE_PATH=/gpfs/projects/imt526a/group2/cache/cuda
export VLLM_CACHE_ROOT=/gpfs/projects/imt526a/group2/cache/vllm
export TMPDIR=/gpfs/projects/imt526a/group2/tmp
export VLLM_NO_USAGE_STATS=1
export HOSTED_VLLM_API_BASE=http://127.0.0.1:8000/v1
export HOSTED_VLLM_API_KEY=local-token
export FAULT_MODE
export OPENAI_DISABLE_PARALLEL_TOOL_CALLS=true

export LITELLM_USE_FUNCTION_CALLING=True
export LITELLM_JSON_MODE=True
export LITELLM_FORCE_OPENAI_SCHEMA=True
export LITELLM_DROP_INVALID_TOOL_CALLS=True

export LOGURU_LEVEL=CRITICAL

# ========================
# PARAMS (OVERRIDE VIA SBATCH)
# ========================
MODEL_NAME=${MODEL_NAME:-"Qwen3-14B"}
MODEL_PATH=${MODEL_PATH:-"/gpfs/projects/imt526a/group2/agent-reliability/models/Qwen3-14B"}
SEED=${SEED:-10}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-165}
FAULT_MODE=${FAULT_MODE:-"light"}   # clean | light | heavy | schema

JOB_ID=${SLURM_JOB_ID}
PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
RESULT_DIR=$PROJECT_DIR/results/${MODEL_NAME}/${FAULT_MODE}/seed_${SEED}
RESULT_TAU=${MODEL_NAME}/${FAULT_MODE}/seed_${SEED}_${START_IDX}_${END_IDX}_${JOB_ID}_

echo "Results Directory: ${RESULT_TAU}"

git config --global --add safe.directory "${TAU_DIR}" || true

cd "${TAU_DIR}"

echo "Started: $(date)"
nvidia-smi

vllm serve "Qwen/Qwen3-14B" \
  --served-model-name "Qwen/Qwen3-14B" \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local-token \
  --generation-config vllm \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --reasoning-parser qwen3 \
  --chat-template-content-format auto \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

for i in {1..150}; do
  if curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer local-token" | grep -q "Qwen/Qwen3-14B"; then
    echo "vLLM ready: $(date)"
    break
  fi
  sleep 5
done

START_TS=$(date +%s)

tau2 run \
  --domain retail \
  --agent-llm hosted_vllm/Qwen/Qwen3-14B \
  --user-llm gpt-4o-mini \
  --agent-llm-args "{\"api_base\": \"$HOSTED_VLLM_API_BASE\", \"api_key\": \"$HOSTED_VLLM_API_KEY\", \"temperature\": 0.0, \"top_p\": 1.0, \"seed\": $SEED, \"parallel_tool_calls\": false,  \"tool_choice\": \"auto\"}" \
  --save-to $RESULT_TAU \
  --num-trials 1 \
  --num-tasks 203 \
  --max-concurrency 1 \
  --max-steps 300


END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "Elapsed seconds: ${ELAPSED}"
python - <<PY
print("Average seconds per episode:", round(${ELAPSED}/10, 2))
PY

kill "${VLLM_PID}" || true
echo "Finished: $(date)"
