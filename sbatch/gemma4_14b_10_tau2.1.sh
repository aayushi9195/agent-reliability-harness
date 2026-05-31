#!/bin/bash
#SBATCH --job-name=gemma-4-31B-i
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
VLLM_LOG=${LOG_DIR}/vllm-gemma-${SLURM_JOB_ID}.log

mkdir -p "${LOG_DIR}" /gpfs/projects/imt526a/group2/cache/{triton,torchinductor,cuda,vllm} /gpfs/projects/imt526a/group2/tmp


export MPLCONFIGDIR=/gpfs/projects/imt526a/group2/agent-reliability/cache/matplotlib
export VLLM_CACHE_DIR=/gpfs/projects/imt526a/group2/agent-reliability/cache/vllm_cache
export TRITON_CACHE_DIR=/gpfs/projects/imt526a/group2/agent-reliability/cache/triton_cache
export HF_HOME=/gpfs/projects/imt526a/group2/agent-reliability/cache/huggingface


mkdir -p /gpfs/projects/imt526a/group2/agent-reliability/cache/matplotlib
mkdir -p /gpfs/projects/imt526a/group2/agent-reliability/cache/vllm_cache
mkdir -p /gpfs/projects/imt526a/group2/agent-reliability/cache/triton_cache

export HF_HOME=${PROJECT_DIR}/models
export HF_HUB_CACHE=${PROJECT_DIR}/models
export XDG_CACHE_HOME=/gpfs/projects/imt526a/group2/cache

export TRITON_CACHE_DIR=/gpfs/projects/imt526a/group2/agent-reliability/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/gpfs/projects/imt526a/group2/agent-reliability/cache/torchinductor
export CUDA_CACHE_PATH=/gpfs/projects/imt526a/group2/agent-reliability/cache/cuda
export VLLM_CACHE_ROOT=/gpfs/projects/imt526a/group2/agent-reliability/cache/vllm
export TMPDIR=/gpfs/projects/imt526a/group2/agent-reliability/cache
export VLLM_NO_USAGE_STATS=1
export HOSTED_VLLM_API_BASE=http://127.0.0.1:8000/v1
export HOSTED_VLLM_API_KEY=local-token
export FAULT_MODE

# ========================
# PARAMS (OVERRIDE VIA SBATCH)
# ========================
MODEL_NAME=${MODEL_NAME:-"openai/gemma-4-31B-it"}
MODEL_PATH=${MODEL_PATH:-"/gpfs/projects/imt526a/group2/agent-reliability/models/gemma-4-31B-it"}
SEED=${SEED:-10}
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-165}
FAULT_MODE=${FAULT_MODE:-"clean"}   # clean | light | heavy | schema

JOB_ID=${SLURM_JOB_ID}
PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
RESULT_DIR=$PROJECT_DIR/results/${MODEL_NAME}/${FAULT_MODE}/seed_${SEED}
RESULT_TAU=${MODEL_NAME}/${FAULT_MODE}/seed_${SEED}_${START_IDX}_${END_IDX}_${JOB_ID}_

echo "Results Directory: ${RESULT_TAU}"

cd "${TAU_DIR}"

echo "Started: $(date)"
nvidia-smi

vllm serve $MODEL_PATH \
  --served-model-name $MODEL_NAME \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local-token \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

for i in {1..150}; do
  if curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer local-token" | grep -q "openai/gemma-4-31B-i"; then
    echo "vLLM ready: $(date)"
    break
  fi
  sleep 5
done

START_TS=$(date +%s)

tau2 run \
  --domain retail \
  --agent-llm hosted_vllm/$MODEL_NAME \
  --user-llm gpt-4o-mini \
  --agent-llm-args "{\"api_base\": \"$HOSTED_VLLM_API_BASE\", \"api_key\": \"$HOSTED_VLLM_API_KEY\", \"temperature\": 0.0, \"top_p\": 1.0, \"seed\": $SEED, \"parallel_tool_calls\": false}" 
  --save-to $RESULT_TAU \
  --num-trials 1 \
  --num-tasks 10 \
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
