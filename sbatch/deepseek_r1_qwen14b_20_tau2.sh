#!/bin/bash
#SBATCH --job-name=deepseek14b-20
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=05:00:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    kill "${VLLM_PID}" || true
  fi
}
trap cleanup EXIT

module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM
source ~/.secrets/tau_env.sh

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
TAU_DIR=${PROJECT_DIR}/repos/tau-bench
LOG_DIR=${PROJECT_DIR}/logs
VLLM_LOG=${LOG_DIR}/vllm-deepseek-r1-qwen14b-${SLURM_JOB_ID}.log

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
export LOGURU_LEVEL=CRITICAL
export FAULT_MODE=clean
# Keep OpenAI env untouched so gpt-4o-mini user simulator still uses real OpenAI
export HOSTED_VLLM_API_BASE=http://127.0.0.1:8000/v1

cd "${TAU_DIR}"

echo "Started: $(date)"
nvidia-smi

vllm serve "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --served-model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --generation-config vllm \
  --enforce-eager \
  --max-model-len 32768 \
  > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

READY=0
for i in {1..180}; do
  if curl -s http://127.0.0.1:8000/v1/models | grep -q "DeepSeek-R1-Distill-Qwen-14B"; then
    READY=1
    echo "vLLM ready: $(date)"
    break
  fi

  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited early."
    tail -n 180 "${VLLM_LOG}"
    exit 1
  fi

  sleep 5
done

if [[ "${READY}" -ne 1 ]]; then
  echo "vLLM did not become ready."
  tail -n 180 "${VLLM_LOG}"
  exit 1
fi

echo "Smoke test:"
curl -s -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 64
  }'
echo ""

START_TS=$(date +%s)

tau2 run \
  --domain retail \
  --agent-llm hosted_vllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --user-llm gpt-4o-mini \
  --num-trials 1 \
  --num-tasks 10 \
  --max-steps 300

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "Elapsed seconds: ${ELAPSED}"
python - <<PY
print("Average seconds per episode:", round(${ELAPSED}/20, 2))
PY

echo "Finished: $(date)"
