#!/bin/bash
#SBATCH --job-name=hermes10
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM
source ~/.secrets/tau_env.sh

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
TAU_DIR=${PROJECT_DIR}/repos/tau-bench
LOG_DIR=${PROJECT_DIR}/logs
VLLM_LOG=${LOG_DIR}/vllm-hermes-${SLURM_JOB_ID}.log

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

cd "${TAU_DIR}"

echo "Started: $(date)"
nvidia-smi

vllm serve "NousResearch/Hermes-2-Pro-Llama-3-8B" \
  --served-model-name "NousResearch/Hermes-2-Pro-Llama-3-8B" \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local-token \
  --generation-config vllm \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

for i in {1..120}; do
  if curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer local-token" | grep -q "Hermes"; then
    echo "vLLM ready: $(date)"
    break
  fi
  sleep 5
done

START_TS=$(date +%s)

tau2 run \
  --domain retail \
  --agent-llm hosted_vllm/NousResearch/Hermes-2-Pro-Llama-3-8B \
  --user-llm gpt-4o-mini \
  --num-trials 1 \
  --num-tasks 10

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "Elapsed seconds: ${ELAPSED}"
python - <<PY
print("Average seconds per episode:", round(${ELAPSED}/10, 2))
PY

kill "${VLLM_PID}" || true
echo "Finished: $(date)"
