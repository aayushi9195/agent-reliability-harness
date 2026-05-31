#!/bin/bash
#SBATCH --job-name=qwen-tau2-50
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

# Avoid inherited conda/module state from interactive sessions.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate || true
fi

module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM

source ~/.secrets/tau_env.sh

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
TAU_BENCH_DIR=${PROJECT_DIR}/repos/tau-bench
MODEL_DIR=${PROJECT_DIR}/models/qwen2.5-7b-instruct
LOG_DIR=${PROJECT_DIR}/logs
RESULT_DIR=${PROJECT_DIR}/results/qwen_tau2_calib_50_${SLURM_JOB_ID}
VLLM_LOG=${LOG_DIR}/vllm-qwen-${SLURM_JOB_ID}.log

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
mkdir -p /gpfs/projects/imt526a/group2/cache/triton
mkdir -p /gpfs/projects/imt526a/group2/cache/torchinductor
mkdir -p /gpfs/projects/imt526a/group2/cache/cuda
mkdir -p /gpfs/projects/imt526a/group2/cache/vllm
mkdir -p /gpfs/projects/imt526a/group2/tmp

# Keep caches and temp files out of home directories.
export HF_HOME=${PROJECT_DIR}/models/huggingface
export HF_HUB_CACHE=${PROJECT_DIR}/models/huggingface/hub
export TRANSFORMERS_CACHE=${PROJECT_DIR}/models/huggingface
export XDG_CACHE_HOME=/gpfs/projects/imt526a/group2/cache
export TRITON_CACHE_DIR=/gpfs/projects/imt526a/group2/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/gpfs/projects/imt526a/group2/cache/torchinductor
export CUDA_CACHE_PATH=/gpfs/projects/imt526a/group2/cache/cuda
export VLLM_CACHE_ROOT=/gpfs/projects/imt526a/group2/cache/vllm
export TMPDIR=/gpfs/projects/imt526a/group2/tmp
export VLLM_NO_USAGE_STATS=1

# LiteLLM hosted_vllm connection.
export HOSTED_VLLM_API_BASE=http://127.0.0.1:8000/v1
export HOSTED_VLLM_API_KEY=local-token

cd "${TAU_BENCH_DIR}"

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python: $(which python)"
python --version
echo "tau2 path: $(which tau2)"
tau2 --help | head -20
echo "Project dir: ${PROJECT_DIR}"
echo "Tau2 dir: ${TAU_BENCH_DIR}"
echo "Model dir: ${MODEL_DIR}"
echo "Result dir: ${RESULT_DIR}"
echo "vLLM log: ${VLLM_LOG}"

echo "Checking GPU..."
nvidia-smi

echo "Checking model files..."
ls -lh "${MODEL_DIR}" | head

echo "Starting vLLM Qwen server..."
vllm serve "${MODEL_DIR}" \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
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
echo "vLLM PID: ${VLLM_PID}"

echo "Waiting for vLLM..."
READY=0
for i in {1..180}; do
  if curl -s http://127.0.0.1:8000/v1/models \
    -H "Authorization: Bearer local-token" | grep -q "Qwen/Qwen2.5-7B-Instruct"; then
    READY=1
    echo "vLLM is ready at $(date)."
    break
  fi

  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM process exited early."
    tail -n 160 "${VLLM_LOG}"
    exit 1
  fi

  sleep 5
done

if [[ "${READY}" -ne 1 ]]; then
  echo "vLLM failed to become ready."
  tail -n 160 "${VLLM_LOG}"
  kill "${VLLM_PID}" || true
  exit 1
fi

echo "vLLM model endpoint:"
curl -s http://127.0.0.1:8000/v1/models -H "Authorization: Bearer local-token"
echo ""

echo "Testing local Qwen endpoint..."
curl -s -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-token" \
  --data '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
    "max_tokens": 30
  }'
echo ""

echo "Starting tau2-bench 50-task Qwen calibration..."
START_TS=$(date +%s)

tau2 run \
  --domain retail \
  --agent-llm hosted_vllm/Qwen/Qwen2.5-7B-Instruct \
  --user-llm gpt-4o-mini \
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
