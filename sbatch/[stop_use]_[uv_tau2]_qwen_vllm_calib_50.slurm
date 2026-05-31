#!/bin/bash
#SBATCH --job-name=qwen-calib-50
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/projects/imt526a/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

# Reset any inherited conda/module state from interactive sessions.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate || true
fi

module purge
module load conda
conda activate /gpfs/projects/imt526a/group2/envLLM

source ~/.secrets/tau_env.sh

export HF_HOME=/gpfs/projects/imt526a/group2/agent-reliability/models/huggingface
export HF_HUB_CACHE=/gpfs/projects/imt526a/group2/agent-reliability/models/huggingface/hub
export TRANSFORMERS_CACHE=/gpfs/projects/imt526a/group2/agent-reliability/models/huggingface
export UV_CACHE_DIR=/gpfs/projects/imt526a/group2/uv-cache
export PIP_CACHE_DIR=/gpfs/projects/imt526a/group2/pip-cache

export HOSTED_VLLM_API_BASE="http://127.0.0.1:8000/v1"
export HOSTED_VLLM_API_KEY="local-token"
export VLLM_USE_V1=0
export VLLM_DISABLE_COMPILE_CACHE=1

PROJECT_DIR=/gpfs/projects/imt526a/group2/agent-reliability
TAU_BENCH_DIR=${PROJECT_DIR}/repos/tau-bench
MODEL_DIR=${PROJECT_DIR}/models/qwen2.5-7b-instruct
MAIN_LOG=${PROJECT_DIR}/logs/qwen-calib-50-${SLURM_JOB_ID}.out
VLLM_LOG=${PROJECT_DIR}/logs/vllm-qwen-${SLURM_JOB_ID}.log
RESULT_DIR=${PROJECT_DIR}/results/qwen_vllm_calib_50_${SLURM_JOB_ID}

mkdir -p ${PROJECT_DIR}/logs ${RESULT_DIR}

cd ${TAU_BENCH_DIR}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Python: $(which python)"
python --version
echo "Project dir: ${PROJECT_DIR}"
echo "Tau2 dir: ${TAU_BENCH_DIR}"
echo "Model dir: ${MODEL_DIR}"
echo "Result dir: ${RESULT_DIR}"
echo "vLLM log: ${VLLM_LOG}"
echo "Checking GPU..."
nvidia-smi

echo "Checking model files..."
ls -lh ${MODEL_DIR} | head

echo "Checking tau2 CLI..."
uv run tau2 --help | head -30

echo "Starting vLLM Qwen server..."

vllm serve ${MODEL_DIR} \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 8000 \
  --api-key local-token \
  --generation-config vllm \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > ${VLLM_LOG} 2>&1 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"
echo "vLLM log: ${VLLM_LOG}"

echo "Waiting for vLLM..."
READY=0
for i in {1..180}; do
  if curl -s http://127.0.0.1:8000/v1/models \
    -H "Authorization: Bearer local-token" | grep -q "Qwen/Qwen2.5-7B-Instruct"; then
    READY=1
    echo "vLLM is ready at $(date)."
    break
  fi

  if ! kill -0 ${VLLM_PID} 2>/dev/null; then
    echo "vLLM process exited early."
    tail -n 160 ${VLLM_LOG}
    exit 1
  fi

  sleep 5
done

if [[ "${READY}" -ne 1 ]]; then
  echo "vLLM failed to become ready."
  tail -n 160 ${VLLM_LOG}
  kill ${VLLM_PID} || true
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

uv run tau2 run \
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


