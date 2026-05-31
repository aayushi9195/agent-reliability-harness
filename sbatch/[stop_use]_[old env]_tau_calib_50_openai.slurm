#!/bin/bash
#SBATCH --job-name=tau-calib-50
#SBATCH --account=group2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/gpfs/projects/imt526a/ytmao/group2/agent-reliability/logs/%x-%j.out

set -euo pipefail

module load conda
conda activate group2-agent

source ~/.secrets/tau_env.sh

cd /gpfs/projects/imt526a/ytmao/group2/agent-reliability/repos/tau-bench

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Python: $(which python)"
python --version

START_TS=$(date +%s)

python run.py \
  --num-trials 1 \
  --agent-strategy tool-calling \
  --env retail \
  --model gpt-4o-mini \
  --model-provider openai \
  --user-model gpt-4o-mini \
  --user-model-provider openai \
  --user-strategy llm \
  --temperature 0.0 \
  --task-split test \
  --start-index 0 \
  --end-index 50 \
  --max-concurrency 1 \
  --seed 10 \
  --log-dir /gpfs/projects/imt526a/ytmao/group2/agent-reliability/results/calib_50_openai

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo "Job ended at: $(date)"
echo "Elapsed seconds: ${ELAPSED}"
echo "Elapsed minutes: $((ELAPSED / 60))"
echo "Average seconds per episode: $((ELAPSED / 50))"
