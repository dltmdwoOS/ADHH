#!/usr/bin/env bash
set -euo pipefail

# Run the LLaVA-1.5-7B CHAIR baseline ablations used for comparison:
#   1) PAI,    max_new_tokens=128, alpha=0.5, layers 2..31
#   2) PAI-CD, max_new_tokens=128, alpha=0.5, gamma=1.1, layers 2..31
#   3) VAF,    max_new_tokens=128, enh=1.15, sup=0.95, layers 9..14
#
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/_paths.sh"

export MODEL_NAME="${MODEL_NAME:-llava-v1.5-7b}"
export MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
export DATASET="${DATASET:-coco}"
export DATA_PATH="${DATA_PATH:-$(adhh_default_data_path)}"
export RESULTS_ROOT="${RESULTS_ROOT:-$(adhh_default_results_root)}"
export PYTHON_BIN="${PYTHON_BIN:-$(adhh_python_bin)}"
export GPU="${GPU:-0}"
export SEED="${SEED:-42}"
export NUM_SAMPLES="${NUM_SAMPLES:-500}"
export MAX_NEW_TOKENS=128

# Default to resume so this script is safe to re-run after an interruption.
export RESUME="${RESUME:-true}"

echo "[pai_baselines] model=${MODEL_NAME} gpu=${GPU} samples=${NUM_SAMPLES} max_new_tokens=${MAX_NEW_TOKENS} resume=${RESUME}"
echo "[pai_baselines] results_root=${RESULTS_ROOT}"

echo "[pai_baselines] running PAI alpha=0.5 layers=2..31"
env \
  PAI_USE_CFG=false \
  PAI_ALPHA=0.5 \
  PAI_START_LAYER=2 \
  PAI_END_LAYER=32 \
  bash "${script_dir}/chair_pai.sh"

echo "[pai_baselines] running PAI-CD alpha=0.5 gamma=1.1 layers=2..31"
env \
  PAI_USE_CFG=true \
  PAI_ALPHA=0.5 \
  PAI_GAMMA=1.1 \
  PAI_START_LAYER=2 \
  PAI_END_LAYER=32 \
  bash "${script_dir}/chair_pai.sh"

echo "[pai_baselines] running VAF enh=1.15 sup=0.95 layers=9..14"
env \
  VAF_ENH_PARA=1.15 \
  VAF_SUP_PARA=0.95 \
  VAF_START_LAYER=9 \
  VAF_END_LAYER=15 \
  bash "${script_dir}/chair_vaf.sh"

echo "[pai_baselines] all done"
