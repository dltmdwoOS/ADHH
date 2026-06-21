#!/usr/bin/env bash
set -euo pipefail

# Threshold sweep wrapper for the head-pool actuator probe.
#
# Run from ADHH/LLaVA/ or the workspace root:
#   bash ADHH/LLaVA/bash_scripts/run_head_pool_threshold_sweep.sh
#
# This keeps the probe on the first target subtoken, which is the paper-facing
# setting for object-token actuation, and sweeps the hard trigger threshold.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LLAVA_DIR}"
source "${SCRIPT_DIR}/_paths.sh"

PYTHON_BIN=${PYTHON_BIN:-$(adhh_python_bin)}
THRESHOLDS=${THRESHOLDS:-"0.4 0.2 0.6 0.8 1.0"}

# Paper-facing defaults. Override any of these from the shell if needed.
POOLS=${POOLS:-layer_matched_random}
AGGREGATE_TARGET_TOKENS=${AGGREGATE_TARGET_TOKENS:-false}
RENORM=${RENORM:-false}
PROBE_THRESHOLDING=${PROBE_THRESHOLDING:-true}
PROBE_SOURCE=${PROBE_SOURCE:-base}
HEAD_SCORE_KEY=${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG_signed}
LAYER_SLUG=${LAYER_SLUG:-l9_l16}
TOPK=${TOPK:-100}
ADHH_TOPK=${ADHH_TOPK:-30}
MAX_PER_BUCKET=${MAX_PER_BUCKET:-200}
RESULTS_ROOT=${RESULTS_ROOT:-$(adhh_default_results_root)/analysis/head_pool_threshold_sweep}
PARALLEL=${PARALLEL:-true}
GPU_LIST=${GPU_LIST:-0 1}

export PYTHON_BIN
export POOLS
export AGGREGATE_TARGET_TOKENS
export RENORM
export PROBE_THRESHOLDING
export PROBE_SOURCE
export HEAD_SCORE_KEY
export LAYER_SLUG
export TOPK
export ADHH_TOPK
export MAX_PER_BUCKET
export RESULTS_ROOT
export PARALLEL
export GPU_LIST

printf '[sweep] thresholds: %s\n' "${THRESHOLDS}"
printf '[sweep] pools     : %s\n' "${POOLS}"
printf '[sweep] output    : %s\n' "${RESULTS_ROOT}"
printf '[sweep] setting   : firstsubtok, renorm=%s, thresholding=%s\n' "${RENORM}" "${PROBE_THRESHOLDING}"

for threshold in ${THRESHOLDS}; do
  printf '\n[sweep] ===== PROBE_THRESHOLD=%s =====\n' "${threshold}"
  PROBE_THRESHOLD="${threshold}" bash bash_scripts/run_head_pool_control_study.sh
done

printf '\n[sweep] complete. Results are under %s\n' "${RESULTS_ROOT}"
