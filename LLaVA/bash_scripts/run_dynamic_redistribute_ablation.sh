#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

# Ablation for explicit redistribution of the text attention mass removed by dynamic.
# Defaults match the current l13-l20 setting the user is inspecting.
#
# Run from LLaVA/:
#   bash bash_scripts/run_dynamic_redistribute_ablation.sh
#
# Override example:
#   LAYER_RANGES="13:31" DYNAMIC_PRESETS="0.7 8.0 1.0" bash bash_scripts/run_dynamic_redistribute_ablation.sh

export LAYER_RANGES=${LAYER_RANGES:-"9:16"}
export TOPK_LIST=${TOPK_LIST:-"100"}
export DYNAMIC_PRESETS=${DYNAMIC_PRESETS:-"1.0 10.0 1.0"}
export DYNAMIC_TAUS=${DYNAMIC_TAUS:-"0.90"}
export DYNAMIC_LATE_TAU=${DYNAMIC_LATE_TAU:-"0.80"}
export DYNAMIC_REDISTRIBUTES=${DYNAMIC_REDISTRIBUTES:-"system sysvis vision renorm"}
export DYNAMIC_CONTEXT_MODE=${DYNAMIC_CONTEXT_MODE:-"ratio_exp"}
export HEAD_SCORE_KEY=${HEAD_SCORE_KEY:-"global__itext_all__C_toi_HminusG_signed"}
export ABLATION_NAME=${ABLATION_NAME:-"redistribution"}
export EXPERIMENT_GROUP=${EXPERIMENT_GROUP:-"ablations"}
export RESUME=${RESUME:-"true"}

bash "${ADHH_SCRIPT_DIR}/run_layer_list_dynamic_pipeline.sh"
