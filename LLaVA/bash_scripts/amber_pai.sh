#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
amber_root=${AMBER_ROOT:-$(adhh_default_amber_root)}
image_folder=${AMBER_IMAGE_FOLDER:-$(adhh_default_amber_image_folder)}
seed=${SEED:-42}
gpu=${GPU:-0}
python_bin=$(adhh_python_bin)
max_samples=${MAX_SAMPLES:-0}
max_new_tokens=${MAX_NEW_TOKENS:-512}
# VGA baseline protocol uses PAI alpha=0.5 for LLaVA-family models.
alpha=${PAI_ALPHA:-0.5}
gamma=${PAI_GAMMA:-1.1}
use_cfg=${PAI_USE_CFG:-false}
start_layer=${PAI_START_LAYER:-2}
end_layer=${PAI_END_LAYER:-32}
resume=${RESUME:-false}
dry_run=${DRY_RUN:-false}
results_root=${RESULTS_ROOT:-$(adhh_default_results_root)}

if [[ "${max_samples}" == "0" ]]; then
  sample_suffix=full
else
  sample_suffix=n${max_samples}
fi

cfg_suffix=""
if [[ "${use_cfg}" == "true" ]]; then
  cfg_suffix="_cfg${gamma}"
fi
result_path=${RESULT_PATH:-${results_root}/amber/${model_name}/baselines/pai/tok${max_new_tokens}/${sample_suffix}_alpha${alpha}${cfg_suffix}_l${start_layer}-${end_layer}}
mkdir -p "${result_path}"

resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi

official_args=()
if [[ "${RUN_OFFICIAL_EVAL:-true}" == "true" ]]; then
  official_args+=(--run-official-eval)
fi
cfg_args=()
if [[ "${use_cfg}" == "true" ]]; then
  cfg_args+=(--pai-use-cfg --pai-gamma "${gamma}")
fi

export PYTHONUNBUFFERED=1

if [[ "${dry_run}" == "true" ]]; then
  echo "[dry-run] AMBER PAI would write -> ${result_path}"
  echo "[dry-run] amber_root=${amber_root}"
  echo "[dry-run] image_folder=${image_folder}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_amber \
  --model-path "${model_path}" \
  --image-folder "${image_folder}" \
  --amber-root "${amber_root}" \
  --query-file "${amber_root}/data/query/query_generative.json" \
  --answers-file "${result_path}/answers.jsonl" \
  --response-file "${result_path}/amber_responses.json" \
  --metrics-file "${result_path}/amber_metrics.json" \
  --temperature 0 \
  --conv-mode vicuna_v1 \
  --seed "${seed}" \
  --num-workers 4 \
  --max_new_tokens "${max_new_tokens}" \
  --max-samples "${max_samples}" \
  --intervention pai \
  --topk 0 \
  --baseline-start-layer "${start_layer}" \
  --baseline-end-layer "${end_layer}" \
  --pai-alpha "${alpha}" \
  "${cfg_args[@]}" \
  --no-decode-log \
  "${official_args[@]}" \
  "${resume_args[@]}" \
  > "${result_path}/decode.log" 2>&1

echo "AMBER PAI done -> ${result_path}"
