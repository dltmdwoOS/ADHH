#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
amber_root=${AMBER_ROOT:-$(adhh_default_amber_root)}
image_folder=${AMBER_IMAGE_FOLDER:-$(adhh_default_amber_image_folder)}
seed=${SEED:-42}
gpu=${GPU:-0}
max_samples=${MAX_SAMPLES:-0}
max_new_tokens=${MAX_NEW_TOKENS:-512}
results_root=${RESULTS_ROOT:-$(adhh_default_results_root)}
python_bin=${PYTHON_BIN:-python}
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin=python3
fi

if [[ "${max_samples}" == "0" ]]; then
  sample_suffix=full
  result_path=${RESULT_PATH:-${results_root}/amber/${model_name}/baselines/greedy/tok${max_new_tokens}}
else
  sample_suffix=n${max_samples}
  result_path=${RESULT_PATH:-${results_root}/amber/${model_name}/baselines/greedy/tok${max_new_tokens}/${sample_suffix}}
fi

mkdir -p "${result_path}"

resume=${RESUME:-false}
resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi
dry_run=${DRY_RUN:-false}

run_official_eval=${RUN_OFFICIAL_EVAL:-true}
official_args=()
if [[ "${run_official_eval}" == "true" ]]; then
  official_args+=(--run-official-eval)
fi

export PYTHONUNBUFFERED=1

if [[ "${dry_run}" == "true" ]]; then
  echo "[dry-run] AMBER base would write -> ${result_path}"
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
  --intervention none \
  "${official_args[@]}" \
  "${resume_args[@]}" \
  > "${result_path}/decode.log" 2>&1

echo "AMBER generative base done -> ${result_path}"
