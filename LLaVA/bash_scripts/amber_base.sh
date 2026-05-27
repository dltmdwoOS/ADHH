#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
amber_root=${AMBER_ROOT:-../third_party/AMBER}
image_folder=${AMBER_IMAGE_FOLDER:-../dataset/AMBER/images}
seed=${SEED:-42}
gpu=${GPU:-0}
max_samples=${MAX_SAMPLES:-0}

if [[ "${max_samples}" == "0" ]]; then
  sample_suffix=full
else
  sample_suffix=n${max_samples}
fi

result_path=./results_amber/generative/${model_name}_base_${sample_suffix}
mkdir -p "${result_path}"

resume=${RESUME:-false}
resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi

run_official_eval=${RUN_OFFICIAL_EVAL:-true}
official_args=()
if [[ "${run_official_eval}" == "true" ]]; then
  official_args+=(--run-official-eval)
fi

export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES="${gpu}" python -m eval_scripts.eval_amber \
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
  --max_new_tokens 256 \
  --max-samples "${max_samples}" \
  --intervention none \
  "${official_args[@]}" \
  "${resume_args[@]}" \
  > "${result_path}/decode.log" 2>&1

echo "AMBER generative base done -> ${result_path}"
