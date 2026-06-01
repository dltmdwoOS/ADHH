#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
amber_root=${AMBER_ROOT:-../third_party/AMBER}
image_folder=${AMBER_IMAGE_FOLDER:-../dataset/AMBER/images}
seed=${SEED:-42}
gpu=${GPU:-1}
max_samples=${MAX_SAMPLES:-0}

# Paper-style AD-HH defaults: top-20 heads and tau=0.4.
adhh_topk=${ADHH_TOPK:-20}
adhh_threshold=${ADHH_THRESHOLD:-0.4}

# default uses the built-in AD-HH head list; file uses a saved attribution file.
head_source=${HEAD_SOURCE:-file}
head_file=${HEAD_FILE:-./results/coco/llava-v1.5-7b_base_original_qa_n3000/identify_attention_head/attribution_result.json}

if [[ "${max_samples}" == "0" ]]; then
  sample_suffix=full
else
  sample_suffix=n${max_samples}
fi

result_path=./results_amber/generative/${model_name}_adhh_${head_source}_k${adhh_topk}_tau${adhh_threshold}_${sample_suffix}
mkdir -p "${result_path}"

extra_head_args=()
if [[ "${head_source}" == "file" ]]; then
  extra_head_args+=(--head-file "${head_file}")
fi

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

echo "[GPU ${gpu}] AMBER AD-HH start: head_source=${head_source}, topk=${adhh_topk}, tau=${adhh_threshold}, samples=${sample_suffix}"

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
  --max_new_tokens 128 \
  --max-samples "${max_samples}" \
  --intervention adhh \
  --head-source "${head_source}" \
  "${extra_head_args[@]}" \
  --topk "${adhh_topk}" \
  --text-threshold "${adhh_threshold}" \
  --log-intervention-stats \
  "${official_args[@]}" \
  "${resume_args[@]}" \
  > "${result_path}/decode.log" 2>&1

echo "[GPU ${gpu}] AMBER AD-HH done -> ${result_path}"
