#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-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}"
model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
mmvet_root=${MMVET_ROOT:-../third_party/MM-Vet}
mmvet_data_root=${MMVET_DATA_ROOT:-../dataset/mm-vet}
image_folder=${MMVET_IMAGE_FOLDER:-${mmvet_data_root}/images}
question_file=${MMVET_QUESTION_FILE:-${mmvet_data_root}/mm-vet.json}
seed=${SEED:-42}
gpu=${GPU:-1}
max_samples=${MAX_SAMPLES:-0}
prompt_style=${PROMPT_STYLE:-exact}
prompt_template=${PROMPT_TEMPLATE:-"{question}"}
system_prompt=${MMVET_SYSTEM_PROMPT:-"You are a helpful assistant. Generate a short and concise response to the following image text pair."}

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

result_path=./results_mmvet/${model_name}_adhh_${head_source}_k${adhh_topk}_tau${adhh_threshold}_${sample_suffix}
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

run_official_eval=${RUN_OFFICIAL_EVAL:-false}
auto_skip_generation=${AUTO_SKIP_GENERATION_IF_COMPLETE:-true}
official_eval_command=${MMVET_OFFICIAL_EVAL_COMMAND:-}
official_args=()
if [[ "${run_official_eval}" == "true" ]]; then
  official_args+=(--run-official-eval)
  if [[ "${auto_skip_generation}" == "true" ]]; then
    official_args+=(--skip-generation-if-complete)
  fi
  if [[ -n "${official_eval_command}" ]]; then
    official_args+=(--official-eval-command "${official_eval_command}")
  fi
fi

export PYTHONUNBUFFERED=1

echo "[GPU ${gpu}] MM-Vet AD-HH start: head_source=${head_source}, topk=${adhh_topk}, tau=${adhh_threshold}, samples=${sample_suffix}"

CUDA_VISIBLE_DEVICES="${gpu}" python -m eval_scripts.eval_mmvet \
  --model-path "${model_path}" \
  --image-folder "${image_folder}" \
  --mmvet-root "${mmvet_root}" \
  --mmvet-data-root "${mmvet_data_root}" \
  --question-file "${question_file}" \
  --answers-file "${result_path}/answers.jsonl" \
  --response-file "${result_path}/mmvet_responses.json" \
  --eval-payload-file "${result_path}/mmvet_eval_payload.json" \
  --metrics-file "${result_path}/mmvet_metrics.json" \
  --temperature 0 \
  --conv-mode vicuna_v1 \
  --system-prompt "${system_prompt}" \
  --seed "${seed}" \
  --num-workers 4 \
  --max_new_tokens 128 \
  --max-samples "${max_samples}" \
  --prompt-style "${prompt_style}" \
  --prompt-template "${prompt_template}" \
  --intervention adhh \
  --head-source "${head_source}" \
  "${extra_head_args[@]}" \
  --topk "${adhh_topk}" \
  --text-threshold "${adhh_threshold}" \
  --log-intervention-stats \
  "${official_args[@]}" \
  "${resume_args[@]}" \
  > "${result_path}/decode.log" 2>&1

echo "[GPU ${gpu}] MM-Vet AD-HH done -> ${result_path}"
