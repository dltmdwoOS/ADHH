#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
mmvet_root=${MMVET_ROOT:-../third_party/MM-Vet}
mmvet_data_root=${MMVET_DATA_ROOT:-../dataset/mm-vet}
image_folder=${MMVET_IMAGE_FOLDER:-${mmvet_data_root}/images}
question_file=${MMVET_QUESTION_FILE:-${mmvet_data_root}/mm-vet.json}
seed=${SEED:-42}
gpu=${GPU:-0}
max_samples=${MAX_SAMPLES:-0}
prompt_style=${PROMPT_STYLE:-exact}
prompt_template=${PROMPT_TEMPLATE:-"{question}"}
system_prompt=${MMVET_SYSTEM_PROMPT:-"You are a helpful assistant. Generate a short and concise response to the following image text pair."}

if [[ "${max_samples}" == "0" ]]; then
  sample_suffix=full
else
  sample_suffix=n${max_samples}
fi

result_path=./results_mmvet/${model_name}_base_${sample_suffix}
mkdir -p "${result_path}"

resume=${RESUME:-false}
resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi

# MM-Vet official scoring usually needs an external evaluator/API setup, so keep
# generation decoupled by default; set RUN_OFFICIAL_EVAL=true to call the local evaluator.
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
  --intervention none \
  "${official_args[@]}" \
  "${resume_args[@]}" \
  > "${result_path}/decode.log" 2>&1

echo "MM-Vet base done -> ${result_path}"
