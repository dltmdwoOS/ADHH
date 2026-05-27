#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
mmvet_root=${MMVET_ROOT:-../third_party/MM-Vet}
mmvet_data_root=${MMVET_DATA_ROOT:-../dataset/mm-vet}
image_folder=${MMVET_IMAGE_FOLDER:-${mmvet_data_root}/images}
question_file=${MMVET_QUESTION_FILE:-${mmvet_data_root}/mm-vet.json}
seed=${SEED:-42}
max_samples=${MAX_SAMPLES:-0}
prompt_style=${PROMPT_STYLE:-exact}
prompt_template=${PROMPT_TEMPLATE:-"{question}"}
system_prompt=${MMVET_SYSTEM_PROMPT:-"You are a helpful assistant. Generate a short and concise response to the following image text pair."}

if [[ "${max_samples}" == "0" ]]; then
  sample_suffix=full
else
  sample_suffix=n${max_samples}
fi

# default | file
head_source=${HEAD_SOURCE:-file}
# txt_attn_raw_all | combo_mean_txtraw_Cratio | C_txt_img_ratio_hall_minus_nonhall
head_score_key=${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG}
head_file=${HEAD_FILE:-./results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_${head_score_key}.json}
head_score_normalize=${HEAD_SCORE_NORMALIZE:-rank_percentile}
use_head_scores=${USE_HEAD_SCORES:-true}

# Current dynamic default: ratio-conditioned exponential suppression.
dynamic_context_mode=${DYNAMIC_CONTEXT_MODE:-ratio_exp}
dynamic_tau=${DYNAMIC_TAU:-0.90}

# Space-separated lists, e.g. TOPK_LIST="100 150 200" GPU_LIST="0 1".
read -r -a topk_list <<< "${TOPK_LIST:-100}"
read -r -a gpu_list <<< "${GPU_LIST:-0 1}"

# strength exp_sharpness score_power
dynamic_presets=(
  "0.8 4.0 1.0"
)

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

run_job() {
  local gpu=$1
  local topk=$2
  local dynamic_strength=$3
  local dynamic_exp_sharpness=$4
  local dynamic_score_power=$5

  local result_path=./results_mmvet/${model_name}_dynamic_${dynamic_context_mode}_${head_source}_k${topk}_s${dynamic_strength}_q${dynamic_exp_sharpness}_tau${dynamic_tau}_p${dynamic_score_power}_${sample_suffix}_${head_score_key}
  mkdir -p "${result_path}"

  local extra_head_args=()
  if [[ "${head_source}" == "file" ]]; then
    extra_head_args+=(
      --head-file "${head_file}"
      --head-score-key "${head_score_key}"
      --head-score-normalize "${head_score_normalize}"
    )
  fi

  local score_args=()
  if [[ "${use_head_scores}" == "true" ]]; then
    score_args+=(--use-head-scores)
  fi

  echo "[GPU ${gpu}] MM-Vet Dynamic start: topk=${topk}, score=${head_score_key}/${head_score_normalize}, mode=${dynamic_context_mode}, strength=${dynamic_strength}, q=${dynamic_exp_sharpness}, tau=${dynamic_tau}, score_power=${dynamic_score_power}, samples=${sample_suffix}"

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
    --max_new_tokens 256 \
    --max-samples "${max_samples}" \
    --prompt-style "${prompt_style}" \
    --prompt-template "${prompt_template}" \
    --intervention dynamic \
    --head-source "${head_source}" \
    "${extra_head_args[@]}" \
    --topk "${topk}" \
    --dynamic-strength "${dynamic_strength}" \
    --dynamic-context-mode "${dynamic_context_mode}" \
    --dynamic-tau "${dynamic_tau}" \
    --dynamic-exp-sharpness "${dynamic_exp_sharpness}" \
    --dynamic-score-power "${dynamic_score_power}" \
    "${score_args[@]}" \
    --log-intervention-stats \
    "${official_args[@]}" \
    "${resume_args[@]}" \
    > "${result_path}/decode.log" 2>&1

  echo "[GPU ${gpu}] MM-Vet Dynamic done -> ${result_path}"
}

pids=()
job_idx=0
for preset in "${dynamic_presets[@]}"; do
  read -r dynamic_strength dynamic_exp_sharpness dynamic_score_power <<< "${preset}"

  for topk in "${topk_list[@]}"; do
    gpu="${gpu_list[$((job_idx % ${#gpu_list[@]}))]}"

    run_job "${gpu}" "${topk}" "${dynamic_strength}" "${dynamic_exp_sharpness}" "${dynamic_score_power}" &
    pids+=("$!")
    job_idx=$((job_idx + 1))

    if (( ${#pids[@]} == ${#gpu_list[@]} )); then
      wait "${pids[@]}"
      pids=()
    fi
  done
done

if (( ${#pids[@]} > 0 )); then
  wait "${pids[@]}"
fi

echo "MM-Vet Dynamic experiments finished."
