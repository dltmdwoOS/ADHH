#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-llama3-llava-next-8b}
model_path=${MODEL_PATH:-lmms-lab/llama3-llava-next-8b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
seed=${SEED:-42}
num_samples=${NUM_SAMPLES:-500}
min_layer=${MIN_LAYER:-0}
max_layer=${MAX_LAYER:-31}
result_root=${RESULT_ROOT:-./results_dynamic}
trace_root=${TRACE_ROOT:-./results/${dataset}/${model_name}_base_original_qa_n3000_txtattn_l${min_layer}_l${max_layer}_allheads}

head_source=${HEAD_SOURCE:-file}
head_score_key=${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG}
head_file=${HEAD_FILE:-${trace_root}/surrogate_score_zoo/ranked_heads_${head_score_key}.json}
head_score_normalize=${HEAD_SCORE_NORMALIZE:-rank_percentile}
use_head_scores=${USE_HEAD_SCORES:-true}

topk_list=(${TOPK_LIST:-100})
dynamic_context_mode=${DYNAMIC_CONTEXT_MODE:-ratio_exp}
dynamic_redistribute=${DYNAMIC_REDISTRIBUTE:-renorm}
read -r -a dynamic_tau_list <<< "${DYNAMIC_TAUS:-0.90}"
dynamic_presets=("1.0 8.0 1.0")
if [[ -n "${DYNAMIC_PRESETS:-}" ]]; then
  IFS=';' read -r -a dynamic_presets <<< "${DYNAMIC_PRESETS}"
fi

log_dynamic_trace=${LOG_DYNAMIC_TRACE:-true}
dynamic_trace_topn=${DYNAMIC_TRACE_TOPN:-10}
dynamic_trace_every=${DYNAMIC_TRACE_EVERY:-5}
resume=${RESUME:-true}
gpu_list_raw=${GPU_LIST:-"0,1"}
read -r -a gpu_list <<< "${gpu_list_raw}"
load_4bit=${LOAD_4BIT:-false}
load_8bit=${LOAD_8BIT:-false}
device_map=${DEVICE_MAP:-auto}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

sample_dir=./results/${dataset}/shared_samples
sample_id_file=${sample_dir}/val_seed${seed}_n${num_samples}.json
mkdir -p "${sample_dir}"

if [[ ! -f "${sample_id_file}" ]]; then
  python3 - <<PY2
import json, random
from pycocotools.coco import COCO
caption_file = "${data_path}/coco/annotations/captions_val2014.json"
random.seed(${seed})
coco = COCO(caption_file)
sampled = random.sample(coco.getImgIds(), ${num_samples})
with open("${sample_id_file}", "w") as f:
    json.dump(sampled, f, indent=2)
print("saved sample ids -> ${sample_id_file}")
PY2
fi

export PYTHONUNBUFFERED=1

run_job() {
  local gpu=$1
  local topk=$2
  local dynamic_strength=$3
  local dynamic_exp_sharpness=$4
  local dynamic_score_power=$5
  local dynamic_tau=$6

  local redir_suffix=""
  if [[ "${dynamic_redistribute}" != "renorm" ]]; then
    redir_suffix="_redir${dynamic_redistribute}"
  fi
  local result_path=${result_root}/${dataset}/${model_name}_dynamic_${dynamic_context_mode}_${head_source}_k${topk}_s${dynamic_strength}_q${dynamic_exp_sharpness}_tau${dynamic_tau}_p${dynamic_score_power}${redir_suffix}_n${num_samples}_${head_score_key}
  mkdir -p "${result_path}"

  local extra_head_args=()
  if [[ "${head_source}" == "file" ]]; then
    extra_head_args+=(--head-file "${head_file}" --head-score-key "${head_score_key}" --head-score-normalize "${head_score_normalize}")
  fi
  local score_args=()
  if [[ "${use_head_scores}" == "true" ]]; then
    score_args+=(--use-head-scores)
  fi
  local trace_args=()
  if [[ "${log_dynamic_trace}" == "true" ]]; then
    trace_args+=(--log-dynamic-trace --dynamic-trace-topn "${dynamic_trace_topn}" --dynamic-trace-every "${dynamic_trace_every}")
  fi
  local resume_args=()
  if [[ "${resume}" == "true" ]]; then
    resume_args+=(--resume)
  fi
  local load_args=()
  if [[ "${load_4bit}" == "true" ]]; then
    load_args+=(--load-4bit)
  elif [[ "${load_8bit}" == "true" ]]; then
    load_args+=(--load-8bit)
  fi

  echo "[GPU ${gpu}] Dynamic start: topk=${topk}, score=${head_score_key}, s=${dynamic_strength}, q=${dynamic_exp_sharpness}, tau=${dynamic_tau}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m eval_scripts.eval_caption_dynamic \
    --model-path "${model_path}" \
    --device-map "${device_map}" \
    --attn-implementation "${attn_implementation}" \
    "${load_args[@]}" \
    --image-folder "${data_path}/coco/val2014" \
    --caption_file_path "${data_path}/coco/annotations/captions_val2014.json" \
    --answers-file "${result_path}/captions.jsonl" \
    --dataset "${dataset}" \
    --temperature 0 \
    --conv-mode "${CONV_MODE:-llava_llama_3}" \
    --num_samples "${num_samples}" \
    --seed "${seed}" \
    --intervention dynamic \
    --head-source "${head_source}" \
    "${extra_head_args[@]}" \
    --topk "${topk}" \
    --dynamic-strength "${dynamic_strength}" \
    --dynamic-context-mode "${dynamic_context_mode}" \
    --dynamic-tau "${dynamic_tau}" \
    --dynamic-exp-sharpness "${dynamic_exp_sharpness}" \
    --dynamic-score-power "${dynamic_score_power}" \
    --dynamic-redistribute "${dynamic_redistribute}" \
    "${score_args[@]}" \
    "${trace_args[@]}" \
    --log-intervention-stats \
    --sample-id-file "${sample_id_file}" \
    "${resume_args[@]}" \
    > "${result_path}/decode.log" 2>&1

  python eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${result_path}/captions.jsonl" \
    --caption_file captions_val2014.json \
    > "${result_path}/chair.log" 2>&1
  echo "[GPU ${gpu}] Dynamic done -> ${result_path}"
}

pids=()
job_idx=0
for preset in "${dynamic_presets[@]}"; do
  read -r dynamic_strength dynamic_exp_sharpness dynamic_score_power <<< "${preset}"
  for dynamic_tau in "${dynamic_tau_list[@]}"; do
    for topk in "${topk_list[@]}"; do
      gpu="${gpu_list[$((job_idx % ${#gpu_list[@]}))]}"
      run_job "${gpu}" "${topk}" "${dynamic_strength}" "${dynamic_exp_sharpness}" "${dynamic_score_power}" "${dynamic_tau}" &
      pids+=("$!")
      job_idx=$((job_idx + 1))
      if (( ${#pids[@]} == ${#gpu_list[@]} )); then
        wait "${pids[@]}"
        pids=()
      fi
    done
  done
done
if (( ${#pids[@]} > 0 )); then
  wait "${pids[@]}"
fi

echo "Dynamic experiments finished."
