#!/usr/bin/env bash
set -euo pipefail

model_name=llava-v1.5-7b
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
seed=${SEED:-42}
num_samples=${NUM_SAMPLES:-500}
min_layer=${MIN_LAYER:-16}
max_layer=${MAX_LAYER:-31}
topk_list=(100)

# default | file
head_source=${HEAD_SOURCE:-file}
# txt_attn_raw_all | combo_mean_txtraw_Cratio | C_txt_img_ratio_hall_minus_nonhall
head_score_key=${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG}
#head_file=${HEAD_FILE:-./results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/ranked_heads_${head_score_key}.json}
head_file=${HEAD_FILE:-./results/coco/llava-v1.5-7b_base_original_qa_n3000_txtattn_l${min_layer}_l${max_layer}_allheads/surrogate_score_zoo/ranked_heads_${head_score_key}.json}

head_score_normalize=${HEAD_SCORE_NORMALIZE:-rank_percentile}
use_head_scores=${USE_HEAD_SCORES:-true}

# Dynamic: threshold-free continuous score-weighted text suppression.
# text_exp uses c = exp(k * (I_text - tau)); ratio_power recovers the earlier dynamic draft.
dynamic_context_mode=${DYNAMIC_CONTEXT_MODE:-ratio_exp}
dynamic_redistribute=${DYNAMIC_REDISTRIBUTE:-renorm}
read -r -a dynamic_tau_list <<< "${DYNAMIC_TAUS:-0.90}"

# strength exp_sharpness score_power
dynamic_presets=(
  "0.7 8.0 1.0"
  "0.6 10.0 1.0"
)

log_dynamic_trace=${LOG_DYNAMIC_TRACE:-true}
dynamic_trace_topn=${DYNAMIC_TRACE_TOPN:-10}
dynamic_trace_every=${DYNAMIC_TRACE_EVERY:-5}
resume=${RESUME:-true}

gpu_list=(0 1)

sample_dir=./results/${dataset}/shared_samples
sample_id_file=${sample_dir}/val_seed${seed}_n${num_samples}.json
mkdir -p "${sample_dir}"

export PYTHONUNBUFFERED=1

if [[ ! -f "${sample_id_file}" ]]; then
  python - <<PY
import json, random
from pycocotools.coco import COCO

caption_file = "${data_path}/coco/annotations/captions_val2014.json"
seed = ${seed}
num_samples = ${num_samples}
out_file = "${sample_id_file}"

random.seed(seed)
coco = COCO(caption_file)
img_ids = coco.getImgIds()
sampled = random.sample(img_ids, num_samples)

with open(out_file, "w") as f:
    json.dump(sampled, f, indent=2)
print(f"saved sample ids -> {out_file}")
PY
fi

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

  local result_path=./results_l${min_layer}_l${max_layer}/${dataset}/${model_name}_dynamic_${dynamic_context_mode}_${head_source}_k${topk}_s${dynamic_strength}_q${dynamic_exp_sharpness}_tau${dynamic_tau}_p${dynamic_score_power}${redir_suffix}_n${num_samples}_${head_score_key}
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

  local trace_args=()
  if [[ "${log_dynamic_trace}" == "true" ]]; then
    trace_args+=(
      --log-dynamic-trace
      --dynamic-trace-topn "${dynamic_trace_topn}"
      --dynamic-trace-every "${dynamic_trace_every}"
    )
  fi

  local resume_args=()
  if [[ "${resume}" == "true" ]]; then
    resume_args+=(--resume)
  fi

  echo "[GPU ${gpu}] Dynamic start: topk=${topk}, score=${head_score_key}/${head_score_normalize}, mode=${dynamic_context_mode}, redistribute=${dynamic_redistribute}, strength=${dynamic_strength}, exp_sharpness=${dynamic_exp_sharpness}, tau=${dynamic_tau}, score_power=${dynamic_score_power}, use_head_scores=${use_head_scores}, resume=${resume}"

  CUDA_VISIBLE_DEVICES="${gpu}" python -m eval_scripts.eval_caption_dynamic \
    --model-path "${model_path}" \
    --image-folder "${data_path}/coco/val2014" \
    --caption_file_path "${data_path}/coco/annotations/captions_val2014.json" \
    --answers-file "${result_path}/captions.jsonl" \
    --dataset "${dataset}" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
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
    >> "${result_path}/decode.log" 2>&1

  python eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${result_path}/captions.jsonl" \
    --caption_file captions_val2014.json \
    > "${result_path}/chair.log" 2>&1

  echo "[GPU ${gpu}] Dynamic done: topk=${topk}, mode=${dynamic_context_mode}, redistribute=${dynamic_redistribute}, strength=${dynamic_strength}, exp_sharpness=${dynamic_exp_sharpness}, tau=${dynamic_tau}, score_power=${dynamic_score_power}"
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
