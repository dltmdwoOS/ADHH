#!/usr/bin/env bash
set -euo pipefail

# End-to-end layer-list dynamic/DEACT sweep:
# 1) filter txtattn_summary.json to each requested layer list
# 2) rebuild surrogate head pools from the filtered summary
# 3) run dynamic CHAIR captioning/evaluation into the compact results_deact tree
#
# Run from LLaVA/:
#   bash bash_scripts/run_layer_list_dynamic_pipeline.sh
#
# Common overrides:
#   LAYER_SPECS="9:16" TOPK_LIST="100" DYNAMIC_PRESETS="1.0 10.0 1.0" DYNAMIC_TAUS="0.90" DYNAMIC_LATE_TAU="0.80" bash ...
#   EXPERIMENT_GROUP=ablations ABLATION_NAME=q DYNAMIC_PRESETS="1.0 8.0 1.0;1.0 10.0 1.0" bash ...
#   DRY_RUN=true bash ...

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
seed=${SEED:-42}
num_samples=${NUM_SAMPLES:-500}
train_num_samples=${TRAIN_NUM_SAMPLES:-500}
max_new_tokens=${MAX_NEW_TOKENS:-128}
results_root=${RESULTS_ROOT:-./results_deact}
experiment_group=${EXPERIMENT_GROUP:-ablations}
ablation_name=${ABLATION_NAME:-token_length}
shared_image_folder=${SHARED_IMAGE_FOLDER:-./shared_images/seed${seed}_${num_samples}}

# Source summary must contain all heads covering every requested layer list.
# Existing filtered summaries under results_deact/resources are reused when present.
source_summary=${SOURCE_SUMMARY:-./results/${dataset}/${model_name}_base_original_qa_n${train_num_samples}_txtattn_l0_l31_allheads/txtattn_summary.json}

# Space-separated layer specs. Each spec can be a comma-separated explicit list
# (e.g. 9,10,11,12,13,15,16) or a compact range (e.g. 9:16).
# LAYER_RANGES is kept as a backward-compatible alias.
read -r -a layer_specs <<< "${LAYER_SPECS:-${LAYER_RANGES:-9:16}}"

# Dynamic/head-pool settings.
intervention=${INTERVENTION:-late_boost}
head_source=${HEAD_SOURCE:-file}
head_score_key=${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG_signed}
head_score_normalize=${HEAD_SCORE_NORMALIZE:-rank_percentile}
min_head_back_raw=${MIN_HEAD_BACK_RAW:-0.0}
use_head_scores=${USE_HEAD_SCORES:-true}
dynamic_context_mode=${DYNAMIC_CONTEXT_MODE:-ratio_exp}
dynamic_late_boost_start=${DYNAMIC_LATE_BOOST_START:-0}
dynamic_late_boost_end=${DYNAMIC_LATE_BOOST_END:-${max_new_tokens}}
dynamic_late_boost_mode=${DYNAMIC_LATE_BOOST_MODE:-linear}
dynamic_late_tau=${DYNAMIC_LATE_TAU:-0.80}
read -r -a dynamic_redistribute_list <<< "${DYNAMIC_REDISTRIBUTES:-none}"
dynamic_renorm=${DYNAMIC_RENORM:-false}
IFS=';' read -r -a dynamic_tau_list <<< "${DYNAMIC_TAUS:-0.90}"
AUTO_DYNAMIC_TAU=${AUTO_DYNAMIC_TAU:-true}
auto_dynamic_tau=${AUTO_DYNAMIC_TAU}
auto_tau_round_step=${AUTO_TAU_ROUND_STEP:-0.01}
auto_tau_round_mode=${AUTO_TAU_ROUND_MODE:-floor}
auto_tau_calibration_scope=${AUTO_TAU_CALIBRATION_SCOPE:-selected_head}
auto_tau_calibration_bucket=${AUTO_TAU_CALIBRATION_BUCKET:-all}
auto_tau_hi_quantile=${AUTO_TAU_HI_QUANTILE:-q66}
auto_tau_lo_quantile=${AUTO_TAU_LO_QUANTILE:-q33}
auto_tau_topk_list=${AUTO_TAU_TOPK_LIST:-100}
read -r -a topk_list <<< "${TOPK_LIST:-100}"
auto_tau_topk=${AUTO_TAU_TOPK:-${topk_list[0]}}

# Presets are separated by semicolon, each preset is: strength exp_sharpness score_power.
IFS=';' read -r -a dynamic_presets <<< "${DYNAMIC_PRESETS:-1.0 8.0 1.0; 1.0 10.0 1.0}"

log_dynamic_trace=${LOG_DYNAMIC_TRACE:-true}
dynamic_trace_topn=${DYNAMIC_TRACE_TOPN:-10}
dynamic_trace_every=${DYNAMIC_TRACE_EVERY:-5}
resume=${RESUME:-true}
dry_run=${DRY_RUN:-false}

read -r -a gpu_list <<< "${GPU_LIST:-0 1}"

sample_dir=${results_root}/${dataset}/${model_name}/shared_samples
sample_id_file=${sample_dir}/val_seed${seed}_n${num_samples}.json
mkdir -p "${sample_dir}"

export PYTHONUNBUFFERED=1
python_bin=${PYTHON_BIN:-python}
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin=python3
fi


if [[ ! -f "${sample_id_file}" ]]; then
  echo "[sample] creating fixed sample id file: ${sample_id_file}"
  if [[ "${dry_run}" == "true" ]]; then
    echo "[dry-run] would create ${sample_id_file}"
  else
    "${python_bin}" - <<PY
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
fi

run_cmd() {
  echo "+ $*"
  if [[ "${dry_run}" != "true" ]]; then
    "$@"
  fi
}

slug_float() {
  "${python_bin}" - "$1" <<'PY'
import sys
value = sys.argv[1]
if value in {"AUTO", "auto"}:
    print("auto")
    raise SystemExit
x = float(value)
if 0 < x < 1:
    s = f"{x:.4f}".rstrip('0')
    decimals = s.split('.', 1)[1] if '.' in s else ''
    if len(decimals) < 2:
        s = f"{x:.2f}"
    print(s.replace('.', ''))
else:
    s = f"{x:.4f}".rstrip('0').rstrip('.')
    print(s.replace('.', '') or "0")
PY
}

update_slug() {
  local redir=$1
  local renorm_flag=$2
  if [[ "${redir}" == "none" && "${renorm_flag}" != "true" ]]; then
    echo "direct"
  elif [[ "${redir}" == "none" && "${renorm_flag}" == "true" ]]; then
    echo "renorm"
  elif [[ "${redir}" == "renorm" ]]; then
    echo "renorm"
  else
    echo "redir_${redir}"
  fi
}

build_result_path() {
  local layer_slug_name=$1
  local topk=$2
  local dynamic_strength=$3
  local dynamic_exp_sharpness=$4
  local dynamic_score_power=$5
  local dynamic_tau=$6
  local dynamic_redistribute=$7
  local dynamic_late_tau_job=$8

  local model_root=${results_root}/${dataset}/${model_name}
  local update_name
  update_name=$(update_slug "${dynamic_redistribute}" "${dynamic_renorm}")
  local q_slug hi_slug lo_slug s_slug p_slug braw_slug
  q_slug=$(slug_float "${dynamic_exp_sharpness}")
  hi_slug=$(slug_float "${dynamic_tau}")
  lo_slug=$(slug_float "${dynamic_late_tau_job}")
  s_slug=$(slug_float "${dynamic_strength}")
  p_slug=$(slug_float "${dynamic_score_power}")
  braw_slug=$(slug_float "${min_head_back_raw}")

  local base_group=${experiment_group}
  if [[ "${base_group}" == "ablation" ]]; then
    base_group="ablations"
  fi

  if [[ "${base_group}" == "main" ]]; then
    echo "${model_root}/main/${layer_slug_name}/k${topk}/${update_name}/tok${max_new_tokens}/q${q_slug}_tau${hi_slug}-${lo_slug}"
  elif [[ "${base_group}" == "ablations" ]]; then
    case "${ablation_name}" in
      q)
        echo "${model_root}/ablations/q/${layer_slug_name}/k${topk}/${update_name}/tok${max_new_tokens}/tau${hi_slug}-${lo_slug}/q${q_slug}"
        ;;
      topk)
        echo "${model_root}/ablations/topk/${layer_slug_name}/${update_name}/tok${max_new_tokens}/q${q_slug}_tau${hi_slug}-${lo_slug}/k${topk}"
        ;;
      max_tokens|tokens)
        echo "${model_root}/ablations/max_tokens/${layer_slug_name}/k${topk}/${update_name}/q${q_slug}_tau${hi_slug}-${lo_slug}/tok${max_new_tokens}"
        ;;
      attention_update|redistribution|redir)
        echo "${model_root}/ablations/attention_update/${layer_slug_name}/k${topk}/tok${max_new_tokens}/q${q_slug}_tau${hi_slug}-${lo_slug}/${update_name}"
        ;;
      tau_schedule|tau)
        echo "${model_root}/ablations/tau_schedule/${layer_slug_name}/k${topk}/${update_name}/tok${max_new_tokens}/q${q_slug}/late_tau${hi_slug}-${lo_slug}"
        ;;
      min_back_raw|back_raw)
        echo "${model_root}/ablations/min_back_raw/${layer_slug_name}/k${topk}/${update_name}/tok${max_new_tokens}/q${q_slug}_tau${hi_slug}-${lo_slug}/bmin${braw_slug}"
        ;;
      *)
        local name=${ablation_name:-custom}
        echo "${model_root}/ablations/${name}/${layer_slug_name}/k${topk}/${update_name}/tok${max_new_tokens}/s${s_slug}_q${q_slug}_tau${hi_slug}-${lo_slug}_p${p_slug}"
        ;;
    esac
  else
    echo "${model_root}/${base_group}/${layer_slug_name}/k${topk}/${update_name}/tok${max_new_tokens}/s${s_slug}_q${q_slug}_tau${hi_slug}-${lo_slug}_p${p_slug}"
  fi
}

layer_slug() {
  "${python_bin}" - "$1" <<'PY'
import sys
spec = sys.argv[1]

def parse(value):
    layers = []
    for part in value.replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            a, b = part.split(':', 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            layers.extend(range(a, b + step, step))
        elif '-' in part and not part.startswith('-'):
            a, b = part.split('-', 1)
            a, b = int(a), int(b)
            step = 1 if b >= a else -1
            layers.extend(range(a, b + step, step))
        else:
            layers.append(int(part))
    seen, out = set(), []
    for layer in layers:
        if layer not in seen:
            out.append(layer)
            seen.add(layer)
    return out

layers = parse(spec)
if not layers:
    raise SystemExit('empty layer spec')
if len(layers) > 1 and layers == list(range(layers[0], layers[-1] + 1)):
    print(f"l{layers[0]}_l{layers[-1]}")
else:
    print('l' + '_l'.join(str(x) for x in layers))
PY
}

prepare_layer_spec() {
  local layer_spec=$1
  local layer_slug_name
  layer_slug_name=$(layer_slug "${layer_spec}")
  local model_root=${results_root}/${dataset}/${model_name}
  local stats_root=${model_root}/resources/${layer_slug_name}_train_n${train_num_samples}
  local filtered_summary=${stats_root}/txtattn_summary.json
  local surrogate_dir=${stats_root}/surrogate_score_zoo

  mkdir -p "${stats_root}" "${surrogate_dir}" "${shared_image_folder}"

  if [[ -f "${filtered_summary}" ]]; then
    echo "[layers ${layer_spec}] reusing filtered summary: ${filtered_summary}"
  else
    if [[ ! -f "${source_summary}" ]]; then
      echo "Missing source summary: ${source_summary}" >&2
      echo "No reusable filtered summary found at: ${filtered_summary}" >&2
      echo "Run decoding_base_with_original_qa.sh for all layers first, or set SOURCE_SUMMARY to an existing all-head txtattn_summary.json." >&2
      exit 1
    fi
    echo "[layers ${layer_spec}] filtering summary -> ${filtered_summary}"
    run_cmd "${python_bin}" eval_scripts/filter_txtattn_summary.py \
      --summary-file "${source_summary}" \
      --output-file "${filtered_summary}" \
      --layers "${layer_spec}"
  fi

  echo "[layers ${layer_spec}] building surrogate score zoo -> ${surrogate_dir}"
  run_cmd "${python_bin}" eval_scripts/compute_surrogate_score_zoo.py \
    --summary-file "${filtered_summary}" \
    --output-dir "${surrogate_dir}"

  local head_file=${surrogate_dir}/ranked_heads_${head_score_key}.json
  if [[ "${dry_run}" != "true" && ! -f "${head_file}" ]]; then
    echo "Missing requested head file after surrogate build: ${head_file}" >&2
    exit 1
  fi

  if [[ "${auto_dynamic_tau}" == "true" ]]; then
    local tau_file=${stats_root}/dynamic_tau_estimate.json
    echo "[layers ${layer_spec}] estimating dynamic tau_hi/tau_lo for top-${auto_tau_topk} -> ${tau_file}"
    run_cmd "${python_bin}" eval_scripts/estimate_dynamic_tau.py \
      --summary-file "${filtered_summary}" \
      --head-file "${head_file}" \
      --topk "${auto_tau_topk}" \
      --topk-list "${auto_tau_topk_list}" \
      --calibration-scope "${auto_tau_calibration_scope}" \
      --calibration-bucket "${auto_tau_calibration_bucket}" \
      --hi-quantile "${auto_tau_hi_quantile}" \
      --lo-quantile "${auto_tau_lo_quantile}" \
      --output-file "${tau_file}" \
      --round-step "${auto_tau_round_step}" \
      --round-mode "${auto_tau_round_mode}"
  fi
}

run_dynamic_job() {
  local layer_spec=$1
  local layer_slug_name=$2
  local gpu=$3
  local topk=$4
  local dynamic_strength=$5
  local dynamic_exp_sharpness=$6
  local dynamic_score_power=$7
  local dynamic_tau=$8
  local dynamic_redistribute=$9
  local dynamic_late_tau_job=${10:-${dynamic_late_tau}}

  local model_root=${results_root}/${dataset}/${model_name}
  local stats_root=${model_root}/resources/${layer_slug_name}_train_n${train_num_samples}
  local head_file=${stats_root}/surrogate_score_zoo/ranked_heads_${head_score_key}.json
  local result_path
  result_path=$(build_result_path "${layer_slug_name}" "${topk}" "${dynamic_strength}" "${dynamic_exp_sharpness}" "${dynamic_score_power}" "${dynamic_tau}" "${dynamic_redistribute}" "${dynamic_late_tau_job}")
  mkdir -p "${result_path}" "${shared_image_folder}"

  local extra_head_args=()
  if [[ "${head_source}" == "file" ]]; then
    extra_head_args+=(
      --head-file "${head_file}"
      --head-score-key "${head_score_key}"
      --head-score-normalize "${head_score_normalize}"
      --min-head-back-raw "${min_head_back_raw}"
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

  local renorm_args=()
  if [[ "${dynamic_renorm}" != "true" ]]; then
    renorm_args+=(--no-dynamic-renorm)
  fi

  echo "[GPU ${gpu}] intervention=${intervention} layers=${layer_spec} topk=${topk} min_back_raw=${min_head_back_raw} s=${dynamic_strength} q=${dynamic_exp_sharpness} tau=${dynamic_tau} p=${dynamic_score_power} redistribute=${dynamic_redistribute} renorm=${dynamic_renorm} late_mode=${dynamic_late_boost_mode} late_start=${dynamic_late_boost_start} late_end=${dynamic_late_boost_end} late_tau=${dynamic_late_tau_job}"

  if [[ "${dry_run}" == "true" ]]; then
    echo "[dry-run] would write ${result_path}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption_dynamic \
    --model-path "${model_path}" \
    --image-folder "${data_path}/coco/val2014" \
    --caption_file_path "${data_path}/coco/annotations/captions_val2014.json" \
    --answers-file "${result_path}/captions.jsonl" \
    --dataset "${dataset}" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --num_samples "${num_samples}" \
    --max_new_tokens "${max_new_tokens}" \
    --copy-image-folder "${shared_image_folder}" \
    --seed "${seed}" \
    --intervention "${intervention}" \
    --head-source "${head_source}" \
    "${extra_head_args[@]}" \
    --topk "${topk}" \
    --dynamic-strength "${dynamic_strength}" \
    --dynamic-context-mode "${dynamic_context_mode}" \
    --dynamic-tau "${dynamic_tau}" \
    --dynamic-exp-sharpness "${dynamic_exp_sharpness}" \
    --dynamic-late-boost-start "${dynamic_late_boost_start}" \
    --dynamic-late-boost-end "${dynamic_late_boost_end}" \
    --dynamic-late-boost-mode "${dynamic_late_boost_mode}" \
    --dynamic-late-tau "${dynamic_late_tau_job}" \
    --dynamic-score-power "${dynamic_score_power}" \
    --dynamic-redistribute "${dynamic_redistribute}" \
    "${renorm_args[@]}" \
    "${score_args[@]}" \
    "${trace_args[@]}" \
    --log-intervention-stats \
    --sample-id-file "${sample_id_file}" \
    "${resume_args[@]}" \
    >> "${result_path}/decode.log" 2>&1

  "${python_bin}" eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${result_path}/captions.jsonl" \
    --caption_file captions_val2014.json \
    > "${result_path}/chair.log" 2>&1

  echo "[GPU ${gpu}] done intervention=${intervention} layers=${layer_spec} topk=${topk} s=${dynamic_strength} q=${dynamic_exp_sharpness} tau=${dynamic_tau} p=${dynamic_score_power} redistribute=${dynamic_redistribute} renorm=${dynamic_renorm} late_mode=${dynamic_late_boost_mode} late_start=${dynamic_late_boost_start} late_end=${dynamic_late_boost_end} late_tau=${dynamic_late_tau_job}"
}

for layer_spec in "${layer_specs[@]}"; do
  prepare_layer_spec "${layer_spec}"
done

pids=()
job_idx=0
for layer_spec in "${layer_specs[@]}"; do
  layer_slug_name=$(layer_slug "${layer_spec}")
  tau_values=("${dynamic_tau_list[@]}")
  layer_dynamic_late_tau="${dynamic_late_tau}"
  if [[ "${auto_dynamic_tau}" == "true" ]]; then
    tau_file=${results_root}/${dataset}/${model_name}/resources/${layer_slug_name}_train_n${train_num_samples}/dynamic_tau_estimate.json
    if [[ "${dry_run}" == "true" && ! -f "${tau_file}" ]]; then
      tau_values=(AUTO)
    else
      tau_values=("$("${python_bin}" - <<PY
import json
with open("${tau_file}") as f:
    data=json.load(f)
print(data.get("recommended_tau_hi_str", data["recommended_tau_str"]))
PY
)")
      layer_dynamic_late_tau="$("${python_bin}" - <<PY
import json
with open("${tau_file}") as f:
    data=json.load(f)
print(data.get("recommended_tau_lo_str", data.get("recommended_tau_str", "${dynamic_late_tau}")))
PY
)"
    fi
  fi

  for preset in "${dynamic_presets[@]}"; do
    read -r dynamic_strength dynamic_exp_sharpness dynamic_score_power <<< "${preset}"
    for dynamic_tau in "${tau_values[@]}"; do
      for dynamic_redistribute in "${dynamic_redistribute_list[@]}"; do
        for topk in "${topk_list[@]}"; do
          gpu="${gpu_list[$((job_idx % ${#gpu_list[@]}))]}"
          run_dynamic_job "${layer_spec}" "${layer_slug_name}" "${gpu}" "${topk}" "${dynamic_strength}" "${dynamic_exp_sharpness}" "${dynamic_score_power}" "${dynamic_tau}" "${dynamic_redistribute}" "${layer_dynamic_late_tau}" &
          pids+=("$!")
          job_idx=$((job_idx + 1))

          if (( ${#pids[@]} == ${#gpu_list[@]} )); then
            wait "${pids[@]}"
            pids=()
          fi
        done
      done
    done
  done
done

if (( ${#pids[@]} > 0 )); then
  wait "${pids[@]}"
fi

echo "Layer-list dynamic pipeline finished."
