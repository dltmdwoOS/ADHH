#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-llama-3-vila1.5-8b}
model_path=${MODEL_PATH:-Efficient-Large-Model/Llama-3-VILA1.5-8B}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
seed=${SEED:-42}
num_samples=${NUM_SAMPLES:-500}
train_num_samples=${TRAIN_NUM_SAMPLES:-500}

source_summary=${SOURCE_SUMMARY:-./results/${dataset}/${model_name}_base_original_qa_n${train_num_samples}_txtattn_l0_l31_allheads/txtattn_summary.json}
read -r -a layer_specs <<< "${LAYER_SPECS:-9:16}"

intervention=${INTERVENTION:-late_boost}
head_source=${HEAD_SOURCE:-file}
head_score_key=${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG_signed}
head_score_normalize=${HEAD_SCORE_NORMALIZE:-rank_percentile}
min_head_back_raw=${MIN_HEAD_BACK_RAW:-0.0}
use_head_scores=${USE_HEAD_SCORES:-true}
dynamic_context_mode=${DYNAMIC_CONTEXT_MODE:-ratio_exp}
dynamic_late_boost_start=${DYNAMIC_LATE_BOOST_START:-0}
dynamic_late_boost_end=${DYNAMIC_LATE_BOOST_END:-128}
dynamic_late_boost_mode=${DYNAMIC_LATE_BOOST_MODE:-linear}
dynamic_late_tau=${DYNAMIC_LATE_TAU:-0.82}
read -r -a dynamic_redistribute_list <<< "${DYNAMIC_REDISTRIBUTES:-none}"
dynamic_renorm=${DYNAMIC_RENORM:-false}
IFS=';' read -r -a dynamic_tau_list <<< "${DYNAMIC_TAUS:-0.90}"
AUTO_DYNAMIC_TAU=${AUTO_DYNAMIC_TAU:-false}
auto_tau_round_step=${AUTO_TAU_ROUND_STEP:-0.01}
auto_tau_round_mode=${AUTO_TAU_ROUND_MODE:-floor}
read -r -a topk_list <<< "${TOPK_LIST:-100}"
IFS=';' read -r -a dynamic_presets <<< "${DYNAMIC_PRESETS:-1.0 8.0 1.0; 1.0 10.0 1.0;}"

log_dynamic_trace=${LOG_DYNAMIC_TRACE:-true}
dynamic_trace_topn=${DYNAMIC_TRACE_TOPN:-10}
dynamic_trace_every=${DYNAMIC_TRACE_EVERY:-5}
resume=${RESUME:-true}
dry_run=${DRY_RUN:-false}
attn_implementation=${VILA_ATTN_IMPLEMENTATION:-eager}
read -r -a gpu_list <<< "${GPU_LIST:-0 1}"

sample_dir=./results/${dataset}/shared_samples
sample_id_file=${sample_dir}/val_seed${seed}_n${num_samples}.json
mkdir -p "${sample_dir}"

export PYTHONUNBUFFERED=1
python_bin=${PYTHON_BIN:-python}
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin=python3
fi

if [[ ! -f "${source_summary}" ]]; then
  echo "Missing source summary: ${source_summary}" >&2
  echo "Run VILA/bash_scripts/decoding_base_with_original_qa.sh first." >&2
  exit 1
fi

if [[ ! -f "${sample_id_file}" ]]; then
  echo "[sample] creating fixed sample id file: ${sample_id_file}"
  if [[ "${dry_run}" != "true" ]]; then
    "${python_bin}" - <<PY
import json, random
from pycocotools.coco import COCO
caption_file = "${data_path}/coco/annotations/captions_val2014.json"
random.seed(${seed})
coco = COCO(caption_file)
sampled = random.sample(coco.getImgIds(), ${num_samples})
with open("${sample_id_file}", "w", encoding="utf-8") as f:
    json.dump(sampled, f, indent=2)
print(f"saved sample ids -> ${sample_id_file}")
PY
  fi
fi

run_cmd() {
  echo "+ $*"
  if [[ "${dry_run}" != "true" ]]; then
    "$@"
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
            a, b = map(int, part.split(':', 1))
            step = 1 if b >= a else -1
            layers.extend(range(a, b + step, step))
        elif '-' in part and not part.startswith('-'):
            a, b = map(int, part.split('-', 1))
            step = 1 if b >= a else -1
            layers.extend(range(a, b + step, step))
        else:
            layers.append(int(part))
    seen, out = set(), []
    for layer in layers:
        if layer not in seen:
            out.append(layer); seen.add(layer)
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
  local slug
  slug=$(layer_slug "${layer_spec}")
  local range_root=./results_${slug}/${dataset}
  local stats_root=${range_root}/${model_name}_base_original_qa_n${train_num_samples}_txtattn_${slug}_allheads
  local filtered_summary=${stats_root}/txtattn_summary.json
  local surrogate_dir=${stats_root}/surrogate_score_zoo
  mkdir -p "${stats_root}" "${surrogate_dir}"

  echo "[layers ${layer_spec}] filtering summary -> ${filtered_summary}"
  run_cmd "${python_bin}" eval_scripts/filter_txtattn_summary.py \
    --summary-file "${source_summary}" \
    --output-file "${filtered_summary}" \
    --layers "${layer_spec}"

  echo "[layers ${layer_spec}] building surrogate score zoo -> ${surrogate_dir}"
  run_cmd "${python_bin}" eval_scripts/compute_surrogate_score_zoo.py \
    --summary-file "${filtered_summary}" \
    --output-dir "${surrogate_dir}"

  if [[ "${AUTO_DYNAMIC_TAU}" == "true" ]]; then
    local tau_file=${stats_root}/dynamic_tau_estimate.json
    echo "[layers ${layer_spec}] estimating dynamic tau -> ${tau_file}"
    run_cmd "${python_bin}" eval_scripts/estimate_dynamic_tau.py \
      --summary-file "${filtered_summary}" \
      --output-file "${tau_file}" \
      --round-step "${auto_tau_round_step}" \
      --round-mode "${auto_tau_round_mode}"
  fi

  local head_file=${surrogate_dir}/ranked_heads_${head_score_key}.json
  if [[ "${dry_run}" != "true" && ! -f "${head_file}" ]]; then
    echo "Missing requested head file after surrogate build: ${head_file}" >&2
    exit 1
  fi
}

run_dynamic_job() {
  local layer_spec=$1
  local slug=$2
  local gpu=$3
  local topk=$4
  local strength=$5
  local q=$6
  local score_power=$7
  local tau=$8
  local redistribute=$9

  local redir_suffix=""
  if [[ "${redistribute}" != "renorm" ]]; then
    redir_suffix="_redir${redistribute}"
  fi
  if [[ "${dynamic_renorm}" != "true" ]]; then
    redir_suffix="${redir_suffix}_norenorm"
  fi
  local braw_suffix=""
  if [[ "${min_head_back_raw}" != "0" && "${min_head_back_raw}" != "0.0" ]]; then
    braw_suffix="_bmin${min_head_back_raw}"
  fi
  local late_suffix=""
  if [[ "${intervention}" == "late_boost" && "${dynamic_late_boost_start}" != "-1" && "${dynamic_late_tau}" != "-1" && "${dynamic_late_tau}" != "-1.0" ]]; then
    late_suffix="_late${dynamic_late_boost_mode}${dynamic_late_boost_start}-${dynamic_late_boost_end}tau${dynamic_late_tau}"
  fi

  local range_root=./results_${slug}/${dataset}
  local stats_root=${range_root}/${model_name}_base_original_qa_n${train_num_samples}_txtattn_${slug}_allheads
  local head_file=${stats_root}/surrogate_score_zoo/ranked_heads_${head_score_key}.json
  local result_path=${range_root}/${model_name}_${intervention}_${dynamic_context_mode}_${head_source}_k${topk}_s${strength}_q${q}_tau${tau}_p${score_power}${redir_suffix}${late_suffix}${braw_suffix}_n${num_samples}_${head_score_key}
  mkdir -p "${result_path}"

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
  local renorm_args=()
  if [[ "${dynamic_renorm}" != "true" ]]; then
    renorm_args+=(--no-dynamic-renorm)
  fi

  echo "[GPU ${gpu}] layers=${layer_spec} intervention=${intervention} topk=${topk} min_back_raw=${min_head_back_raw} s=${strength} q=${q} tau=${tau} p=${score_power} redistribute=${redistribute} renorm=${dynamic_renorm} late_mode=${dynamic_late_boost_mode} late_start=${dynamic_late_boost_start} late_end=${dynamic_late_boost_end} late_tau=${dynamic_late_tau}"
  if [[ "${dry_run}" == "true" ]]; then
    echo "[dry-run] would write ${result_path}"
    return 0
  fi

  VILA_ATTN_IMPLEMENTATION="${attn_implementation}" CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption_dynamic \
    --model-path "${model_path}" \
    --model-name "${model_name}" \
    --image-folder "${data_path}/coco/val2014" \
    --caption_file_path "${data_path}/coco/annotations/captions_val2014.json" \
    --answers-file "${result_path}/captions.jsonl" \
    --dataset "${dataset}" \
    --temperature 0 \
    --conv-mode auto \
    --num_samples "${num_samples}" \
    --max_new_tokens 128 \
    --num_beams 1 \
    --seed "${seed}" \
    --sample-id-file "${sample_id_file}" \
    --save-sample-id-file "${sample_id_file}" \
    --intervention "${intervention}" \
    --head-source "${head_source}" \
    --head-file "${head_file}" \
    --topk "${topk}" \
    --head-score-key "${head_score_key}" \
    --head-score-normalize "${head_score_normalize}" \
    --min-head-back-raw "${min_head_back_raw}" \
    --dynamic-context-mode "${dynamic_context_mode}" \
    --dynamic-strength "${strength}" \
    --dynamic-exp-sharpness "${q}" \
    --dynamic-late-boost-start "${dynamic_late_boost_start}" \
    --dynamic-late-boost-end "${dynamic_late_boost_end}" \
    --dynamic-late-boost-mode "${dynamic_late_boost_mode}" \
    --dynamic-late-tau "${dynamic_late_tau}" \
    --dynamic-tau "${tau}" \
    --dynamic-score-power "${score_power}" \
    --dynamic-redistribute "${redistribute}" \
    --log-intervention-stats \
    --intervention-stats-file "${result_path}/intervention_stats.json" \
    "${score_args[@]}" \
    "${trace_args[@]}" \
    "${renorm_args[@]}" \
    "${resume_args[@]}" \
    > "${result_path}/decode.log" 2>&1

  "${python_bin}" eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${result_path}/captions.jsonl" \
    --caption_file captions_val2014.json \
    > "${result_path}/chair.log" 2>&1
}

for spec in "${layer_specs[@]}"; do
  prepare_layer_spec "${spec}"
done

job_idx=0
pids=()
for spec in "${layer_specs[@]}"; do
  slug=$(layer_slug "${spec}")
  tau_values=("${dynamic_tau_list[@]}")
  if [[ "${AUTO_DYNAMIC_TAU}" == "true" ]]; then
    tau_file=./results_${slug}/${dataset}/${model_name}_base_original_qa_n${train_num_samples}_txtattn_${slug}_allheads/dynamic_tau_estimate.json
    if [[ -f "${tau_file}" ]]; then
      tau_values=("$("${python_bin}" - <<PY
import json
with open("${tau_file}", "r", encoding="utf-8") as f:
    print(json.load(f).get("tau", ${dynamic_tau_list[0]}))
PY
)")
    fi
  fi
  for topk in "${topk_list[@]}"; do
    for preset in "${dynamic_presets[@]}"; do
      read -r strength q score_power <<< "${preset}"
      for tau in "${tau_values[@]}"; do
        for redistribute in "${dynamic_redistribute_list[@]}"; do
          gpu=${gpu_list[$((job_idx % ${#gpu_list[@]}))]}
          run_dynamic_job "${spec}" "${slug}" "${gpu}" "${topk}" "${strength}" "${q}" "${score_power}" "${tau}" "${redistribute}" &
          pids+=("$!")
          job_idx=$((job_idx + 1))
          if (( ${#pids[@]} >= ${#gpu_list[@]} )); then
            wait "${pids[0]}"
            pids=("${pids[@]:1}")
          fi
        done
      done
    done
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
