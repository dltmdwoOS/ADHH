#!/usr/bin/env bash
set -euo pipefail

# End-to-end layer-list sweep:
# 1) filter txtattn_summary.json to each requested layer list
# 2) rebuild surrogate head pools from the filtered summary
# 3) run fixed-hyperparameter dynamic CHAIR captioning/evaluation
#
# Run from LLaVA/:
#   bash bash_scripts/run_layer_range_dynamic_pipeline.sh
#
# Common overrides:
#   LAYER_SPECS="12:31 13:31 9,10,11,12,13,15,16" TOPK_LIST="100" DYNAMIC_PRESETS="0.7 8.0 1.0" bash ...
#   DRY_RUN=true bash ...

model_name=${MODEL_NAME:-llama3-llava-next-8b}
model_path=${MODEL_PATH:-lmms-lab/llama3-llava-next-8b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
seed=${SEED:-42}
num_samples=${NUM_SAMPLES:-500}
train_num_samples=${TRAIN_NUM_SAMPLES:-500}

# Source summary must contain all heads covering every requested layer range.
source_summary=${SOURCE_SUMMARY:-./results/${dataset}/${model_name}_base_original_qa_n${train_num_samples}_txtattn_l0_l31_allheads/txtattn_summary.json}

# Space-separated layer specs. Each spec can be a comma-separated explicit list
# (e.g. 9,10,11,12,13,15,16) or a compact range (e.g. 9:16).
# LAYER_RANGES is kept as a backward-compatible alias.
read -r -a layer_specs <<< "${LAYER_SPECS:-${LAYER_RANGES:-9:16}}"

# Dynamic/head-pool settings.
head_source=${HEAD_SOURCE:-file}
head_score_key=${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG}
head_score_normalize=${HEAD_SCORE_NORMALIZE:-rank_percentile}
use_head_scores=${USE_HEAD_SCORES:-true}
dynamic_context_mode=${DYNAMIC_CONTEXT_MODE:-ratio_exp}
read -r -a dynamic_redistribute_list <<< "${DYNAMIC_REDISTRIBUTES:-renorm}"
dynamic_taus_env_set=false
if [[ -n "${DYNAMIC_TAUS+x}" ]]; then
  dynamic_taus_env_set=true
fi
read -r -a dynamic_tau_list <<< "${DYNAMIC_TAUS:-0.9}"
AUTO_DYNAMIC_TAU=false
if [[ -n "${AUTO_DYNAMIC_TAU+x}" ]]; then
  auto_dynamic_tau=${AUTO_DYNAMIC_TAU}
elif [[ "${dynamic_taus_env_set}" == "true" ]]; then
  auto_dynamic_tau=false
else
  auto_dynamic_tau=true
fi
auto_tau_round_step=${AUTO_TAU_ROUND_STEP:-0.05}
auto_tau_round_mode=${AUTO_TAU_ROUND_MODE:-floor}
read -r -a topk_list <<< "${TOPK_LIST:-100}"

# Presets are separated by semicolon, each preset is: strength exp_sharpness score_power.
IFS=';' read -r -a dynamic_presets <<< "${DYNAMIC_PRESETS:-1.0 9.0 1.0; 1.0 8.0 1.0; 1.0 7.0 1.0; 1.0 10.0 1.0;}"

log_dynamic_trace=${LOG_DYNAMIC_TRACE:-true}
dynamic_trace_topn=${DYNAMIC_TRACE_TOPN:-10}
dynamic_trace_every=${DYNAMIC_TRACE_EVERY:-5}
resume=${RESUME:-true}
dry_run=${DRY_RUN:-false}

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
  echo "Expected a layer-0-to-31 all-head txtattn_summary.json. Run LLaVA_NeXT/bash_scripts/decoding_base_with_original_qa.sh first." >&2
  exit 1
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
  local range_root=./results_${layer_slug_name}/${dataset}
  local stats_root=${range_root}/${model_name}_base_original_qa_n${train_num_samples}_txtattn_${layer_slug_name}_allheads
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

  echo "[layers ${layer_spec}] building combo head pools -> ${surrogate_dir}"
  run_cmd "${python_bin}" eval_scripts/build_layer_surrogate_combos.py \
    --summary-file "${filtered_summary}" \
    --output-dir "${surrogate_dir}"

  if [[ "${auto_dynamic_tau}" == "true" ]]; then
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
  local layer_slug_name=$2
  local gpu=$3
  local topk=$4
  local dynamic_strength=$5
  local dynamic_exp_sharpness=$6
  local dynamic_score_power=$7
  local dynamic_tau=$8
  local dynamic_redistribute=$9

  local redir_suffix=""
  if [[ "${dynamic_redistribute}" != "renorm" ]]; then
    redir_suffix="_redir${dynamic_redistribute}"
  fi

  local range_root=./results_${layer_slug_name}/${dataset}
  local stats_root=${range_root}/${model_name}_base_original_qa_n${train_num_samples}_txtattn_${layer_slug_name}_allheads
  local head_file=${stats_root}/surrogate_score_zoo/ranked_heads_${head_score_key}.json
  local result_path=${range_root}/${model_name}_dynamic_${dynamic_context_mode}_${head_source}_k${topk}_s${dynamic_strength}_q${dynamic_exp_sharpness}_tau${dynamic_tau}_p${dynamic_score_power}${redir_suffix}_n${num_samples}_${head_score_key}
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

  echo "[GPU ${gpu}] layers=${layer_spec} topk=${topk} s=${dynamic_strength} q=${dynamic_exp_sharpness} tau=${dynamic_tau} p=${dynamic_score_power} redistribute=${dynamic_redistribute}"

  if [[ "${dry_run}" == "true" ]]; then
    echo "[dry-run] would write ${result_path}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption_dynamic \
    --model-path "${model_path}" \
    --device-map "${DEVICE_MAP:-auto}" \
    --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
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
    >> "${result_path}/decode.log" 2>&1

  "${python_bin}" eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${result_path}/captions.jsonl" \
    --caption_file captions_val2014.json \
    > "${result_path}/chair.log" 2>&1

  echo "[GPU ${gpu}] done layers=${layer_spec} topk=${topk} s=${dynamic_strength} q=${dynamic_exp_sharpness} tau=${dynamic_tau} p=${dynamic_score_power} redistribute=${dynamic_redistribute}"
}

for layer_spec in "${layer_specs[@]}"; do
  prepare_layer_spec "${layer_spec}"
done

pids=()
job_idx=0
for layer_spec in "${layer_specs[@]}"; do
  layer_slug_name=$(layer_slug "${layer_spec}")
  tau_values=("${dynamic_tau_list[@]}")
  if [[ "${auto_dynamic_tau}" == "true" ]]; then
    tau_file=./results_${layer_slug_name}/${dataset}/${model_name}_base_original_qa_n${train_num_samples}_txtattn_${layer_slug_name}_allheads/dynamic_tau_estimate.json
    if [[ "${dry_run}" == "true" && ! -f "${tau_file}" ]]; then
      tau_values=(AUTO)
    else
      tau_values=("$("${python_bin}" - <<PY
import json
with open("${tau_file}") as f:
    print(json.load(f)["recommended_tau_str"])
PY
)")
    fi
  fi

  for preset in "${dynamic_presets[@]}"; do
    read -r dynamic_strength dynamic_exp_sharpness dynamic_score_power <<< "${preset}"
    for dynamic_tau in "${tau_values[@]}"; do
      for dynamic_redistribute in "${dynamic_redistribute_list[@]}"; do
        for topk in "${topk_list[@]}"; do
          gpu="${gpu_list[$((job_idx % ${#gpu_list[@]}))]}"
          run_dynamic_job "${layer_spec}" "${layer_slug_name}" "${gpu}" "${topk}" "${dynamic_strength}" "${dynamic_exp_sharpness}" "${dynamic_score_power}" "${dynamic_tau}" "${dynamic_redistribute}" &
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
