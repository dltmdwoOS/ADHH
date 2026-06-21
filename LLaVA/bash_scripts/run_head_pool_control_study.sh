#!/usr/bin/env bash
set -euo pipefail

# Head-pool actuator control study for LLaVA 1.5.
# Run from ADHH/LLaVA/ or the workspace root:
#   bash ADHH/LLaVA/bash_scripts/run_head_pool_control_study.sh
#
# Logs are written under each experiment/pool directory. The terminal prints
# only coarse status. Parallel mode runs at most one model process per GPU.

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

MODEL_NAME=${MODEL_NAME:-llava-v1.5-7b}
MODEL_PATH=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
DATASET=${DATASET:-coco}
DATA_PATH=${DATA_PATH:-$(adhh_default_data_path)}
LAYER_SLUG=${LAYER_SLUG:-l9_l16}
TRAIN_N=${TRAIN_N:-500}
TOPK=${TOPK:-100}
ADHH_TOPK=${ADHH_TOPK:-20}
MAX_PER_BUCKET=${MAX_PER_BUCKET:-200}
SEED=${SEED:-42}
CONV_MODE=${CONV_MODE:-vicuna_v1}
DEFAULT_HEAD_SCORE_KEYS=${DEFAULT_HEAD_SCORE_KEYS:-"global__itext_all__C_toi_HminusG_signed"}
HEAD_SCORE_KEYS=${HEAD_SCORE_KEYS:-${HEAD_SCORE_KEY:-${DEFAULT_HEAD_SCORE_KEYS}}}
HEAD_SCORE_KEY=${HEAD_SCORE_KEY:-${HEAD_SCORE_KEYS%% *}}
POOLS=${POOLS:-proposed,text_only,contrast_only,layer_matched_random}
PROBE_TEXT_SCALE=${PROBE_TEXT_SCALE:-0.0}
PROBE_THRESHOLD=${PROBE_THRESHOLD:-0.4}
PROBE_THRESHOLDING=${PROBE_THRESHOLDING:-true}
RENORM=${RENORM:-false}
AGGREGATE_TARGET_TOKENS=${AGGREGATE_TARGET_TOKENS:-true}
MAX_PROBES=${MAX_PROBES:-0}
IMG_START_POS=${IMG_START_POS:-35}
IMG_LENGTH=${IMG_LENGTH:-576}
PARALLEL=${PARALLEL:-true}
GPU_LIST=${GPU_LIST:-0 1}
RESULTS_DEACT_ROOT=${RESULTS_DEACT_ROOT:-$(adhh_default_results_root)}
RESULTS_ROOT=${RESULTS_ROOT:-${RESULTS_DEACT_ROOT}/analysis}
PROBE_SOURCE=${PROBE_SOURCE:-base}
REPRODUCED_ADHH_SOURCE=${REPRODUCED_ADHH_SOURCE:-top30_preserved}
ADHH_ALT_SOURCE=${ADHH_ALT_SOURCE:-none}
DRY_RUN=${DRY_RUN:-false}

# Helps a little with repeated model-load fragmentation; harmless if unsupported.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

PYTHON_BIN=$(adhh_python_bin)

# If multiple surrogate keys are requested, run this script once per key.
# Each run gets its own OUT_ROOT because HEAD_SCORE_KEY is part of the path.
if [[ -z "${_HEAD_POOL_CONTROL_SINGLE:-}" ]]; then
  read -r -a _head_score_key_array <<< "${HEAD_SCORE_KEYS}"
  if [[ ${#_head_score_key_array[@]} -gt 1 ]]; then
    printf '[multi] probing %d surrogate rankings:
' "${#_head_score_key_array[@]}"
    printf '  %s
' "${_head_score_key_array[@]}"
    for _head_score_key in "${_head_score_key_array[@]}"; do
      printf '
[multi] ===== HEAD_SCORE_KEY=%s =====
' "${_head_score_key}"
      _HEAD_POOL_CONTROL_SINGLE=1 HEAD_SCORE_KEY="${_head_score_key}" HEAD_SCORE_KEYS="${_head_score_key}" bash "$0"
    done
    exit 0
  fi
fi

DYNAMIC_RESULT=${DYNAMIC_RESULT:-${RESULTS_DEACT_ROOT}/${DATASET}/${MODEL_NAME}/main/l9_l16/k100/direct/tok128/q10_tau090-080}
BASE_RESULT=${BASE_RESULT:-${RESULTS_DEACT_ROOT}/${DATASET}/${MODEL_NAME}/baselines/greedy/tok128}
SOURCE_SUMMARY=${SOURCE_SUMMARY:-${RESULTS_DEACT_ROOT}/${DATASET}/${MODEL_NAME}/resources/${LAYER_SLUG}_train_n${TRAIN_N}/txtattn_summary.json}
case "${PROBE_SOURCE}" in
  base)
    DEFAULT_CAPTION_EVAL=${BASE_RESULT}/captions_eval_results.json
    probe_source_slug=probe_base
    ;;
  dynamic)
    DEFAULT_CAPTION_EVAL=${DYNAMIC_RESULT}/captions_eval_results.json
    probe_source_slug=probe_dynamic
    ;;
  custom)
    if [[ -z "${CAPTION_EVAL+x}" ]]; then
      echo "PROBE_SOURCE=custom requires CAPTION_EVAL=/path/to/captions_eval_results.json" >&2
      exit 1
    fi
    DEFAULT_CAPTION_EVAL=${CAPTION_EVAL}
    probe_source_slug=${PROBE_SOURCE_SLUG:-probe_custom}
    ;;
  *)
    echo "Unsupported PROBE_SOURCE=${PROBE_SOURCE}; use base, dynamic, or custom" >&2
    exit 1
    ;;
esac
CAPTION_EVAL=${CAPTION_EVAL:-${DEFAULT_CAPTION_EVAL}}
IMAGE_FOLDER=${IMAGE_FOLDER:-${DATA_PATH}/${DATASET}/val2014}
COCO_PATH=${COCO_PATH:-${DATA_PATH}/${DATASET}/annotations}
DEFAULT_HEAD_FILE=${RESULTS_DEACT_ROOT}/${DATASET}/${MODEL_NAME}/resources/${LAYER_SLUG}_train_n${TRAIN_N}/surrogate_score_zoo/ranked_heads_${HEAD_SCORE_KEY}.json
HEAD_FILE=${HEAD_FILE:-${DEFAULT_HEAD_FILE}}
case "${REPRODUCED_ADHH_SOURCE}" in
  top30_preserved)
    DEFAULT_REPRODUCED_ADHH_FILE=./results/coco/llava-v1.5-7b_reproduced_adhh_zero_ablation_top30.json
    adhh_source_slug=adhh_top30_preserved
    ;;
  backup_top100)
    DEFAULT_REPRODUCED_ADHH_FILE=./results/coco/llava-v1.5-7b_reproduced_adhh_zero_ablation_backup_ranked_top100.json
    adhh_source_slug=adhh_backup_top100
    ;;
  custom)
    if [[ -z "${REPRODUCED_ADHH_FILE+x}" ]]; then
      echo "REPRODUCED_ADHH_SOURCE=custom requires REPRODUCED_ADHH_FILE=/path/to/head_file.json" >&2
      exit 1
    fi
    DEFAULT_REPRODUCED_ADHH_FILE=${REPRODUCED_ADHH_FILE}
    adhh_source_slug=${REPRODUCED_ADHH_SLUG:-adhh_custom}
    ;;
  *)
    echo "Unsupported REPRODUCED_ADHH_SOURCE=${REPRODUCED_ADHH_SOURCE}; use top30_preserved, backup_top100, or custom" >&2
    exit 1
    ;;
esac
REPRODUCED_ADHH_FILE=${REPRODUCED_ADHH_FILE:-${DEFAULT_REPRODUCED_ADHH_FILE}}

case "${ADHH_ALT_SOURCE}" in
  top30_preserved)
    DEFAULT_ADHH_ALT_FILE=./results/coco/llava-v1.5-7b_reproduced_adhh_zero_ablation_top30.json
    adhh_alt_source_slug=adhh_alt_top30_preserved
    ;;
  n3000_reproduced)
    DEFAULT_ADHH_ALT_FILE=../LLaVA_backup/results/coco/llava_3000/identify_attention_head/ranked_hal_heads.json
    adhh_alt_source_slug=adhh_alt_n3000_reproduced
    ;;
  backup_top100)
    DEFAULT_ADHH_ALT_FILE=./results/coco/llava-v1.5-7b_reproduced_adhh_zero_ablation_backup_ranked_top100.json
    adhh_alt_source_slug=adhh_alt_legacy_backup_top100
    ;;
  custom)
    if [[ -z "${ADHH_ALT_FILE+x}" ]]; then
      echo "ADHH_ALT_SOURCE=custom requires ADHH_ALT_FILE=/path/to/head_file.json" >&2
      exit 1
    fi
    DEFAULT_ADHH_ALT_FILE=${ADHH_ALT_FILE}
    adhh_alt_source_slug=${ADHH_ALT_SLUG:-adhh_alt_custom}
    ;;
  none)
    DEFAULT_ADHH_ALT_FILE=""
    adhh_alt_source_slug=adhh_alt_none
    ;;
  *)
    echo "Unsupported ADHH_ALT_SOURCE=${ADHH_ALT_SOURCE}; use top30_preserved, n3000_reproduced, backup_top100, custom, or none" >&2
    exit 1
    ;;
esac
ADHH_ALT_FILE=${ADHH_ALT_FILE:-${DEFAULT_ADHH_ALT_FILE}}
renorm_slug=norenorm
if [[ "${RENORM}" == "true" ]]; then
  renorm_slug=renorm
fi
agg_slug=firstsubtok
if [[ "${AGGREGATE_TARGET_TOKENS}" == "true" ]]; then
  agg_slug=allsubtok
fi
threshold_slug=thr${PROBE_THRESHOLD}
if [[ "${PROBE_THRESHOLDING}" != "true" ]]; then
  threshold_slug=unconditional
fi
probe_setting_slug=${probe_source_slug}_${agg_slug}_${renorm_slug}_${threshold_slug}
adhh_topk_slug=all
if [[ "${ADHH_TOPK}" -gt 0 ]]; then
  adhh_topk_slug=k${ADHH_TOPK}
fi
source_slug=adhh-${adhh_source_slug}_${adhh_topk_slug}_alt-${adhh_alt_source_slug}
if [[ "${POOLS}" != *"adhh"* ]]; then
  source_slug=head_pools
fi
experiment_slug=k${TOPK}_h${MAX_PER_BUCKET}_g${MAX_PER_BUCKET}_${probe_setting_slug}
OUT_ROOT=${OUT_ROOT:-${RESULTS_ROOT}/head_pool_control/${MODEL_NAME}/${DATASET}/${LAYER_SLUG}/${HEAD_SCORE_KEY}/${experiment_slug}/${source_slug}}
PROBE_FILE=${PROBE_FILE:-${OUT_ROOT}/actuation_probes.jsonl}
PROBE_SET_LOG=${OUT_ROOT}/build_probe_set.log

mkdir -p "${OUT_ROOT}"

if [[ ! -f "${CAPTION_EVAL}" ]]; then
  echo "Missing caption eval: ${CAPTION_EVAL}" >&2
  exit 1
fi

layer_spec_from_slug() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import sys
slug = sys.argv[1].strip()
if not slug.startswith('l'):
    print(slug)
    raise SystemExit
parts = [p for p in slug[1:].split('_l') if p]
layers = [int(p) for p in parts]
if len(layers) == 2:
    a, b = layers
    step = 1 if b >= a else -1
    if layers == [a, b]:
        print(f"{a}:{b}")
    else:
        print(','.join(map(str, layers)))
elif len(layers) > 2:
    print(','.join(map(str, layers)))
elif len(layers) == 1:
    print(str(layers[0]))
else:
    raise SystemExit(f"Cannot parse LAYER_SLUG={slug}")
PY
}

build_missing_head_file() {
  if [[ "${HEAD_FILE}" != "${DEFAULT_HEAD_FILE}" ]]; then
    echo "Missing custom head file: ${HEAD_FILE}" >&2
    exit 1
  fi

  local stats_root=${RESULTS_DEACT_ROOT}/${DATASET}/${MODEL_NAME}/resources/${LAYER_SLUG}_train_n${TRAIN_N}
  local filtered_summary=${stats_root}/txtattn_summary.json
  local surrogate_dir=${stats_root}/surrogate_score_zoo
  local layer_spec
  layer_spec=$(layer_spec_from_slug "${LAYER_SLUG}")

  mkdir -p "${stats_root}" "${surrogate_dir}"
  if [[ ! -f "${filtered_summary}" ]]; then
    if [[ ! -f "${SOURCE_SUMMARY}" ]]; then
      echo "Missing head file: ${HEAD_FILE}" >&2
      echo "Also missing source summary for auto-build: ${SOURCE_SUMMARY}" >&2
      echo "Run base txtattn tracing for l0_l31 first, or pass SOURCE_SUMMARY=/path/to/txtattn_summary.json." >&2
      exit 1
    fi
    echo "[setup] missing head file; filtering summary for ${LAYER_SLUG} (${layer_spec}) -> ${filtered_summary}"
    "${PYTHON_BIN}" eval_scripts/filter_txtattn_summary.py \
      --summary-file "${SOURCE_SUMMARY}" \
      --output-file "${filtered_summary}" \
      --layers "${layer_spec}"
  else
    echo "[setup] missing head file; reusing filtered summary: ${filtered_summary}"
  fi

  echo "[setup] building surrogate rankings -> ${surrogate_dir}"
  "${PYTHON_BIN}" eval_scripts/compute_surrogate_score_zoo.py \
    --summary-file "${filtered_summary}" \
    --output-dir "${surrogate_dir}"

  if [[ ! -f "${HEAD_FILE}" ]]; then
    echo "Missing requested head file after auto-build: ${HEAD_FILE}" >&2
    exit 1
  fi
}

if [[ ! -f "${HEAD_FILE}" ]]; then
  build_missing_head_file
fi
if [[ ( "${POOLS}" == *"reproduced_adhh"* || "${POOLS}" == *"adhh"* ) && ! -f "${REPRODUCED_ADHH_FILE}" ]]; then
  echo "Missing reproduced AD-HH pool: ${REPRODUCED_ADHH_FILE}" >&2
  exit 1
fi
if [[ "${POOLS}" == *"adhh_alt"* && ( -z "${ADHH_ALT_FILE}" || ! -f "${ADHH_ALT_FILE}" ) ]]; then
  echo "Missing alternative AD-HH pool: ${ADHH_ALT_FILE}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  printf '[dry-run] output root: %s\n' "${OUT_ROOT}"
  printf '[dry-run] caption eval: %s\n' "${CAPTION_EVAL}"
  printf '[dry-run] head file   : %s\n' "${HEAD_FILE}"
  printf '[dry-run] probe file  : %s\n' "${PROBE_FILE}"
  printf '[dry-run] pools       : %s\n' "${POOLS}"
  exit 0
fi

printf '[setup] probe source: %s (%s)\n' "${PROBE_SOURCE}" "${CAPTION_EVAL}"
printf '[setup] reproduced AD-HH source: %s (%s)\n' "${REPRODUCED_ADHH_SOURCE}" "${REPRODUCED_ADHH_FILE}"
printf '[setup] alternative AD-HH source: %s (%s)\n' "${ADHH_ALT_SOURCE}" "${ADHH_ALT_FILE}"
printf '[setup] output root: %s\n' "${OUT_ROOT}"
printf '[setup] building probe set -> %s\n' "${PROBE_FILE}"
{
  echo "[cmd] ${PYTHON_BIN} eval_scripts/build_actuation_probe_set.py ..."
  "${PYTHON_BIN}" eval_scripts/build_actuation_probe_set.py \
    --caption-eval "${CAPTION_EVAL}" \
    --coco-path "${COCO_PATH}" \
    --output-file "${PROBE_FILE}" \
    --max-per-bucket "${MAX_PER_BUCKET}" \
    --seed "${SEED}"
} > "${PROBE_SET_LOG}" 2>&1
printf '[setup] probe set ready (log: %s)\n' "${PROBE_SET_LOG}"

renorm_arg=()
if [[ "${RENORM}" == "true" ]]; then
  renorm_arg=(--renorm)
fi
probe_thresholding_arg=()
if [[ "${PROBE_THRESHOLDING}" != "true" ]]; then
  probe_thresholding_arg=(--no-probe-thresholding)
fi
aggregate_arg=()
if [[ "${AGGREGATE_TARGET_TOKENS}" == "true" ]]; then
  aggregate_arg=(--aggregate-target-tokens)
fi
max_probe_arg=(--max-probes "${MAX_PROBES}")
adhh_arg=()
if [[ -f "${REPRODUCED_ADHH_FILE}" ]]; then
  adhh_arg=(--reproduced-adhh-file "${REPRODUCED_ADHH_FILE}")
fi
adhh_alt_arg=()
if [[ -n "${ADHH_ALT_FILE}" && -f "${ADHH_ALT_FILE}" ]]; then
  adhh_alt_arg=(--adhh-alt-file "${ADHH_ALT_FILE}")
fi
restrict_layers_arg=(--restrict-layers "$(layer_spec_from_slug "${LAYER_SLUG}")")

run_probe_job() {
  local pool_name=$1
  local gpu_id=$2
  local out_dir=$3
  local log_file=${out_dir}/probe.log
  mkdir -p "${out_dir}"
  {
    echo "[start] $(date -Is) pool=${pool_name} gpu=${gpu_id}"
    echo "[output] ${out_dir}"
    echo "[cmd] CUDA_VISIBLE_DEVICES=${gpu_id} ${PYTHON_BIN} eval_scripts/probe_head_pool_actuation.py --pools ${pool_name}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" eval_scripts/probe_head_pool_actuation.py \
      --model-path "${MODEL_PATH}" \
      --image-folder "${IMAGE_FOLDER}" \
      --probe-file "${PROBE_FILE}" \
      --head-file "${HEAD_FILE}" \
      --output-dir "${out_dir}" \
      --conv-mode "${CONV_MODE}" \
      --topk "${TOPK}" \
      --adhh-topk "${ADHH_TOPK}" \
      --pools "${pool_name}" \
      --random-seed "${SEED}" \
      --img-start-pos "${IMG_START_POS}" \
      --img-length "${IMG_LENGTH}" \
      --probe-text-scale "${PROBE_TEXT_SCALE}" \
      --probe-threshold "${PROBE_THRESHOLD}" \
      "${probe_thresholding_arg[@]}" \
      "${max_probe_arg[@]}" \
      "${renorm_arg[@]}" \
      "${aggregate_arg[@]}" \
      "${adhh_arg[@]}" \
      "${adhh_alt_arg[@]}" \
      "${restrict_layers_arg[@]}"
    echo "[done] $(date -Is) pool=${pool_name}"
  } > "${log_file}" 2>&1
}

IFS=',' read -r -a raw_pool_array <<< "${POOLS}"
pool_array=()
for raw_pool in "${raw_pool_array[@]}"; do
  pool_name=$(echo "${raw_pool}" | xargs)
  [[ -n "${pool_name}" ]] && pool_array+=("${pool_name}")
done
read -r -a gpu_array <<< "${GPU_LIST}"
if [[ ${#gpu_array[@]} -eq 0 ]]; then
  gpu_array=(0)
fi

printf '[run] pools: %s\n' "${pool_array[*]}"
printf '[run] gpus : %s\n' "${gpu_array[*]}"

if [[ "${PARALLEL}" == "true" ]]; then
  # Batch scheduling: one process per GPU, then wait for the whole batch before
  # launching more. This avoids loading multiple 7B models onto the same GPU.
  next_pool=0
  total=${#pool_array[@]}
  while [[ ${next_pool} -lt ${total} ]]; do
    pids=()
    names=()
    logs=()
    for gpu_id in "${gpu_array[@]}"; do
      if [[ ${next_pool} -ge ${total} ]]; then
        break
      fi
      pool_name=${pool_array[${next_pool}]}
      out_dir=${OUT_ROOT}/pool_${pool_name}
      log_file=${out_dir}/probe.log
      mkdir -p "${out_dir}"
      printf '[start] pool=%s gpu=%s log=%s\n' "${pool_name}" "${gpu_id}" "${log_file}"
      run_probe_job "${pool_name}" "${gpu_id}" "${out_dir}" &
      pids+=("$!")
      names+=("${pool_name}")
      logs+=("${log_file}")
      next_pool=$((next_pool + 1))
    done

    batch_status=0
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        printf '[done]  pool=%s log=%s\n' "${names[$i]}" "${logs[$i]}"
      else
        printf '[fail]  pool=%s log=%s\n' "${names[$i]}" "${logs[$i]}" >&2
        printf '[fail]  tail of %s:\n' "${logs[$i]}" >&2
        tail -n 40 "${logs[$i]}" >&2 || true
        batch_status=1
      fi
    done
    if [[ ${batch_status} -ne 0 ]]; then
      echo "At least one pool job failed; stopping before launching the next batch." >&2
      exit ${batch_status}
    fi
  done
else
  for pool_name in "${pool_array[@]}"; do
    out_dir=${OUT_ROOT}/pool_${pool_name}
    log_file=${out_dir}/probe.log
    mkdir -p "${out_dir}"
    printf '[start] pool=%s gpu=%s log=%s\n' "${pool_name}" "${gpu_array[0]}" "${log_file}"
    if run_probe_job "${pool_name}" "${gpu_array[0]}" "${out_dir}"; then
      printf '[done]  pool=%s log=%s\n' "${pool_name}" "${log_file}"
    else
      printf '[fail]  pool=%s log=%s\n' "${pool_name}" "${log_file}" >&2
      tail -n 40 "${log_file}" >&2 || true
      exit 1
    fi
  done
fi

printf '[merge] collecting pool summaries\n'
"${PYTHON_BIN}" - "${OUT_ROOT}" > "${OUT_ROOT}/merge.log" 2>&1 <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
merged = {"by_pool": {}, "pool_summaries": {}, "root": str(root)}
missing = []
for pool_dir in sorted(root.glob('pool_*')):
    summary_path = pool_dir / 'head_pool_probe_summary.json'
    if not summary_path.exists():
        missing.append(str(summary_path))
        continue
    data = json.loads(summary_path.read_text())
    merged["pool_summaries"][pool_dir.name] = str(summary_path)
    merged["by_pool"].update(data.get("by_pool", {}))
merged["missing_summaries"] = missing
merged_path = root / 'head_pool_probe_summary.json'
merged_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
print(f"wrote {merged_path}")
PY
printf '[merge] wrote %s\n' "${OUT_ROOT}/head_pool_probe_summary.json"
printf '[done] logs and outputs are under %s\n' "${OUT_ROOT}"
