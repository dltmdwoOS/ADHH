#!/usr/bin/env bash
set -euo pipefail

# CHAIR appendix/control experiment:
# run DEACT online dynamic suppression with layer-matched random control heads.
# By default ADHH generates controls from the current ranked heads; set
# LAYER_MATCHED_SOURCE=probe only when a historical probe-pool file is present.

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-$(adhh_default_data_path)}
seed=${SEED:-42}
num_samples=${NUM_SAMPLES:-500}
train_num_samples=${TRAIN_NUM_SAMPLES:-500}
gpu=${GPU:-0}
python_bin=$(adhh_python_bin)
max_new_tokens=${MAX_NEW_TOKENS:-128}
layer_spec=${LAYER_SPECS:-9:16}
topk=${TOPK:-100}
random_seeds=${RANDOM_SEEDS:-42}
layer_matched_source=${LAYER_MATCHED_SOURCE:-generate}  # probe | generate

probe_root=${LAYER_MATCHED_PROBE_ROOT:-./old/results_old/results_prove_threshold_sweep/head_pool_control/llava-v1.5-7b/coco/l9_l16/global__itext_all__C_toi_HminusG_signed/k100_h200_g200_probe_base_firstsubtok_norenorm_thr0.4/adhh-adhh_top30_preserved_k20_alt-adhh_alt_n3000_reproduced}
probe_pool_file=${LAYER_MATCHED_POOL_FILE:-${probe_root}/pool_layer_matched_random/head_pool_probe_pools.json}
probe_summary_file=${LAYER_MATCHED_SUMMARY_FILE:-${probe_root}/pool_layer_matched_random/head_pool_probe_summary.json}
probe_pool_name=${LAYER_MATCHED_POOL_NAME:-layer_matched_random}

results_root=${RESULTS_ROOT:-$(adhh_default_results_root)}
head_score_key=${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG_signed}
head_score_normalize=${HEAD_SCORE_NORMALIZE:-rank_percentile}
dynamic_strength=${DYNAMIC_STRENGTH:-1.0}
dynamic_exp_sharpness=${DYNAMIC_EXP_SHARPNESS:-10.0}
dynamic_score_power=${DYNAMIC_SCORE_POWER:-1.0}
dynamic_context_mode=${DYNAMIC_CONTEXT_MODE:-ratio_exp}
dynamic_late_boost_start=${DYNAMIC_LATE_BOOST_START:-0}
dynamic_late_boost_end=${DYNAMIC_LATE_BOOST_END:-${max_new_tokens}}
dynamic_late_boost_mode=${DYNAMIC_LATE_BOOST_MODE:-linear}
dynamic_redistribute=${DYNAMIC_REDISTRIBUTE:-none}
dynamic_renorm=${DYNAMIC_RENORM:-false}
resume=${RESUME:-true}
log_intervention_stats=${LOG_INTERVENTION_STATS:-false}
auto_dynamic_tau=${AUTO_DYNAMIC_TAU:-true}
auto_tau_round_step=${AUTO_TAU_ROUND_STEP:-0.01}
auto_tau_round_mode=${AUTO_TAU_ROUND_MODE:-floor}
auto_tau_calibration_scope=${AUTO_TAU_CALIBRATION_SCOPE:-selected_head}
auto_tau_calibration_bucket=${AUTO_TAU_CALIBRATION_BUCKET:-all}
auto_tau_hi_quantile=${AUTO_TAU_HI_QUANTILE:-q66}
auto_tau_lo_quantile=${AUTO_TAU_LO_QUANTILE:-q33}
dynamic_tau_fallback=${DYNAMIC_TAU:-0.90}
dynamic_late_tau_fallback=${DYNAMIC_LATE_TAU:-0.80}
dry_run=${DRY_RUN:-false}

export PYTHONUNBUFFERED=1

slug_float() {
  "${python_bin}" - "$1" <<'PY'
import sys
x = float(sys.argv[1])
if 0 < x < 1:
    s = f"{x:.4f}".rstrip("0")
    decimals = s.split(".", 1)[1] if "." in s else ""
    if len(decimals) < 2:
        s = f"{x:.2f}"
    print(s.replace(".", ""))
else:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    print(s.replace(".", "") or "0")
PY
}

layer_slug() {
  "${python_bin}" - "$1" <<'PY'
import sys
text = sys.argv[1]
layers = []
for part in text.replace(";", ",").split(","):
    part = part.strip()
    if not part:
        continue
    if ":" in part:
        a, b = [int(x) for x in part.split(":", 1)]
        step = 1 if b >= a else -1
        layers.extend(range(a, b + step, step))
    elif "-" in part and not part.startswith("-"):
        a, b = [int(x) for x in part.split("-", 1)]
        step = 1 if b >= a else -1
        layers.extend(range(a, b + step, step))
    else:
        layers.append(int(part))
seen, out = set(), []
for layer in layers:
    if layer not in seen:
        out.append(layer)
        seen.add(layer)
if len(out) > 1 and out == list(range(out[0], out[-1] + 1)):
    print(f"l{out[0]}_l{out[-1]}")
else:
    print("l" + "_l".join(str(x) for x in out))
PY
}

update_slug() {
  if [[ "${dynamic_redistribute}" == "none" && "${dynamic_renorm}" != "true" ]]; then
    echo "direct"
  elif [[ "${dynamic_redistribute}" == "none" && "${dynamic_renorm}" == "true" ]]; then
    echo "renorm"
  elif [[ "${dynamic_redistribute}" == "renorm" ]]; then
    echo "renorm"
  else
    echo "redir_${dynamic_redistribute}"
  fi
}

layer_slug_name=$(layer_slug "${layer_spec}")
resource_root=${results_root}/${dataset}/${model_name}/resources/${layer_slug_name}_train_n${train_num_samples}
summary_file=${SUMMARY_FILE:-${resource_root}/txtattn_summary.json}
proposed_head_file=${HEAD_FILE:-${resource_root}/surrogate_score_zoo/ranked_heads_${head_score_key}.json}

if [[ ! -f "${proposed_head_file}" ]]; then
  echo "Missing proposed head file: ${proposed_head_file}" >&2
  echo "Run run_layer_list_dynamic_pipeline.sh first for LAYER_SPECS=${layer_spec}, or set HEAD_FILE." >&2
  exit 1
fi
if [[ "${auto_dynamic_tau}" == "true" && ! -f "${summary_file}" ]]; then
  echo "Missing txt-attn summary for automatic tau estimation: ${summary_file}" >&2
  echo "Run run_layer_list_dynamic_pipeline.sh first for LAYER_SPECS=${layer_spec}, or set SUMMARY_FILE." >&2
  exit 1
fi

sample_dir=${results_root}/${dataset}/${model_name}/shared_samples
sample_id_file=${SAMPLE_ID_FILE:-${sample_dir}/val_seed${seed}_n${num_samples}.json}
mkdir -p "${sample_dir}"
if [[ "${dry_run}" != "true" && ! -f "${sample_id_file}" ]]; then
  "${python_bin}" - <<PY
import json, random
from pycocotools.coco import COCO
caption_file = "${data_path}/coco/annotations/captions_val2014.json"
random.seed(${seed})
coco = COCO(caption_file)
sampled = random.sample(coco.getImgIds(), ${num_samples})
with open("${sample_id_file}", "w") as f:
    json.dump(sampled, f, indent=2)
print("saved sample ids -> ${sample_id_file}")
PY
fi

q_slug=$(slug_float "${dynamic_exp_sharpness}")
update_name=$(update_slug)

renorm_args=()
if [[ "${dynamic_renorm}" != "true" ]]; then
  renorm_args+=(--no-dynamic-renorm)
fi
resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi
stats_args=()
if [[ "${log_intervention_stats}" == "true" ]]; then
  stats_args+=(--log-intervention-stats)
fi

if [[ "${layer_matched_source}" == "probe" ]]; then
  control_runs="probe_table"
else
  control_runs="${random_seeds}"
fi

for control_run in ${control_runs}; do
  control_slug=${layer_matched_source}_${control_run}
  control_resource_root=${resource_root}/control_pools/layer_matched_random/${control_slug}/top${topk}
  mkdir -p "${control_resource_root}"
  if [[ "${layer_matched_source}" == "probe" ]]; then
    random_head_file=${control_resource_root}/ranked_heads_layer_matched_random_probe_table.json
  else
    random_head_file=${control_resource_root}/ranked_heads_layer_matched_random_seed${control_run}.json
  fi

  if [[ "${layer_matched_source}" == "probe" ]]; then
    if [[ ! -f "${probe_pool_file}" ]]; then
      echo "Missing saved probe pool file: ${probe_pool_file}" >&2
      exit 1
    fi
    echo "[layer-matched-random] export saved probe pool -> ${random_head_file}"
    "${python_bin}" eval_scripts/export_probe_pool_heads.py \
      --pool-file "${probe_pool_file}" \
      --pool-name "${probe_pool_name}" \
      --reference-head-file "${proposed_head_file}" \
      --output-file "${random_head_file}" \
      --topk "${topk}" \
      --score-key "${head_score_key}" \
      --source-summary-file "${probe_summary_file}"
  else
    echo "[layer-matched-random] build heads seed=${control_run} -> ${random_head_file}"
    "${python_bin}" eval_scripts/build_layer_matched_random_heads.py \
      --head-file "${proposed_head_file}" \
      --output-file "${random_head_file}" \
      --topk "${topk}" \
      --seed "${control_run}" \
      --score-key "${head_score_key}"
  fi

  tau_file=${control_resource_root}/dynamic_tau_estimate.json
  if [[ "${auto_dynamic_tau}" == "true" ]]; then
    echo "[layer-matched-random] estimate tau from selected top-${topk} control heads -> ${tau_file}"
    "${python_bin}" eval_scripts/estimate_dynamic_tau.py \
      --summary-file "${summary_file}" \
      --head-file "${random_head_file}" \
      --topk "${topk}" \
      --topk-list "${topk}" \
      --calibration-scope "${auto_tau_calibration_scope}" \
      --calibration-bucket "${auto_tau_calibration_bucket}" \
      --hi-quantile "${auto_tau_hi_quantile}" \
      --lo-quantile "${auto_tau_lo_quantile}" \
      --output-file "${tau_file}" \
      --round-step "${auto_tau_round_step}" \
      --round-mode "${auto_tau_round_mode}"
    read -r dynamic_tau dynamic_late_tau < <("${python_bin}" - <<PY
import json
with open("${tau_file}", "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("recommended_tau_hi_str", data["recommended_tau_str"]), data.get("recommended_tau_lo_str", data.get("recommended_tau_str", "${dynamic_late_tau_fallback}")))
PY
)
  else
    dynamic_tau=${dynamic_tau_fallback}
    dynamic_late_tau=${dynamic_late_tau_fallback}
    "${python_bin}" - <<PY
import json, os
out = {
    "method": "manual_fallback",
    "recommended_tau_hi": float("${dynamic_tau}"),
    "recommended_tau_hi_str": "${dynamic_tau}",
    "recommended_tau_lo": float("${dynamic_late_tau}"),
    "recommended_tau_lo_str": "${dynamic_late_tau}",
    "recommended_tau": float("${dynamic_tau}"),
    "recommended_tau_str": "${dynamic_tau}",
    "head_file": "${random_head_file}",
    "topk": int("${topk}"),
}
os.makedirs(os.path.dirname("${tau_file}"), exist_ok=True)
with open("${tau_file}", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
PY
  fi

  hi_slug=$(slug_float "${dynamic_tau}")
  lo_slug=$(slug_float "${dynamic_late_tau}")
  if [[ "${layer_matched_source}" == "probe" ]]; then
    result_path=${results_root}/${dataset}/${model_name}/appendix/layer_matched_random_probe_table/${layer_slug_name}/k${topk}/${update_name}/tok${max_new_tokens}/q${q_slug}_tau${hi_slug}-${lo_slug}
  else
    result_path=${results_root}/${dataset}/${model_name}/appendix/layer_matched_random/${layer_slug_name}/k${topk}/${update_name}/tok${max_new_tokens}/q${q_slug}_tau${hi_slug}-${lo_slug}/seed${control_run}
  fi
  mkdir -p "${result_path}"
  cp "${random_head_file}" "${result_path}/$(basename "${random_head_file}")"
  cp "${tau_file}" "${result_path}/dynamic_tau_estimate.json"

  if [[ "${dry_run}" == "true" ]]; then
    echo "[dry-run] CHAIR layer-matched random would write -> ${result_path}"
    echo "[dry-run] data_path=${data_path}"
    echo "[dry-run] sample_id_file=${sample_id_file}"
    continue
  fi

  echo "[GPU ${gpu}] CHAIR layer-matched random start: source=${layer_matched_source}, run=${control_run}, layers=${layer_spec}, topk=${topk}, q=${dynamic_exp_sharpness}, tau=${dynamic_tau}->${dynamic_late_tau}, tok=${max_new_tokens}"
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
    --seed "${seed}" \
    --intervention late_boost \
    --head-source file \
    --head-file "${random_head_file}" \
    --head-score-key "${head_score_key}" \
    --head-score-normalize "${head_score_normalize}" \
    --topk "${topk}" \
    --dynamic-strength "${dynamic_strength}" \
    --dynamic-context-mode "${dynamic_context_mode}" \
    --dynamic-tau "${dynamic_tau}" \
    --dynamic-exp-sharpness "${dynamic_exp_sharpness}" \
    --dynamic-late-boost-start "${dynamic_late_boost_start}" \
    --dynamic-late-boost-end "${dynamic_late_boost_end}" \
    --dynamic-late-boost-mode "${dynamic_late_boost_mode}" \
    --dynamic-late-tau "${dynamic_late_tau}" \
    --dynamic-score-power "${dynamic_score_power}" \
    --dynamic-redistribute "${dynamic_redistribute}" \
    "${renorm_args[@]}" \
    --use-head-scores \
    "${stats_args[@]}" \
    --sample-id-file "${sample_id_file}" \
    "${resume_args[@]}" \
    > "${result_path}/decode.log" 2>&1

  "${python_bin}" eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${result_path}/captions.jsonl" \
    --caption_file captions_val2014.json \
    > "${result_path}/chair.log" 2>&1

  echo "[GPU ${gpu}] CHAIR layer-matched random done -> ${result_path}"
done
