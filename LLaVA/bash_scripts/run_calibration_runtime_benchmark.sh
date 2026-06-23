#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-$(adhh_default_data_path)}
results_root=${RESULTS_ROOT:-$(adhh_default_results_root)}
seed=${SEED:-42}
train_num_samples=${TRAIN_NUM_SAMPLES:-500}
max_new_tokens=${MAX_NEW_TOKENS:-128}
gpu=${GPU:-0}
num_workers=${NUM_WORKERS:-4}
layer_spec=${LAYER_SPEC:-9:16}
num_heads=${NUM_HEADS:-32}
topk=${TOPK:-100}
auto_tau_topk_list=${AUTO_TAU_TOPK_LIST:-100}
auto_tau_calibration_scope=${AUTO_TAU_CALIBRATION_SCOPE:-selected_head}
auto_tau_calibration_bucket=${AUTO_TAU_CALIBRATION_BUCKET:-all}
auto_tau_hi_quantile=${AUTO_TAU_HI_QUANTILE:-q66}
auto_tau_lo_quantile=${AUTO_TAU_LO_QUANTILE:-q33}
auto_tau_round_step=${AUTO_TAU_ROUND_STEP:-0.01}
auto_tau_round_mode=${AUTO_TAU_ROUND_MODE:-floor}
keep_trace=${KEEP_TRACE:-false}
dry_run=${DRY_RUN:-false}
python_bin=${PYTHON_BIN:-$(adhh_python_bin)}

caption_file=${data_path}/coco/annotations/captions_train2014.json
image_folder=${data_path}/coco/train2014

if [[ "${dry_run}" != "true" ]]; then
  [[ -d "${image_folder}" ]] || { echo "Missing train image folder: ${image_folder}" >&2; exit 1; }
  [[ -f "${caption_file}" ]] || { echo "Missing train caption file: ${caption_file}" >&2; exit 1; }
fi

layer_slug=$("${python_bin}" - <<PY
layers = []
for part in "${layer_spec}".replace(";", ",").split(","):
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
seen = set()
layers = [x for x in layers if not (x in seen or seen.add(x))]
if len(layers) > 1 and layers == list(range(layers[0], layers[-1] + 1)):
    print(f"l{layers[0]}_l{layers[-1]}")
else:
    print("l" + "_l".join(str(x) for x in layers))
PY
)

out_dir=${OUTPUT_DIR:-${results_root}/${dataset}/${model_name}/calibration_runtime/${layer_slug}_train_n${train_num_samples}/last_row}
sample_dir=${results_root}/${dataset}/${model_name}/shared_samples
sample_file=${SAMPLE_FILE:-${sample_dir}/train_seed${seed}_n${train_num_samples}.json}
head_file=${HEAD_FILE:-${out_dir}/candidate_heads_${layer_slug}.json}
answers_file=${out_dir}/captions.jsonl
trace_file=${out_dir}/txtattn_trace.jsonl
summary_file=${out_dir}/txtattn_summary.json
surrogate_dir=${out_dir}/surrogate_score_zoo
tau_file=${out_dir}/dynamic_tau_estimate.json
timings_file=${out_dir}/calibration_timings.jsonl
runtime_summary_file=${out_dir}/calibration_runtime_summary.json
runtime_table_file=${out_dir}/calibration_runtime_summary.md

mkdir -p "${out_dir}" "${sample_dir}" "${surrogate_dir}"
: > "${timings_file}"

if [[ ! -f "${sample_file}" ]]; then
  echo "[sample] creating fixed train sample file: ${sample_file}"
  if [[ "${dry_run}" == "true" ]]; then
    echo "[dry-run] would create ${sample_file}"
  else
    "${python_bin}" - <<PY
import json
import random
from pycocotools.coco import COCO

random.seed(${seed})
coco = COCO("${caption_file}")
sampled = random.sample(coco.getImgIds(), ${train_num_samples})
id_to_img = {int(img["id"]): img for img in coco.dataset["images"]}
records = [
    {
        "question_id": int(image_id),
        "image": id_to_img[int(image_id)]["file_name"],
        "prompt": "Please describe this image in detail.",
    }
    for image_id in sampled
]
with open("${sample_file}", "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)
print(f"saved sample ids -> ${sample_file}")
PY
  fi
else
  echo "[sample] reusing fixed train sample file: ${sample_file}"
fi

if [[ ! -f "${head_file}" ]]; then
  echo "[heads] creating layer candidate head file: ${head_file}"
  if [[ "${dry_run}" == "true" ]]; then
    echo "[dry-run] would create ${head_file}"
  else
    "${python_bin}" - <<PY
import json
import os

layers = []
for part in "${layer_spec}".replace(";", ",").split(","):
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
seen = set()
layers = [x for x in layers if not (x in seen or seen.add(x))]
heads = [{"layer": int(layer), "head": int(head)} for layer in layers for head in range(${num_heads})]
obj = {
    "heads": heads,
    "config": {
        "layer_spec": "${layer_spec}",
        "layers": layers,
        "num_heads_per_layer": ${num_heads},
        "num_candidate_heads": len(heads),
        "note": "Candidate text-side actuator heads for calibration runtime measurement.",
    },
}
os.makedirs(os.path.dirname("${head_file}"), exist_ok=True)
with open("${head_file}", "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2)
print(f"saved candidate heads -> ${head_file} ({len(heads)} heads)")
PY
  fi
else
  echo "[heads] reusing candidate head file: ${head_file}"
fi

if [[ "${keep_trace}" == "true" ]]; then
  txtattn_output_file="${trace_file}"
else
  txtattn_output_file="/dev/null"
fi

write_config() {
  "${python_bin}" - <<PY
import json
import os

config = {
    "model_name": "${model_name}",
    "model_path": "${model_path}",
    "dataset": "${dataset}",
    "data_path": "${data_path}",
    "image_folder": "${image_folder}",
    "caption_file": "${caption_file}",
    "seed": ${seed},
    "train_num_samples": ${train_num_samples},
    "max_new_tokens": ${max_new_tokens},
    "gpu": "${gpu}",
    "num_workers": ${num_workers},
    "layer_spec": "${layer_spec}",
    "layer_slug": "${layer_slug}",
    "num_heads_per_layer": ${num_heads},
    "head_file": "${head_file}",
    "sample_file": "${sample_file}",
    "trace_mode": "last_row",
    "keep_trace": "${keep_trace}" == "true",
    "txtattn_output_file": "${txtattn_output_file}",
    "summary_file": "${summary_file}",
    "surrogate_dir": "${surrogate_dir}",
    "tau_file": "${tau_file}",
    "notes": [
        "This benchmark measures calibration trace collection and head-selection postprocess only.",
        "It uses last-row txt-attn tracing without full output_attentions, attention-analysis, or pre-token analysis.",
        "The txt_img_ratio definition follows the current text-side actuator definition: I_text / image_attn.",
    ],
}
with open("${out_dir}/calibration_benchmark_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
PY
}

record_timing() {
  local name=$1
  local seconds=$2
  "${python_bin}" - <<PY
import json
with open("${timings_file}", "a", encoding="utf-8") as f:
    f.write(json.dumps({"stage": "${name}", "seconds": float("${seconds}")}) + "\\n")
PY
}

elapsed_since() {
  local start=$1
  local end=$2
  "${python_bin}" - <<PY
print(float("${end}") - float("${start}"))
PY
}

run_timed() {
  local name=$1
  local log_file=$2
  shift 2
  echo "[timing] start ${name}"
  if [[ "${dry_run}" == "true" ]]; then
    printf '[dry-run]' | tee "${log_file}"
    printf ' %q' "$@" | tee -a "${log_file}"
    printf '\n' | tee -a "${log_file}"
    record_timing "${name}" "0"
    return 0
  fi
  local start
  local end
  local seconds
  start=$(date +%s.%N)
  "$@" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  end=$(date +%s.%N)
  seconds=$(elapsed_since "${start}" "${end}")
  record_timing "${name}" "${seconds}"
  echo "[timing] ${name}: ${seconds}s"
  return "${status}"
}

write_config

run_timed "trace_summary" "${out_dir}/trace_summary.log" \
  env CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption \
    --model-path "${model_path}" \
    --image-folder "${image_folder}" \
    --caption_file_path "${caption_file}" \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${answers_file}" \
    --dataset "${dataset}" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --num_samples "${train_num_samples}" \
    --max_new_tokens "${max_new_tokens}" \
    --seed "${seed}" \
    --num-workers "${num_workers}" \
    --use-existing-sample-file \
    --existing-sample-file "${sample_file}" \
    --enable-txtattn-trace \
    --txtattn-trace-mode last_row \
    --txtattn-head-file "${head_file}" \
    --txtattn-topk 0 \
    --txtattn-output-file "${txtattn_output_file}" \
    --txtattn-summary-file "${summary_file}" \
    --num-chunks 1 \
    --chunk-idx 0

run_timed "surrogate_score_zoo" "${out_dir}/surrogate_score_zoo.log" \
  "${python_bin}" eval_scripts/compute_surrogate_score_zoo.py \
    --summary-file "${summary_file}" \
    --output-dir "${surrogate_dir}"

head_score_key=global__itext_all__C_toi_HminusG_signed
ranked_head_file=${surrogate_dir}/ranked_heads_${head_score_key}.json
run_timed "dynamic_tau_estimate" "${out_dir}/dynamic_tau_estimate.log" \
  "${python_bin}" eval_scripts/estimate_dynamic_tau.py \
    --summary-file "${summary_file}" \
    --head-file "${ranked_head_file}" \
    --topk "${topk}" \
    --topk-list "${auto_tau_topk_list}" \
    --calibration-scope "${auto_tau_calibration_scope}" \
    --calibration-bucket "${auto_tau_calibration_bucket}" \
    --hi-quantile "${auto_tau_hi_quantile}" \
    --lo-quantile "${auto_tau_lo_quantile}" \
    --output-file "${tau_file}" \
    --round-step "${auto_tau_round_step}" \
    --round-mode "${auto_tau_round_mode}"

"${python_bin}" - <<PY
import json
from pathlib import Path

timings_path = Path("${timings_file}")
rows = [json.loads(line) for line in timings_path.read_text().splitlines() if line.strip()]
total = sum(float(row["seconds"]) for row in rows)
summary = {
    "total_seconds": total,
    "stages": rows,
    "config_file": "${out_dir}/calibration_benchmark_config.json",
    "summary_file": "${summary_file}",
    "ranked_head_file": "${ranked_head_file}",
    "tau_file": "${tau_file}",
}
Path("${runtime_summary_file}").write_text(json.dumps(summary, indent=2), encoding="utf-8")

md = [
    "# Calibration Runtime Summary",
    "",
    "Output: " + Path("${out_dir}").as_posix(),
    "",
    "| Stage | Seconds | Minutes |",
    "|---|---:|---:|",
]
for row in rows:
    seconds = float(row["seconds"])
    md.append(f"| {row['stage']} | {seconds:.3f} | {seconds / 60.0:.2f} |")
md.append(f"| total | {total:.3f} | {total / 60.0:.2f} |")
md.append("")
md.append("Trace collection uses last-row txt-attn only; full attention analysis and pre-token analysis are disabled.")
Path("${runtime_table_file}").write_text("\\n".join(md) + "\\n", encoding="utf-8")
print("\\n".join(md))
PY

echo "Calibration runtime benchmark prepared outputs under: ${out_dir}"
