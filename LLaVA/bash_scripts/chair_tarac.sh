#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-$(adhh_default_data_path)}
seed=${SEED:-42}
gpu=${GPU:-1}
python_bin=$(adhh_python_bin)
num_samples=${NUM_SAMPLES:-500}
max_new_tokens=${MAX_NEW_TOKENS:-128}
alpha=${TARAC_ALPHA:-0.5}
beta=${TARAC_BETA:-0.5}
start_layer=${TARAC_START_LAYER:-9}
end_layer=${TARAC_END_LAYER:-16}
resume=${RESUME:-false}
dry_run=${DRY_RUN:-false}
results_root=${RESULTS_ROOT:-$(adhh_default_results_root)}

sample_dir=${results_root}/${dataset}/${model_name}/shared_samples
sample_id_file=${SAMPLE_ID_FILE:-${sample_dir}/val_seed${seed}_n${num_samples}.json}
mkdir -p "${sample_dir}"

alpha_slug=$("${python_bin}" - <<PY
v = float("${alpha}")
print((f"{v:.4f}".rstrip("0").rstrip(".")).replace(".", ""))
PY
)
beta_slug=$("${python_bin}" - <<PY
v = float("${beta}")
print((f"{v:.4f}".rstrip("0").rstrip(".")).replace(".", ""))
PY
)

result_path=${RESULT_PATH:-${results_root}/${dataset}/${model_name}/baselines/tarac/tok${max_new_tokens}/alpha${alpha_slug}_beta${beta_slug}_l${start_layer}-${end_layer}}
mkdir -p "${result_path}"

resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi

export PYTHONUNBUFFERED=1

if [[ "${dry_run}" == "true" ]]; then
  echo "[dry-run] CHAIR TARAC would write -> ${result_path}"
  echo "[dry-run] data_path=${data_path}"
  echo "[dry-run] sample_id_file=${sample_id_file}"
  exit 0
fi

if [[ ! -f "${sample_id_file}" ]]; then
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

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption_dynamic \
  --model-path "${model_path}" \
  --image-folder "${data_path}/coco/val2014" \
  --caption_file_path "${data_path}/coco/annotations/captions_val2014.json" \
  --answers-file "${result_path}/captions.jsonl" \
  --dataset "${dataset}" \
  --temperature 0 \
  --conv-mode vicuna_v1 \
  --num_samples "${num_samples}" \
  --seed "${seed}" \
  --max_new_tokens "${max_new_tokens}" \
  --intervention tarac \
  --topk 0 \
  --tarac-alpha "${alpha}" \
  --tarac-beta "${beta}" \
  --tarac-start-layer "${start_layer}" \
  --tarac-end-layer "${end_layer}" \
  --sample-id-file "${sample_id_file}" \
  "${resume_args[@]}" \
  > "${result_path}/decode.log" 2>&1

"${python_bin}" eval_scripts/eval_utils/eval_chair.py \
  --annotation-dir "${data_path}/coco/annotations" \
  --answers-file "${result_path}/captions.jsonl" \
  --caption_file captions_val2014.json \
  > "${result_path}/chair.log" 2>&1

echo "CHAIR TARAC done -> ${result_path}"
