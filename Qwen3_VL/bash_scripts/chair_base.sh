#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-qwen2.5-vl-7b}
model_path=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
seed=${SEED:-42}
num_samples=${NUM_SAMPLES:-500}
result_root=${RESULT_ROOT:-./results_base}
result_path=${RESULT_PATH:-${result_root}/${dataset}/${model_name}_base_n${num_samples}}
analysis_path=${result_path}/analysis
sample_dir=${SAMPLE_DIR:-./results/${dataset}/shared_samples}
sample_file=${SAMPLE_FILE:-${sample_dir}/val_seed${seed}_n${num_samples}.json}
gpu=${GPU:-0}
resume=${RESUME:-true}
max_new_tokens=${MAX_NEW_TOKENS:-128}
device_map=${DEVICE_MAP:-auto}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
python_bin=${PYTHON_BIN:-python}

export PYTHONUNBUFFERED=1
mkdir -p "${result_path}" "${analysis_path}" "${sample_dir}"

if [[ ! -f "${sample_file}" ]]; then
  "${python_bin}" - <<PY2
import json, os, random
from pycocotools.coco import COCO
caption_file = "${data_path}/coco/annotations/captions_val2014.json"
out_file = "${sample_file}"
random.seed(${seed})
coco = COCO(caption_file)
sampled = random.sample(coco.getImgIds(), ${num_samples})
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump([int(x) for x in sampled], f, indent=2)
print(f"saved sample ids -> {out_file}")
PY2
fi

resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption   --model-path "${model_path}"   --device-map "${device_map}"   --attn-implementation "${attn_implementation}"   --image-folder "${data_path}/coco/val2014"   --caption_file_path "${data_path}/coco/annotations/captions_val2014.json"   --annotation-dir "${data_path}/coco/annotations"   --answers-file "${result_path}/captions.jsonl"   --dataset "${dataset}"   --temperature 0   --num_samples "${num_samples}"   --seed "${seed}"   --max_new_tokens "${max_new_tokens}"   --use-existing-sample-file   --existing-sample-file "${sample_file}"   --save-sample-ids "${result_path}/sample_ids.json"   "${resume_args[@]}"   > "${result_path}/decode.log" 2>&1

"${python_bin}" eval_scripts/eval_utils/eval_chair.py   --annotation-dir "${data_path}/coco/annotations"   --answers-file "${result_path}/captions.jsonl"   --caption_file captions_val2014.json   > "${result_path}/chair.log" 2>&1

echo "Qwen vanilla CHAIR decoding finished -> ${result_path}"
