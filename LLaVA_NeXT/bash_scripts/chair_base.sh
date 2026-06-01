#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-llama3-llava-next-8b}
model_path=${MODEL_PATH:-lmms-lab/llama3-llava-next-8b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
seed=${SEED:-42}
num_samples=${NUM_SAMPLES:-500}
result_root=${RESULT_ROOT:-./results_base}
result_path=${RESULT_PATH:-${result_root}/${dataset}/${model_name}_base_n${num_samples}}
analysis_path=${result_path}/analysis
sample_dir=${SAMPLE_DIR:-./results/${dataset}/shared_samples}
sample_file=${SAMPLE_FILE:-${sample_dir}/val_seed${seed}_n${num_samples}_samples.json}
gpu=${GPU:-0}
resume=${RESUME:-true}
max_new_tokens=${MAX_NEW_TOKENS:-128}
device_map=${DEVICE_MAP:-auto}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
num_workers=${NUM_WORKERS:-4}
conv_mode=${CONV_MODE:-llava_llama_3}

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}
python_bin=${PYTHON_BIN:-python}
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin=/venv/ha/bin/python
fi

mkdir -p "${result_path}" "${analysis_path}" "${sample_dir}"

if [[ ! -f "${sample_file}" ]]; then
  "${python_bin}" - <<PY2
import json, os, random
from pycocotools.coco import COCO
caption_file = "${data_path}/coco/annotations/captions_val2014.json"
out_file = "${sample_file}"
seed = ${seed}
num_samples = ${num_samples}
random.seed(seed)
coco = COCO(caption_file)
sampled = random.sample(coco.getImgIds(), num_samples)
samples = []
for image_id in sampled:
    image = coco.loadImgs(image_id)[0]["file_name"]
    samples.append({"question_id": int(image_id), "image": image, "prompt": "Please describe this image in detail."})
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({"samples": samples, "seed": seed, "num_samples": num_samples}, f, indent=2)
print(f"saved sample file -> {out_file}")
PY2
fi

resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption \
  --model-path "${model_path}" \
  --device-map "${device_map}" \
  --attn-implementation "${attn_implementation}" \
  --image-folder "${data_path}/coco/val2014" \
  --caption_file_path "${data_path}/coco/annotations/captions_val2014.json" \
  --annotation-dir "${data_path}/coco/annotations" \
  --answers-file "${result_path}/captions.jsonl" \
  --output-path "${analysis_path}" \
  --dataset "${dataset}" \
  --temperature 0 \
  --conv-mode "${conv_mode}" \
  --num_samples "${num_samples}" \
  --seed "${seed}" \
  --max_new_tokens "${max_new_tokens}" \
  --num-workers "${num_workers}" \
  --use-existing-sample-file \
  --existing-sample-file "${sample_file}" \
  --save-sample-ids "${result_path}/sample_ids.json" \
  "${resume_args[@]}" \
  > "${result_path}/decode.log" 2>&1

"${python_bin}" eval_scripts/eval_utils/eval_chair.py \
  --annotation-dir "${data_path}/coco/annotations" \
  --answers-file "${result_path}/captions.jsonl" \
  --caption_file captions_val2014.json \
  > "${result_path}/chair.log" 2>&1

echo "Vanilla CHAIR decoding finished -> ${result_path}"
