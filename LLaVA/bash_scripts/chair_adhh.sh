#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

##########################
# 기본 설정
##########################

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-$(adhh_default_data_path)}
results_root=${RESULTS_ROOT:-$(adhh_default_results_root)}
seed=${SEED:-42}
num_samples=${NUM_SAMPLES:-500}
max_new_tokens=${MAX_NEW_TOKENS:-128}

# 논문 AD-HH: top-20 hallucination heads, tau = 0.4 [file:147]
adhh_topk=${ADHH_TOPK:-20}
adhh_threshold=${ADHH_THRESHOLD:-0.4}

# baseline HH (코드에 내장된 HH set) vs 우리가 구한 HH (head_file)
default_head_file=${results_root}/${dataset}/${model_name}/baselines/adhh_reproduced/attribution_result.json
if [[ ! -f "${default_head_file}" && -f "./results_deact/${dataset}/${model_name}/baselines/adhh_reproduced/attribution_result.json" ]]; then
  default_head_file=./results_deact/${dataset}/${model_name}/baselines/adhh_reproduced/attribution_result.json
fi
head_file=${HEAD_FILE:-${default_head_file}}

gpu_list=(0)
python_bin=${PYTHON_BIN:-python}
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin=python3
fi

sample_dir=${results_root}/${dataset}/${model_name}/shared_samples
sample_id_file=${sample_dir}/val_seed${seed}_n${num_samples}.json
mkdir -p "${sample_dir}"

export PYTHONUNBUFFERED=1

##########################
# 샘플 id 고정 (Exp2와 동일 방식)
##########################

if [[ ! -f "${sample_id_file}" ]]; then
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

##########################
# AD-HH 실행 함수
##########################

run_adhh_job() {
  local gpu=$1
  local head_source=$2   # default | file

  local baseline_name=adhh
  if [[ "${head_source}" == "file" ]]; then
    baseline_name=adhh_reproduced
  fi
  local result_path=${results_root}/${dataset}/${model_name}/baselines/${baseline_name}/tok${max_new_tokens}/tau${adhh_threshold}
  mkdir -p "${result_path}"

  local extra_head_args=()
  if [[ "${head_source}" == "file" ]]; then
    extra_head_args+=(--head-file "${head_file}")
  fi

  echo "[GPU ${gpu}] AD-HH start (${head_source} HH): topk=${adhh_topk}, tau=${adhh_threshold}"

  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption_adhh \
    --model-path "${model_path}" \
    --image-folder "${data_path}/coco/val2014" \
    --caption_file_path "${data_path}/coco/annotations/captions_val2014.json" \
    --answers-file "${result_path}/captions.jsonl" \
    --dataset "${dataset}" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --num_samples "${num_samples}" \
    --seed "${seed}" \
    --num-workers 4 \
    --max_new_tokens "${max_new_tokens}" \
    --intervention adhh \
    --topk "${adhh_topk}" \
    --text-threshold "${adhh_threshold}" \
    --head-source "${head_source}" \
    "${extra_head_args[@]}" \
    --sample-id-file "${sample_id_file}" \
    > "${result_path}/decode.log" 2>&1

  "${python_bin}" eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${result_path}/captions.jsonl" \
    --caption_file captions_val2014.json \
    > "${result_path}/chair.log" 2>&1

  echo "[GPU ${gpu}] AD-HH done (${head_source} HH)"
}

pids=()

# our HH: 우리가 구한 head_file 사용 (head_source=file)
run_adhh_job "${gpu_list[0]}" "file" &
pids+=("$!")

wait "${pids[@]}"

echo "AD-HH decoding finished for: baseline (default HH) and our HH (file)."
