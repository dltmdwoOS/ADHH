#!/usr/bin/env bash
set -euo pipefail

##########################
# 기본 설정
##########################

model_name=llava-v1.5-7b
model_path=liuhaotian/llava-v1.5-7b
dataset=coco
data_path=../dataset
seed=42
num_samples=500

# 논문 AD-HH: top-20 hallucination heads, tau = 0.4 [file:147]
adhh_topk=20
adhh_threshold=0.4

# baseline HH (코드에 내장된 HH set) vs 우리가 구한 HH (head_file)
# head_file=${HEAD_FILE:-../LLaVA_backup/results/coco/llava_3000/identify_attention_head/attribution_result.json}
#head_file=${HEAD_FILE:-../LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n3000/identify_attention_head/attribution_result.json}
head_file=${HEAD_FILE:-../LLaVA/results/coco/llava_3000/identify_attention_head/attribution_result.json}

gpu_list=(0)

sample_dir=./results/${dataset}/shared_samples
sample_id_file=${sample_dir}/val_seed${seed}_n${num_samples}.json
mkdir -p "${sample_dir}"

export PYTHONUNBUFFERED=1

##########################
# 샘플 id 고정 (Exp2와 동일 방식)
##########################

if [[ ! -f "${sample_id_file}" ]]; then
  python - <<PY
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

  local result_path=./results/${dataset}/${model_name}_adhh_${head_source}_k${adhh_topk}_tau${adhh_threshold}_n${num_samples}_real
  mkdir -p "${result_path}"

  local extra_head_args=()
  if [[ "${head_source}" == "file" ]]; then
    extra_head_args+=(--head-file "${head_file}")
  fi

  echo "[GPU ${gpu}] AD-HH start (${head_source} HH): topk=${adhh_topk}, tau=${adhh_threshold}"

  CUDA_VISIBLE_DEVICES="${gpu}" python -m eval_scripts.eval_caption_adhh \
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
    --intervention adhh \
    --topk "${adhh_topk}" \
    --text-threshold "${adhh_threshold}" \
    --head-source "${head_source}" \
    "${extra_head_args[@]}" \
    --sample-id-file "${sample_id_file}" \
    > "${result_path}/decode.log" 2>&1

  python eval_scripts/eval_utils/eval_chair.py \
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