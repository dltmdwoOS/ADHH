#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-llama-3-vila1.5-8b}
model_path=${MODEL_PATH:-Efficient-Large-Model/Llama-3-VILA1.5-8B}

dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
num_samples=${NUM_SAMPLES:-500}
result_root=${RESULT_ROOT:-./results_base}
result_path=${RESULT_PATH:-$result_root/$dataset/${model_name}_base_n${num_samples}}
attn_implementation=${VILA_ATTN_IMPLEMENTATION:-eager}

gpu=${GPU:-1}
temperature=${TEMPERATURE:-0}
max_new_tokens=${MAX_NEW_TOKENS:-128}
num_beams=${NUM_BEAMS:-1}
seed=${SEED:-42}
conv_mode=${CONV_MODE:-auto}
prompt_text=${PROMPT_TEXT:-Please describe this image in detail.}
resume=${RESUME:-false}
existing_sample_file=${EXISTING_SAMPLE_FILE:-../LLaVA/results/coco/llava-v1.5-7b_base_n500_txtattn/sample_ids.json}

mkdir -p "$result_path"

resume_args=()
if [[ "$resume" == "true" ]]; then
    resume_args+=(--resume)
fi

sample_args=(--num_samples "$num_samples")
if [[ -n "$existing_sample_file" ]]; then
    sample_args+=(--use-existing-sample-file --existing-sample-file "$existing_sample_file")
fi

VILA_ATTN_IMPLEMENTATION="$attn_implementation" CUDA_VISIBLE_DEVICES="$gpu" python -m eval_scripts.eval_caption_base \
    --model-path "$model_path" \
    --model-name "$model_name" \
    --image-folder "$data_path/coco/val2014" \
    --caption_file_path "$data_path/coco/annotations/captions_val2014.json" \
    --answers-file "$result_path/captions.jsonl" \
    --dataset "$dataset" \
    --temperature "$temperature" \
    --conv-mode "$conv_mode" \
    --save-sample-ids "$result_path/sample_ids.json" \
    --max_new_tokens "$max_new_tokens" \
    --num_beams "$num_beams" \
    --seed "$seed" \
    --prompt-text "$prompt_text" \
    "${sample_args[@]}" \
    "${resume_args[@]}" \
    2>&1 | tee "$result_path/decode.log"

python eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "$data_path/coco/annotations" \
    --answers-file "$result_path/captions.jsonl" \
    --caption_file captions_val2014.json \
    2>&1 | tee "$result_path/chair.log"
