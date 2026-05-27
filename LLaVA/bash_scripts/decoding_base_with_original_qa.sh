#!/usr/bin/env bash
set -euo pipefail

model_name=llava-v1.5-7b
model_path=liuhaotian/llava-v1.5-7b
# model_name=llava-v1.5-13b
# model_path=liuhaotian/llava-v1.5-13b
# model_name=llava-v1.6-34b
# model_path=liuhaotian/llava-v1.6-34b

dataset=coco
data_path=../dataset
existing_sample_file=${EXISTING_SAMPLE_FILE:-../LLaVA_backup/results/coco/llava_3000/captions_eval_results.json}
num_samples=3000
result_path=./results/$dataset/${model_name}_base_original_qa_n${num_samples}_txtattn
analysis_path=$result_path/analysis
txtattn_head_file=${TXTATTN_HEAD_FILE:-./results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/ranked_heads_combo_mean_txtraw_Cratio.json}
txtattn_topk=${TXTATTN_TOPK:-100}

mkdir -p "$result_path" "$analysis_path"

CUDA_VISIBLE_DEVICES='1' python -m eval_scripts.eval_caption \
    --model-path "$model_path" \
    --image-folder "$data_path/coco/train2014" \
    --caption_file_path "$data_path/coco/annotations/captions_train2014.json" \
    --annotation-dir "$data_path/coco/annotations" \
    --answers-file "$result_path/captions.jsonl" \
    --output-path "$analysis_path" \
    --dataset "$dataset" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --num_samples "$num_samples" \
    --save-sample-ids "$result_path/sample_ids.json" \
    --max_new_tokens 256 \
    --use-existing-sample-file \
    --existing-sample-file "$existing_sample_file" \
    --enable-attention-analysis \
    --enable-pre-token-analysis \
    --enable-txtattn-trace \
    --txtattn-head-file "$txtattn_head_file" \
    --txtattn-topk "$txtattn_topk" \
    --txtattn-output-file "$result_path/txtattn_trace.jsonl" \
    --txtattn-summary-file "$result_path/txtattn_summary.json"

python eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "$data_path/coco/annotations" \
    --answers-file "$result_path/captions.jsonl" \
    --caption_file captions_train2014.json
