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
existing_sample_file=${EXISTING_SAMPLE_FILE:-./results/coco/llava-v1.5-7b_base_original_qa_n500_txtattn_l0_l31_allheads/misc/captions_eval_results.json}
num_samples=500
result_path=${RESULT_PATH:-./results/$dataset/${model_name}_base_original_qa_n${num_samples}_txtattn_l0_l31_allheads}
analysis_path=$result_path/analysis
txtattn_head_file=${TXTATTN_HEAD_FILE:-./results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/candidate_heads_l0_l31.json}
txtattn_topk=${TXTATTN_TOPK:-0}
gpu_list_raw=${GPU_LIST:-"0 1"}
read -r -a gpu_list <<< "$gpu_list_raw"
num_chunks=${NUM_CHUNKS:-${#gpu_list[@]}}
resume=${RESUME:-true}
keep_merged_trace=${KEEP_MERGED_TRACE:-false}

mkdir -p "$result_path" "$analysis_path"

resume_args=()
if [[ "$resume" == "true" ]]; then
    resume_args+=(--resume)
    if [[ ! -f "$result_path/captions.chunk0.jsonl" && -f "$result_path/captions.jsonl" ]]; then
        cp "$result_path/captions.jsonl" "$result_path/captions.chunk0.jsonl"
        echo "[resume] bootstrapped captions.chunk0.jsonl from existing captions.jsonl"
    fi
    if [[ ! -f "$result_path/txtattn_trace.chunk0.jsonl" && -f "$result_path/txtattn_trace.jsonl" ]]; then
        cp "$result_path/txtattn_trace.jsonl" "$result_path/txtattn_trace.chunk0.jsonl"
        echo "[resume] bootstrapped txtattn_trace.chunk0.jsonl from existing txtattn_trace.jsonl"
    fi
fi

run_chunk() {
    local chunk_idx=$1
    local gpu=${gpu_list[$((chunk_idx % ${#gpu_list[@]}))]}
    local chunk_analysis_path="$analysis_path/chunk${chunk_idx}"
    mkdir -p "$chunk_analysis_path"

    echo "[chunk ${chunk_idx}/${num_chunks}] CUDA_VISIBLE_DEVICES=${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" python -m eval_scripts.eval_caption \
        --model-path "$model_path" \
        --image-folder "$data_path/coco/train2014" \
        --caption_file_path "$data_path/coco/annotations/captions_train2014.json" \
        --annotation-dir "$data_path/coco/annotations" \
        --answers-file "$result_path/captions.chunk${chunk_idx}.jsonl" \
        --output-path "$chunk_analysis_path" \
        --dataset "$dataset" \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num_samples "$num_samples" \
        --save-sample-ids "$result_path/sample_ids.chunk${chunk_idx}.json" \
        --max_new_tokens 128 \
        --use-existing-sample-file \
        --existing-sample-file "$existing_sample_file" \
        --enable-attention-analysis \
        --enable-pre-token-analysis \
        --enable-txtattn-trace \
        --txtattn-head-file "$txtattn_head_file" \
        --txtattn-topk "$txtattn_topk" \
        --txtattn-output-file "$result_path/txtattn_trace.chunk${chunk_idx}.jsonl" \
        --txtattn-summary-file "$result_path/txtattn_summary.chunk${chunk_idx}.json" \
        --num-chunks "$num_chunks" \
        --chunk-idx "$chunk_idx" \
        "${resume_args[@]}" \
        > "$result_path/decode.chunk${chunk_idx}.log" 2>&1
}

pids=()
for ((chunk_idx=0; chunk_idx<num_chunks; chunk_idx++)); do
    run_chunk "$chunk_idx" &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

: > "$result_path/captions.jsonl"
trace_files=()
if [[ "$keep_merged_trace" == "true" ]]; then
    : > "$result_path/txtattn_trace.jsonl"
fi
for ((chunk_idx=0; chunk_idx<num_chunks; chunk_idx++)); do
    if [[ -f "$result_path/captions.chunk${chunk_idx}.jsonl" ]]; then
        cat "$result_path/captions.chunk${chunk_idx}.jsonl" >> "$result_path/captions.jsonl"
    fi
    if [[ -f "$result_path/txtattn_trace.chunk${chunk_idx}.jsonl" ]]; then
        trace_files+=("$result_path/txtattn_trace.chunk${chunk_idx}.jsonl")
        if [[ "$keep_merged_trace" == "true" ]]; then
            cat "$result_path/txtattn_trace.chunk${chunk_idx}.jsonl" >> "$result_path/txtattn_trace.jsonl"
        fi
    fi
done

python -m eval_scripts.summarize_txtattn_trace \
    --trace-file "${trace_files[@]}" \
    --head-file "$txtattn_head_file" \
    --topk "$txtattn_topk" \
    --summary-file "$result_path/txtattn_summary.json"

python eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "$data_path/coco/annotations" \
    --answers-file "$result_path/captions.jsonl" \
    --caption_file captions_train2014.json
