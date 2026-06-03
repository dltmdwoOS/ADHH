#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-llama-3-vila1.5-8b}
model_path=${MODEL_PATH:-Efficient-Large-Model/Llama-3-VILA1.5-8B}

dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
num_samples=${NUM_SAMPLES:-500}
result_path=${RESULT_PATH:-./results/$dataset/${model_name}_base_original_qa_n${num_samples}_txtattn_l0_l31_allheads}
analysis_path=$result_path/analysis
existing_sample_file=${EXISTING_SAMPLE_FILE:-../LLaVA/results/coco/llava-v1.5-7b_base_original_qa_n500_txtattn_l0_l31_allheads/misc/captions_eval_results.json}
txtattn_head_file=${TXTATTN_HEAD_FILE:-$result_path/candidate_heads_l0_l31.json}
txtattn_topk=${TXTATTN_TOPK:-0}
gpu_list_raw=${GPU_LIST:-"1"}
read -r -a gpu_list <<< "$gpu_list_raw"
num_chunks=${NUM_CHUNKS:-${#gpu_list[@]}}
resume=${RESUME:-true}
keep_merged_trace=${KEEP_MERGED_TRACE:-false}
attn_implementation=${VILA_ATTN_IMPLEMENTATION:-eager}
temperature=${TEMPERATURE:-0}
max_new_tokens=${MAX_NEW_TOKENS:-128}
num_beams=${NUM_BEAMS:-1}
seed=${SEED:-42}
conv_mode=${CONV_MODE:-auto}
prompt_text=${PROMPT_TEXT:-Please describe this image in detail.}

mkdir -p "$result_path" "$analysis_path"

if [[ ! -f "$txtattn_head_file" ]]; then
    TXTATTN_HEAD_FILE="$txtattn_head_file" python - <<'PY'
import json, os
path = os.environ["TXTATTN_HEAD_FILE"]
os.makedirs(os.path.dirname(path), exist_ok=True)
heads = [{"layer": l, "head": h, "candidate_global_rank": l * 32 + h + 1} for l in range(32) for h in range(32)]
with open(path, "w", encoding="utf-8") as f:
    json.dump({"heads": heads}, f, indent=2)
print(f"saved candidate heads -> {path} ({len(heads)} heads)")
PY
fi

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
    VILA_ATTN_IMPLEMENTATION="$attn_implementation" CUDA_VISIBLE_DEVICES="$gpu" python -m eval_scripts.eval_caption \
        --model-path "$model_path" \
        --model-name "$model_name" \
        --image-folder "$data_path/coco/train2014" \
        --caption_file_path "$data_path/coco/annotations/captions_train2014.json" \
        --annotation-dir "$data_path/coco/annotations" \
        --answers-file "$result_path/captions.chunk${chunk_idx}.jsonl" \
        --output-path "$chunk_analysis_path" \
        --dataset "$dataset" \
        --temperature "$temperature" \
        --conv-mode "$conv_mode" \
        --num_samples "$num_samples" \
        --save-sample-ids "$result_path/sample_ids.chunk${chunk_idx}.json" \
        --max_new_tokens "$max_new_tokens" \
        --num_beams "$num_beams" \
        --seed "$seed" \
        --prompt-text "$prompt_text" \
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
