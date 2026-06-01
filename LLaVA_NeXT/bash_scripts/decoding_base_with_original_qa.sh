#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-llama3-llava-next-8b}
model_path=${MODEL_PATH:-lmms-lab/llama3-llava-next-8b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
existing_sample_file=${EXISTING_SAMPLE_FILE:-./results/coco/llama3-llava-next-8b_base_original_qa_n500_txtattn_l0_l31_allheads/misc/captions_eval_results.json}
num_samples=${NUM_SAMPLES:-500}
min_layer=${MIN_LAYER:-0}
max_layer=${MAX_LAYER:-31}
num_layers=${NUM_LAYERS:-32}
num_heads=${NUM_HEADS:-32}
result_root=${RESULT_ROOT:-./results}
result_path=${RESULT_PATH:-${result_root}/${dataset}/${model_name}_base_original_qa_n${num_samples}_txtattn_l${min_layer}_l${max_layer}_allheads}
analysis_path=${result_path}/analysis
score_root=${SCORE_ROOT:-${result_path}/surrogate_score_zoo}
txtattn_head_file=${TXTATTN_HEAD_FILE:-${result_path}/candidate_heads_l${min_layer}_l${max_layer}.json}
txtattn_topk=${TXTATTN_TOPK:-0}
gpu_list_raw=${GPU_LIST:-"0,1"}
read -r -a gpu_list <<< "${gpu_list_raw}"
num_chunks=${NUM_CHUNKS:-1}
resume=${RESUME:-true}
keep_merged_trace=${KEEP_MERGED_TRACE:-false}
max_new_tokens=${MAX_NEW_TOKENS:-128}
device_map=${DEVICE_MAP:-auto}
attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}
enable_attention_analysis=${ENABLE_ATTENTION_ANALYSIS:-false}
txtattn_trace_mode=${TXTATTN_TRACE_MODE:-last_row}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

mkdir -p "${result_path}" "${analysis_path}"

if [[ ! -f "${txtattn_head_file}" ]]; then
  python3 - <<PY2
import json, os
min_layer=${min_layer}
max_layer=${max_layer}
num_layers=${num_layers}
num_heads=${num_heads}
out_file="${txtattn_head_file}"
heads=[{"layer": l, "head": h, "score": 1.0} for l in range(min_layer, max_layer + 1) for h in range(num_heads)]
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w") as f:
    json.dump({"heads": heads, "meta": {"num_layers": num_layers, "num_heads": num_heads, "min_layer": min_layer, "max_layer": max_layer}}, f, indent=2)
print(f"saved candidate heads -> {out_file} ({len(heads)} heads)")
PY2
fi

resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi

attention_analysis_args=()
if [[ "${enable_attention_analysis}" == "true" ]]; then
  attention_analysis_args+=(--enable-attention-analysis --enable-pre-token-analysis)
fi
run_chunk() {
  local chunk_idx=$1
  local gpu=${gpu_list[$((chunk_idx % ${#gpu_list[@]}))]}
  local chunk_analysis_path="${analysis_path}/chunk${chunk_idx}"
  mkdir -p "${chunk_analysis_path}"

  echo "[chunk ${chunk_idx}/${num_chunks}] CUDA_VISIBLE_DEVICES=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -m eval_scripts.eval_caption \
    --model-path "${model_path}" \
    --device-map "${device_map}" \
    --attn-implementation "${attn_implementation}" \
    --image-folder "${data_path}/coco/train2014" \
    --caption_file_path "${data_path}/coco/annotations/captions_train2014.json" \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${result_path}/captions.chunk${chunk_idx}.jsonl" \
    --output-path "${chunk_analysis_path}" \
    --dataset "${dataset}" \
    --temperature 0 \
    --conv-mode "${CONV_MODE:-llava_llama_3}" \
    --num_samples "${num_samples}" \
    --save-sample-ids "${result_path}/sample_ids.chunk${chunk_idx}.json" \
    --max_new_tokens "${max_new_tokens}" \
    --use-existing-sample-file \
    --existing-sample-file "${existing_sample_file}" \
    ${attention_analysis_args[@]} \
    --enable-txtattn-trace \
    --txtattn-trace-mode "${txtattn_trace_mode}" \
    --txtattn-head-file "${txtattn_head_file}" \
    --txtattn-topk "${txtattn_topk}" \
    --txtattn-output-file "${result_path}/txtattn_trace.chunk${chunk_idx}.jsonl" \
    --txtattn-summary-file "${result_path}/txtattn_summary.chunk${chunk_idx}.json" \
    --num-chunks "${num_chunks}" \
    --chunk-idx "${chunk_idx}" \
    "${resume_args[@]}" \
    > "${result_path}/decode.chunk${chunk_idx}.log" 2>&1
}

pids=()
for ((chunk_idx=0; chunk_idx<num_chunks; chunk_idx++)); do
  run_chunk "${chunk_idx}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

: > "${result_path}/captions.jsonl"
trace_files=()
if [[ "${keep_merged_trace}" == "true" ]]; then
  : > "${result_path}/txtattn_trace.jsonl"
fi
for ((chunk_idx=0; chunk_idx<num_chunks; chunk_idx++)); do
  if [[ -f "${result_path}/captions.chunk${chunk_idx}.jsonl" ]]; then
    cat "${result_path}/captions.chunk${chunk_idx}.jsonl" >> "${result_path}/captions.jsonl"
  fi
  if [[ -f "${result_path}/txtattn_trace.chunk${chunk_idx}.jsonl" ]]; then
    trace_files+=("${result_path}/txtattn_trace.chunk${chunk_idx}.jsonl")
    if [[ "${keep_merged_trace}" == "true" ]]; then
      cat "${result_path}/txtattn_trace.chunk${chunk_idx}.jsonl" >> "${result_path}/txtattn_trace.jsonl"
    fi
  fi
done

python -m eval_scripts.summarize_txtattn_trace \
  --trace-file "${trace_files[@]}" \
  --head-file "${txtattn_head_file}" \
  --topk "${txtattn_topk}" \
  --summary-file "${result_path}/txtattn_summary.json"

python -m eval_scripts.compute_surrogate_score_zoo \
  --summary-file "${result_path}/txtattn_summary.json" \
  --output-dir "${score_root}"

python -m eval_scripts.build_layer_surrogate_combos \
  --summary-file "${result_path}/txtattn_summary.json" \
  --output-dir "${score_root}"

python eval_scripts/eval_utils/eval_chair.py \
  --annotation-dir "${data_path}/coco/annotations" \
  --answers-file "${result_path}/captions.jsonl" \
  --caption_file captions_train2014.json \
  > "${result_path}/chair.log" 2>&1

echo "Txt-attn tracing finished -> ${result_path}"
