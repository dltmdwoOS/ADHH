#!/usr/bin/env bash
set -euo pipefail

model_name=${MODEL_NAME:-qwen2.5-vl-7b}
model_path=${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-../dataset}
existing_sample_file=${EXISTING_SAMPLE_FILE:-../LLaVA_NeXT/results/coco/llama3-llava-next-8b_base_original_qa_n500_txtattn_l0_l31_allheads/misc/captions_eval_results.json}
num_samples=${NUM_SAMPLES:-500}
min_layer=${MIN_LAYER:-0}
max_layer=${MAX_LAYER:-27}
num_layers=${NUM_LAYERS:-28}
num_heads=${NUM_HEADS:-28}
result_root=${RESULT_ROOT:-./results}
result_path=${RESULT_PATH:-${result_root}/${dataset}/${model_name}_base_original_qa_n${num_samples}_txtattn_l${min_layer}_l${max_layer}_allheads}
score_root=${SCORE_ROOT:-${result_path}/surrogate_score_zoo}
txtattn_head_file=${TXTATTN_HEAD_FILE:-${result_path}/candidate_heads_l${min_layer}_l${max_layer}.json}
txtattn_topk=${TXTATTN_TOPK:-0}
gpu_list_raw=${GPU_LIST:-"0 1"}
read -r -a gpu_list <<< "${gpu_list_raw}"
num_chunks=${NUM_CHUNKS:-2}
resume=${RESUME:-true}
keep_merged_trace=${KEEP_MERGED_TRACE:-false}
max_new_tokens=${MAX_NEW_TOKENS:-128}
device_map=${DEVICE_MAP:-auto}
attn_implementation=${ATTN_IMPLEMENTATION:-eager}
python_bin=${PYTHON_BIN:-python}

export PYTHONUNBUFFERED=1
mkdir -p "${result_path}" "${score_root}"

if [[ ! -f "${txtattn_head_file}" ]]; then
  "${python_bin}" - <<PY2
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

run_chunk() {
  local chunk_idx=$1
  local gpu=${gpu_list[$((chunk_idx % ${#gpu_list[@]}))]}
  echo "[chunk ${chunk_idx}/${num_chunks}] CUDA_VISIBLE_DEVICES=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_caption \
    --model-path "${model_path}" \
    --device-map "${device_map}" \
    --attn-implementation "${attn_implementation}" \
    --image-folder "${data_path}/coco/train2014" \
    --caption_file_path "${data_path}/coco/annotations/captions_train2014.json" \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${result_path}/captions.chunk${chunk_idx}.jsonl" \
    --dataset "${dataset}" \
    --temperature 0 \
    --num_samples "${num_samples}" \
    --save-sample-ids "${result_path}/sample_ids.chunk${chunk_idx}.json" \
    --max_new_tokens "${max_new_tokens}" \
    --use-existing-sample-file \
    --existing-sample-file "${existing_sample_file}" \
    --enable-txtattn-trace \
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
  [[ -f "${result_path}/captions.chunk${chunk_idx}.jsonl" ]] && cat "${result_path}/captions.chunk${chunk_idx}.jsonl" >> "${result_path}/captions.jsonl"
  if [[ -f "${result_path}/txtattn_trace.chunk${chunk_idx}.jsonl" ]]; then
    trace_files+=("${result_path}/txtattn_trace.chunk${chunk_idx}.jsonl")
    [[ "${keep_merged_trace}" == "true" ]] && cat "${result_path}/txtattn_trace.chunk${chunk_idx}.jsonl" >> "${result_path}/txtattn_trace.jsonl"
  fi
done

"${python_bin}" -m eval_scripts.summarize_txtattn_trace \
  --trace-file "${trace_files[@]}" \
  --head-file "${txtattn_head_file}" \
  --topk "${txtattn_topk}" \
  --summary-file "${result_path}/txtattn_summary.json"

"${python_bin}" -m eval_scripts.compute_surrogate_score_zoo \
  --summary-file "${result_path}/txtattn_summary.json" \
  --output-dir "${score_root}"

"${python_bin}" -m eval_scripts.build_layer_surrogate_combos \
  --summary-file "${result_path}/txtattn_summary.json" \
  --output-dir "${score_root}"

"${python_bin}" eval_scripts/eval_utils/eval_chair.py \
  --annotation-dir "${data_path}/coco/annotations" \
  --answers-file "${result_path}/captions.jsonl" \
  --caption_file captions_train2014.json \
  > "${result_path}/chair.log" 2>&1 || true

echo "Qwen txt-attn tracing finished -> ${result_path}"
