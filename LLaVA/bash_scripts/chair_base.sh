#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
# model_name=llava-v1.5-13b
# model_path=liuhaotian/llava-v1.5-13b
# model_name=llava-v1.6-34b
# model_path=liuhaotian/llava-v1.6-34b

dataset=coco
data_path=${DATA_PATH:-$(adhh_default_data_path)}
num_samples=${NUM_SAMPLES:-500}
max_new_tokens=${MAX_NEW_TOKENS:-128}
results_root=${RESULTS_ROOT:-$(adhh_default_results_root)}
result_path=${RESULT_PATH:-${results_root}/${dataset}/${model_name}/baselines/greedy/tok${max_new_tokens}}
analysis_path=$result_path/analysis
txtattn_head_file=${TXTATTN_HEAD_FILE:-${results_root}/analysis/selected_heads/${model_name}/ranked_heads_global__itext_all__C_toi_HminusG_signed.json}
txtattn_topk=${TXTATTN_TOPK:-0}
python_bin=${PYTHON_BIN:-python}
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    python_bin=python3
fi

mkdir -p "$result_path" "$analysis_path"

CUDA_VISIBLE_DEVICES='0' "${python_bin}" -m eval_scripts.eval_caption \
    --model-path "$model_path" \
    --image-folder "$data_path/coco/val2014" \
    --caption_file_path "$data_path/coco/annotations/captions_val2014.json" \
    --annotation-dir "$data_path/coco/annotations" \
    --answers-file "$result_path/captions.jsonl" \
    --output-path "$analysis_path" \
    --dataset "$dataset" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --num_samples "$num_samples" \
    --save-sample-ids "$result_path/sample_ids.json" \
    --max_new_tokens "$max_new_tokens" \
    --enable-attention-analysis \
    --enable-txtattn-trace \
    --txtattn-head-file "$txtattn_head_file" \
    --txtattn-topk "$txtattn_topk" \
    --txtattn-output-file "$result_path/txtattn_trace.jsonl" \
    --txtattn-summary-file "$result_path/txtattn_summary.json"

"${python_bin}" eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "$data_path/coco/annotations" \
    --answers-file "$result_path/captions.jsonl" \
    --caption_file captions_val2014.json
