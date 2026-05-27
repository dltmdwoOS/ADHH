#!/usr/bin/env bash
set -euo pipefail

model_name=llava-v1.5-7b
dataset=${DATASET:-coco}
head_score_key=${HEAD_SCORE_KEY:-combo_mean_txtraw_Cratio}
result_root=${RESULT_ROOT:-./results/${dataset}/${model_name}_base_n500_txtattn}
trace_file=${TRACE_FILE:-${result_root}/txtattn_trace.jsonl}
head_file=${HEAD_FILE:-./results/${dataset}/${model_name}_base_original_qa_n3000/surrogate_hh_scores/ranked_heads_${head_score_key}.json}
output_file=${OUTPUT_FILE:-${result_root}/head_hallucination_evidence_${head_score_key}.json}
python_bin=${PYTHON:-python3}

topk_values=${TOPK_VALUES:-"20 50 100"}
thresholds=${THRESHOLDS:-"0.4 0.5 0.65 0.9"}

"${python_bin}" -m eval_scripts.analyze_head_hallucination_evidence \
  --trace-file "${trace_file}" \
  --head-file "${head_file}" \
  --head-score-key "${head_score_key}" \
  --output-file "${output_file}" \
  --topk ${topk_values} \
  --thresholds ${thresholds}

echo "hallucination-head behavioral evidence saved -> ${output_file}"
