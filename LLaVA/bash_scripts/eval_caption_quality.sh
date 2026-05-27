#!/usr/bin/env bash
set -euo pipefail

data_path=${DATA_PATH:-../dataset}
answers_file=${ANSWERS_FILE:?Set ANSWERS_FILE to a captions.jsonl file}
caption_file=${CAPTION_FILE:-${data_path}/coco/annotations/captions_val2014.json}
output_file=${OUTPUT_FILE:-${answers_file%.jsonl}_quality_metrics.json}
python_bin=${PYTHON:-python3}

"${python_bin}" -m eval_scripts.eval_caption_quality \
  --answers-file "${answers_file}" \
  --annotation-file "${caption_file}" \
  --output-file "${output_file}" \
  --metrics CIDEr SPICE METEOR

echo "quality metrics saved -> ${output_file}"
