#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
amber_root=${AMBER_ROOT:-$(adhh_default_amber_root)}
image_folder=${AMBER_IMAGE_FOLDER:-$(adhh_default_amber_image_folder)}
seed=${SEED:-42}
gpu=${GPU:-1}
max_samples=${MAX_SAMPLES:-0}
max_new_tokens=${MAX_NEW_TOKENS:-512}
results_root=${RESULTS_ROOT:-$(adhh_default_results_root)}
python_bin=${PYTHON_BIN:-python}
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin=python3
fi

# Paper-style AD-HH defaults: top-20 heads and tau=0.4.
adhh_topk=${ADHH_TOPK:-20}
adhh_threshold=${ADHH_THRESHOLD:-0.4}

# default uses the built-in AD-HH head list; file uses a saved attribution file.
head_source=${HEAD_SOURCE:-file}
default_head_file=${results_root}/coco/${model_name}/baselines/adhh_reproduced/attribution_result.json
if [[ ! -f "${default_head_file}" && -f "./results_deact/coco/${model_name}/baselines/adhh_reproduced/attribution_result.json" ]]; then
  default_head_file=./results_deact/coco/${model_name}/baselines/adhh_reproduced/attribution_result.json
fi
head_file=${HEAD_FILE:-${default_head_file}}

if [[ "${max_samples}" == "0" ]]; then
  sample_suffix=full
else
  sample_suffix=n${max_samples}
fi

baseline_name=adhh
if [[ "${head_source}" == "file" ]]; then
  baseline_name=adhh_reproduced
fi
result_path=${RESULT_PATH:-${results_root}/amber/${model_name}/baselines/${baseline_name}/tok${max_new_tokens}/tau${adhh_threshold}}
if [[ "${sample_suffix}" != "full" ]]; then
  result_path=${result_path}/${sample_suffix}
fi
mkdir -p "${result_path}"

extra_head_args=()
if [[ "${head_source}" == "file" ]]; then
  extra_head_args+=(--head-file "${head_file}")
fi

resume=${RESUME:-false}
resume_args=()
if [[ "${resume}" == "true" ]]; then
  resume_args+=(--resume)
fi
dry_run=${DRY_RUN:-false}

run_official_eval=${RUN_OFFICIAL_EVAL:-true}
official_args=()
if [[ "${run_official_eval}" == "true" ]]; then
  official_args+=(--run-official-eval)
fi

export PYTHONUNBUFFERED=1

echo "[GPU ${gpu}] AMBER AD-HH start: head_source=${head_source}, topk=${adhh_topk}, tau=${adhh_threshold}, samples=${sample_suffix}"

if [[ "${dry_run}" == "true" ]]; then
  echo "[dry-run] AMBER AD-HH would write -> ${result_path}"
  echo "[dry-run] amber_root=${amber_root}"
  echo "[dry-run] image_folder=${image_folder}"
  echo "[dry-run] head_file=${head_file}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.eval_amber \
  --model-path "${model_path}" \
  --image-folder "${image_folder}" \
  --amber-root "${amber_root}" \
  --query-file "${amber_root}/data/query/query_generative.json" \
  --answers-file "${result_path}/answers.jsonl" \
  --response-file "${result_path}/amber_responses.json" \
  --metrics-file "${result_path}/amber_metrics.json" \
  --temperature 0 \
  --conv-mode vicuna_v1 \
  --seed "${seed}" \
  --num-workers 4 \
  --max_new_tokens "${max_new_tokens}" \
  --max-samples "${max_samples}" \
  --intervention adhh \
  --head-source "${head_source}" \
  "${extra_head_args[@]}" \
  --topk "${adhh_topk}" \
  --text-threshold "${adhh_threshold}" \
  --log-intervention-stats \
  "${official_args[@]}" \
  "${resume_args[@]}" \
  > "${result_path}/decode.log" 2>&1

echo "[GPU ${gpu}] AMBER AD-HH done -> ${result_path}"
