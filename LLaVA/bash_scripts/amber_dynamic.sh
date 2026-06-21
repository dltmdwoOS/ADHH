#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
results_root=${RESULTS_ROOT:-$(adhh_default_results_root)}
amber_root=${AMBER_ROOT:-$(adhh_default_amber_root)}
image_folder=${AMBER_IMAGE_FOLDER:-$(adhh_default_amber_image_folder)}
seed=${SEED:-42}
gpu=${GPU:-0}
max_samples=${MAX_SAMPLES:-0}
max_new_tokens=${MAX_NEW_TOKENS:-512}
python_bin=${PYTHON_BIN:-python}
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  python_bin=python3
fi

layer_slug_name=${LAYER_SLUG:-l9_l16}
topk=${TOPK:-100}
dynamic_strength=${DYNAMIC_STRENGTH:-1.0}
dynamic_exp_sharpness=${DYNAMIC_EXP_SHARPNESS:-10.0}
dynamic_score_power=${DYNAMIC_SCORE_POWER:-1.0}
dynamic_tau=${DYNAMIC_TAU:-0.90}
dynamic_late_tau=${DYNAMIC_LATE_TAU:-0.80}
dynamic_redistribute=${DYNAMIC_REDISTRIBUTE:-none}
dynamic_renorm=${DYNAMIC_RENORM:-false}
dynamic_context_mode=${DYNAMIC_CONTEXT_MODE:-ratio_exp}
head_score_key=${HEAD_SCORE_KEY:-global__itext_all__C_toi_HminusG_signed}
head_score_normalize=${HEAD_SCORE_NORMALIZE:-rank_percentile}
head_source=${HEAD_SOURCE:-file}
head_file=${HEAD_FILE:-${results_root}/analysis/selected_heads/${model_name}/ranked_heads_${head_score_key}.json}

slug_float() {
  "${python_bin}" - "$1" <<'PY'
import sys
x = float(sys.argv[1])
if 0 < x < 1:
    s = f"{x:.4f}".rstrip("0")
    decimals = s.split(".", 1)[1] if "." in s else ""
    if len(decimals) < 2:
        s = f"{x:.2f}"
    print(s.replace(".", ""))
else:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    print(s.replace(".", "") or "0")
PY
}

update_slug() {
  local redir=$1
  local renorm_flag=$2
  if [[ "${redir}" == "none" && "${renorm_flag}" != "true" ]]; then
    echo "direct"
  elif [[ "${redir}" == "none" && "${renorm_flag}" == "true" ]]; then
    echo "renorm"
  elif [[ "${redir}" == "renorm" ]]; then
    echo "renorm"
  else
    echo "redir_${redir}"
  fi
}

q_slug=$(slug_float "${dynamic_exp_sharpness}")
hi_slug=$(slug_float "${dynamic_tau}")
lo_slug=$(slug_float "${dynamic_late_tau}")
update_name=$(update_slug "${dynamic_redistribute}" "${dynamic_renorm}")

result_path=${RESULT_PATH:-${results_root}/amber/${model_name}/main/${layer_slug_name}/k${topk}/${update_name}/tok${max_new_tokens}/q${q_slug}_tau${hi_slug}-${lo_slug}}
if [[ "${max_samples}" != "0" ]]; then
  result_path=${result_path}/n${max_samples}
fi
mkdir -p "${result_path}"

extra_head_args=()
if [[ "${head_source}" == "file" ]]; then
  extra_head_args+=(
    --head-file "${head_file}"
    --head-score-key "${head_score_key}"
    --head-score-normalize "${head_score_normalize}"
  )
fi

score_args=()
if [[ "${USE_HEAD_SCORES:-true}" == "true" ]]; then
  score_args+=(--use-head-scores)
fi

stats_args=()
if [[ "${LOG_INTERVENTION_STATS:-false}" == "true" ]]; then
  stats_args+=(--log-intervention-stats)
fi

renorm_args=()
if [[ "${dynamic_renorm}" != "true" ]]; then
  renorm_args+=(--no-dynamic-renorm)
fi

resume_args=()
if [[ "${RESUME:-true}" == "true" ]]; then
  resume_args+=(--resume)
fi

official_args=()
if [[ "${RUN_OFFICIAL_EVAL:-true}" == "true" ]]; then
  official_args+=(--run-official-eval)
fi
dry_run=${DRY_RUN:-false}

export PYTHONUNBUFFERED=1

echo "[GPU ${gpu}] AMBER dynamic start: layers=${layer_slug_name}, topk=${topk}, q=${dynamic_exp_sharpness}, tau=${dynamic_tau}-${dynamic_late_tau}, redistribute=${dynamic_redistribute}, renorm=${dynamic_renorm}"

if [[ "${dry_run}" == "true" ]]; then
  echo "[dry-run] AMBER dynamic would write -> ${result_path}"
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
  --intervention dynamic \
  --head-source "${head_source}" \
  "${extra_head_args[@]}" \
  --topk "${topk}" \
  --dynamic-strength "${dynamic_strength}" \
  --dynamic-context-mode "${dynamic_context_mode}" \
  --dynamic-tau "${dynamic_tau}" \
  --dynamic-exp-sharpness "${dynamic_exp_sharpness}" \
  --dynamic-late-tau "${dynamic_late_tau}" \
  --dynamic-score-power "${dynamic_score_power}" \
  --dynamic-redistribute "${dynamic_redistribute}" \
  "${renorm_args[@]}" \
  "${score_args[@]}" \
  "${stats_args[@]}" \
  "${official_args[@]}" \
  "${resume_args[@]}" \
  > "${result_path}/decode.log" 2>&1

echo "[GPU ${gpu}] AMBER dynamic done -> ${result_path}"
