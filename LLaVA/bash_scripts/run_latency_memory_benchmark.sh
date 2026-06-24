#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_paths.sh"

model_name=${MODEL_NAME:-llava-v1.5-7b}
model_path=${MODEL_PATH:-liuhaotian/llava-v1.5-7b}
dataset=${DATASET:-coco}
data_path=${DATA_PATH:-$(adhh_default_data_path)}
results_root=${RESULTS_ROOT:-$(adhh_default_results_root)}
seed=${SEED:-42}
num_samples=${NUM_SAMPLES:-500}
warmup_samples=${WARMUP_SAMPLES:-8}
max_new_tokens=${MAX_NEW_TOKENS:-128}
gpu=${GPU:-0}
methods=${METHODS:-adhh,deact}
pai_alpha=${PAI_ALPHA:-0.5}
pai_gamma=${PAI_GAMMA:-1.1}
pai_start_layer=${PAI_START_LAYER:-2}
pai_end_layer=${PAI_END_LAYER:-32}
vaf_enh_para=${VAF_ENH_PARA:-1.15}
vaf_sup_para=${VAF_SUP_PARA:-0.95}
vaf_start_layer=${VAF_START_LAYER:-9}
vaf_end_layer=${VAF_END_LAYER:-15}
tarac_alpha=${TARAC_ALPHA:-0.5}
tarac_beta=${TARAC_BETA:-0.5}
tarac_start_layer=${TARAC_START_LAYER:-9}
tarac_end_layer=${TARAC_END_LAYER:-16}
# Paper-main DEACT uses direct attenuation: remove text-side mass without
# redistributing it and without row renormalization.
deact_redistribute=${DEACT_REDISTRIBUTE:-none}
deact_renorm=${DEACT_RENORM:-false}
deact_topk=${DEACT_TOPK:-100}
deact_q=${DEACT_Q:-10.0}
deact_tau=${DEACT_TAU:-0.90}
deact_late_tau=${DEACT_LATE_TAU:-0.80}
auto_tau=${DEACT_AUTO_TAU:-true}
force_output_attentions=${FORCE_OUTPUT_ATTENTIONS:-false}
attn_implementation=${ATTN_IMPLEMENTATION:-eager}
python_bin=${PYTHON_BIN:-python3}

sample_dir=${results_root}/${dataset}/${model_name}/shared_samples
sample_id_file=${SAMPLE_ID_FILE:-${sample_dir}/val_seed${seed}_n${num_samples}.json}
if [[ "${deact_redistribute}" == "none" && "${deact_renorm}" != "true" ]]; then
  deact_update_slug=direct
elif [[ "${deact_redistribute}" == "none" && "${deact_renorm}" == "true" ]]; then
  deact_update_slug=renorm
elif [[ "${deact_redistribute}" == "renorm" ]]; then
  deact_update_slug=renorm
else
  deact_update_slug="${deact_redistribute}"
fi
out_dir=${OUTPUT_DIR:-${results_root}/${dataset}/${model_name}/runtime/seed${seed}_n${num_samples}_tok${max_new_tokens}/${deact_update_slug}}
deact_resource_dir=${results_root}/${dataset}/${model_name}/resources/l9_l16_train_n500
deact_head_file=${DEACT_HEAD_FILE:-${deact_resource_dir}/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG_signed.json}
deact_tau_file=${DEACT_TAU_FILE:-${deact_resource_dir}/dynamic_tau_estimate.json}
adhh_head_file=${ADHH_HEAD_FILE:-${results_root}/${dataset}/${model_name}/baselines/adhh_reproduced/attribution_result.json}
if [[ ! -f "${adhh_head_file}" && -f "${ADHH_MODEL_DIR}/results_deact/${dataset}/${model_name}/baselines/adhh_reproduced/attribution_result.json" ]]; then
  adhh_head_file="${ADHH_MODEL_DIR}/results_deact/${dataset}/${model_name}/baselines/adhh_reproduced/attribution_result.json"
fi

mkdir -p "${sample_dir}" "${out_dir}"

if [[ ! -f "${sample_id_file}" ]]; then
  "${python_bin}" - <<PY
import json, random
from pycocotools.coco import COCO
caption_file = "${data_path}/coco/annotations/captions_val2014.json"
random.seed(${seed})
coco = COCO(caption_file)
sampled = random.sample(coco.getImgIds(), ${num_samples})
with open("${sample_id_file}", "w") as f:
    json.dump(sampled, f, indent=2)
print("saved sample ids -> ${sample_id_file}")
PY
fi

extra_args=()
if [[ "${auto_tau}" == "true" ]]; then
  extra_args+=(--deact-auto-tau)
fi
if [[ "${force_output_attentions}" == "true" ]]; then
  extra_args+=(--force-output-attentions)
fi
if [[ "${deact_renorm}" == "true" ]]; then
  extra_args+=(--deact-renorm)
else
  extra_args+=(--no-deact-renorm)
fi

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -m eval_scripts.benchmark_latency_memory \
  --model-path "${model_path}" \
  --image-folder "${data_path}/coco/val2014" \
  --caption_file_path "${data_path}/coco/annotations/captions_val2014.json" \
  --output-dir "${out_dir}" \
  --sample-id-file "${sample_id_file}" \
  --methods "${methods}" \
  --conv-mode vicuna_v1 \
  --num_samples "${num_samples}" \
  --warmup-samples "${warmup_samples}" \
  --max_new_tokens "${max_new_tokens}" \
  --seed "${seed}" \
  --temperature 0 \
  --attn-implementation "${attn_implementation}" \
  --adhh-head-source file \
  --adhh-head-file "${adhh_head_file}" \
  --adhh-topk "${ADHH_TOPK:-20}" \
  --adhh-threshold "${ADHH_THRESHOLD:-0.4}" \
  --pai-alpha "${pai_alpha}" \
  --pai-gamma "${pai_gamma}" \
  --pai-start-layer "${pai_start_layer}" \
  --pai-end-layer "${pai_end_layer}" \
  --vaf-enh-para "${vaf_enh_para}" \
  --vaf-sup-para "${vaf_sup_para}" \
  --vaf-start-layer "${vaf_start_layer}" \
  --vaf-end-layer "${vaf_end_layer}" \
  --tarac-alpha "${tarac_alpha}" \
  --tarac-beta "${tarac_beta}" \
  --tarac-start-layer "${tarac_start_layer}" \
  --tarac-end-layer "${tarac_end_layer}" \
  --deact-head-file "${deact_head_file}" \
  --deact-tau-file "${deact_tau_file}" \
  --deact-topk "${deact_topk}" \
  --deact-exp-sharpness "${deact_q}" \
  --deact-tau "${deact_tau}" \
  --deact-late-tau "${deact_late_tau}" \
  --deact-late-boost-end "${max_new_tokens}" \
  --deact-redistribute "${deact_redistribute}" \
  --deact-use-head-scores \
  "${extra_args[@]}" \
  2>&1 | tee "${out_dir}/benchmark.log"

IFS=',' read -r -a method_list <<< "${methods}"
for method in "${method_list[@]}"; do
  method="$(echo "${method}" | xargs)"
  [[ -z "${method}" ]] && continue
  "${python_bin}" eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir "${data_path}/coco/annotations" \
    --answers-file "${out_dir}/${method}/captions.jsonl" \
    --caption_file captions_val2014.json \
    > "${out_dir}/${method}/chair.log" 2>&1
done

"${python_bin}" -m eval_scripts.summarize_latency_memory --output-dir "${out_dir}" \
  | tee "${out_dir}/paper_table_latency_memory.md"

echo "Latency/memory benchmark finished: ${out_dir}"
