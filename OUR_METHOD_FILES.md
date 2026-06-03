# Dynamic Head Suppression Files

This repository is the original AD-HH codebase plus the files needed to run and inspect our current method. The current method is exposed consistently as `dynamic`.

## Public Method Name

- Use `--intervention dynamic` in runnable scripts.
- Dynamic gate hyperparameters use the `dynamic_*` prefix (`--dynamic-strength`, `--dynamic-tau`, `--dynamic-exp-sharpness`, `--dynamic-redistribute`).
- In the model code, the old sigmoid-based dynamic branch has been removed; `dynamic` executes the latest exponential suppression rule.

## LLaVA 1.5 7B

Core files under `ADHH/LLaVA/`:

- `llava/model/language_model/modeling_llama.py`: attention intervention hook. Public `dynamic` mode uses the current text-side suppression algorithm and supports redistribution to system or vision tokens.
- `eval_scripts/eval_caption_dynamic.py`: COCO/CHAIR generation with dynamic intervention, resume support, and intervention statistics.
- `eval_scripts/eval_caption_adhh.py`, `eval_scripts/eval_caption.py`: AD-HH and base captioning baselines.
- `eval_scripts/compute_surrogate_score_zoo.py`: builds head ranking surrogates from text-attention summaries.
- `eval_scripts/build_layer_surrogate_combos.py`: builds rank-percentile combo head pools.
- `eval_scripts/filter_txtattn_summary.py`: filters all-head summaries to an explicit layer list.
- `eval_scripts/estimate_dynamic_tau.py`: estimates the high text-reliance threshold used by the dynamic gate.
- `eval_scripts/summarize_txtattn_trace.py`: converts per-step traces into bucketed attention summaries.

Main scripts:

- `bash_scripts/chair_dynamic.sh`: run COCO/CHAIR dynamic experiments.
- `bash_scripts/run_layer_list_dynamic_pipeline.sh`: build layer-list head pools and run dynamic experiments. It accepts explicit layer specs such as `9,10,11,12,13,15,16`.
- `bash_scripts/run_dynamic_redistribute_ablation.sh`: compare normal renormalization with system/vision redistribution.
- `bash_scripts/amber_dynamic.sh`, `bash_scripts/mmvet_dynamic.sh`: generative AMBER and MM-Vet entrypoints.
- Base/AD-HH baselines are kept as `chair_base.sh`, `chair_adhh.sh`, `amber_base.sh`, `amber_adhh.sh`, `mmvet_base.sh`, and `mmvet_adhh.sh`.

Compact summaries are under `LLaVA/results_summary/`. Full captions, decode logs, and large per-sample result files are intentionally excluded.

## LLaVA-NeXT Port

`ADHH/LLaVA_NeXT/` contains the minimal files needed to run the same pipeline on the LLaVA-NeXT/LLaMA attention stack:

- `llava_next/model/language_model/modeling_llama.py`
- `eval_scripts/eval_caption_dynamic.py`
- `eval_scripts/eval_caption.py`
- the same head-pool utilities as the LLaVA 1.5 port
- `bash_scripts/chair_dynamic.sh`
- `bash_scripts/run_layer_list_dynamic_pipeline.sh`

Compact metrics and the selected head pool are under `LLaVA_NeXT/results_summary/`.

## Qwen2.5-VL Port

`ADHH/Qwen3_VL/` contains the Qwen2.5-VL adaptation. The upstream directory name is `Qwen3_VL` because the cloned Qwen repository redirects there, but the experiment target is Qwen2.5-VL.

Key files:

- `qwen_dynamic/attention_patch.py`: monkey-patches Qwen2.5-VL attention so the same source-region suppression can be applied.
- `eval_scripts/eval_caption_dynamic.py`: COCO/CHAIR dynamic evaluation for Qwen2.5-VL.
- `eval_scripts/eval_caption.py`: base captioning and text-attention trace collection.
- `tools/check_qwen25_span.py`: quick utility to verify system/vision/text-side token spans.
- `bash_scripts/chair_dynamic.sh`
- `bash_scripts/run_layer_list_dynamic_pipeline.sh`

Compact metrics and the selected head pool are under `Qwen3_VL/results_summary/`.

## VILA Port

`ADHH/VILA/` contains the minimal overlay files needed to reproduce the same dynamic head-suppression workflow on VILA, without vendoring the full upstream VILA repository.

Key files:

- `eval_scripts/eval_caption.py`: base captioning plus text-attention trace collection for VILA prompts/spans.
- `eval_scripts/eval_caption_base.py`: base COCO/CHAIR decoding entrypoint.
- `eval_scripts/eval_caption_dynamic.py`: dynamic intervention evaluation. For VILA, the attention intervention is applied by monkey-patching HuggingFace `LlamaAttention.forward` at runtime rather than by editing VILA's model class directly.
- `eval_scripts/compute_surrogate_score_zoo.py`, `build_layer_surrogate_combos.py`, `filter_txtattn_summary.py`, `summarize_txtattn_trace.py`, and `estimate_dynamic_tau.py`: the same head-pool construction pipeline used by the other model ports.
- `bash_scripts/decoding_base_with_original_qa.sh`: collects all-head text-attention traces on COCO original-QA samples.
- `bash_scripts/run_layer_list_dynamic_pipeline.sh`: filters traces to explicit layer lists, builds surrogate head pools, estimates tau, and launches dynamic runs.
- `bash_scripts/chair_base.sh`: VILA base COCO/CHAIR baseline.
- `llava/model/language_model/llava_llama.py`: included as a reference copy of the VILA LLaMA wrapper used when the port was tested. The active dynamic hook is still in `eval_caption_dynamic.py`.

The VILA result directories are not copied wholesale. Compact artifacts such as `captions_eval_results.json`, `run_config.json`, `intervention_stats.json`, `dynamic_tau_estimate.json`, and ranked-head JSON files can be copied selectively once final runs are chosen.

## Head Pool Used By The Current Method

The main score is `global__itext_all__C_toi_HminusG`:

- `itext_all`: text-side attention mass after the image region, including the question/instruction side plus generated text prefix.
- `C_toi_HminusG`: contrastive text-over-image ratio, hallucinated-object steps minus grounded-object steps.
- The two rankings are combined at the rank-percentile level, then the top heads are used as the dynamic intervention pool.

Representative head-pool files:

- `LLaVA/results_summary/coco/ranked_heads_global__itext_all__C_toi_HminusG.json`
- `LLaVA_NeXT/results_summary/coco/ranked_heads_global__itext_all__C_toi_HminusG.json`
- `Qwen3_VL/results_summary/coco/ranked_heads_global__itext_all__C_toi_HminusG.json`

## Result Policy

Only compact summaries are tracked here:

- `*_metrics.json`: `overall_metrics` only.
- `*_run_config.json`: small reproducibility metadata.
- `*_intervention_stats.json`: aggregate intervention statistics.
- selected ranked-head JSON files.

Excluded on purpose: `captions.jsonl`, decode logs, full per-sample `captions_eval_results.json`, raw text-attention traces, and large benchmark payloads.
