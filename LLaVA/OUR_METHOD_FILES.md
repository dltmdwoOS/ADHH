# Files Added for the Dynamic Head Suppression Experiments

This repository started from the original AD-HH codebase. The files below
collect the code and compact result snapshots used for the current dynamic
suppression method.

## Core model code

- `llava/model/language_model/modeling_llama.py`
  - Contains AD-HH-style intervention hooks plus our `dynamic` suppression
    method. This cleaned version uses `dynamic` as the public method name.

## Main evaluation entrypoints

- `eval_scripts/eval_caption_dynamic.py`
  - COCO caption evaluation with dynamic suppression.
- `eval_scripts/eval_amber.py`
  - AMBER generative evaluation entrypoint.
- `eval_scripts/eval_mmvet.py`
  - MM-Vet generation/evaluation entrypoint.
- `eval_scripts/backfill_object_f1.py`
  - Adds object-level precision/recall/F1 to CHAIR outputs.
- `eval_scripts/backfill_caption_quality_metrics.py`
  - Adds METEOR/CIDEr/SPICE-style caption metrics when dependencies are
    available.

## Head scoring and analysis

- `eval_scripts/compute_surrogate_score_zoo.py`
- `eval_scripts/build_layer_surrogate_combos.py`
- `eval_scripts/analyze_head_proxy_evidence.py`
- `eval_scripts/analyze_head_hallucination_evidence.py`

The main current head pool is:

- `results/coco/llava-v1.5-7b_base_original_qa_n3000/surrogate_hh_scores/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG.json`

This score combines text-side attention mass and text-over-image contrastive
ranking after rank-percentile normalization.

## Bash scripts

- `bash_scripts/chair_dynamic.sh`
- `bash_scripts/amber_dynamic.sh`
- `bash_scripts/mmvet_dynamic.sh`
- `bash_scripts/chair_base.sh`, `chair_adhh.sh`
- `bash_scripts/amber_base.sh`, `amber_adhh.sh`
- `bash_scripts/mmvet_base.sh`, `mmvet_adhh.sh`

## Compact result snapshots

Representative COCO/CHAIR result snapshots were copied under:

- `results/coco/`
- `results_dynamic/coco/`

Representative AMBER and MM-Vet snapshots were copied under:

- `results_amber/generative/`
- `results_mmvet/`

Large decode logs were intentionally not copied.
