# LLaVA-1.5-7B Latency and Memory Aggregate

Source: COCO, n=500, warmup=8, seed42_n500_tok128. Greedy is aggregated from the earlier separate run; ADHH/DEACT are from the corrected direct-DEACT run.

| Method | CHAIRs (%) | CHAIRi (%) | Object F1 (%) | sec/image | runtime x | tokens/s | peak alloc delta (GB) | peak reserved delta (GB) | memory x | mean gen tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 53.4 | 14.3 | 76.5 | 3.782 | 1.00 | 30.34 | 0.746 | 0.822 | 1.00 | 114.8 |
| adhh | 39.6 | 10.0 | 76.6 | 3.874 | 1.02 | 30.86 | 0.746 | 0.822 | 1.00 | 119.5 |
| deact | 31.6 | 7.9 | 77.4 | 5.020 | 1.33 | 20.99 | 0.746 | 0.822 | 1.00 | 105.4 |

## Relative to Greedy

| Method | CHAIRs delta (pp) | CHAIRs rel. reduction (%) | CHAIRi delta (pp) | CHAIRi rel. reduction (%) | sec/image overhead (%) | token throughput change (%) |
|---|---:|---:|---:|---:|---:|---:|
| adhh | -13.8 | 25.8 | -4.2 | 29.7 | 2.4 | 1.7 |
| deact | -21.8 | 40.8 | -6.4 | 44.8 | 32.7 | -30.8 |

## Provenance

- greedy: `results_deact/coco/llava-v1.5-7b/runtime/seed42_n500_tok128/sysvis/greedy`; config `{"intervention": "none", "method": "greedy"}`
- adhh: `results_deact/coco/llava-v1.5-7b/runtime/seed42_n500_tok128/direct/adhh`; config `{"head_file": "/workspace/ADHH/LLaVA/results_deact/coco/llava-v1.5-7b/baselines/adhh_reproduced/attribution_result.json", "head_source": "file", "intervention": "adhh", "method": "adhh", "selected_head_count": 20, "text_threshold": 0.4, "topk": 20}`
- deact: `results_deact/coco/llava-v1.5-7b/runtime/seed42_n500_tok128/direct/deact`; config `{"dynamic_exp_sharpness": 10.0, "dynamic_late_tau": 0.8, "dynamic_redistribute": "none", "dynamic_renorm": false, "dynamic_tau": 0.9, "head_file": "/workspace/ADHH/results_deact/coco/llava-v1.5-7b/resources/l9_l16_train_n500/surrogate_score_zoo/ranked_heads_global__itext_all__C_toi_HminusG_signed.json", "intervention": "late_boost", "method": "deact", "selected_head_count": 100, "topk": 100}`
