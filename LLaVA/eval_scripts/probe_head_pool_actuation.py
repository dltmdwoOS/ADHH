#!/usr/bin/env python3
"""Head-pool control study for text-side actuator behavior.

For each object occurrence probe, this script computes the target object's
log-probability under base decoding and under hard text-side suppression for
several head pools. The main metric is logp_base - logp_probe; larger positive
values mean suppressing that pool made the object token less likely.
"""

import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import random
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def as_head_record(item, fallback_score=1.0):
    if isinstance(item, dict):
        return {
            "layer": int(item["layer"]),
            "head": int(item["head"]),
            "score": float(item.get("score", fallback_score)),
            **{k: v for k, v in item.items() if k not in ("layer", "head", "score")},
        }
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return {"layer": int(item[0]), "head": int(item[1]), "score": float(fallback_score)}
    raise ValueError(f"Unsupported head record: {item}")


def load_head_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if isinstance(data.get("heads"), list):
            records = data["heads"]
        elif isinstance(data.get("hal_heads"), list):
            records = data["hal_heads"]
        else:
            raise ValueError(f"Unsupported head file: {path}")
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported head file: {path}")

    n = len(records)
    out = []
    for i, item in enumerate(records):
        fallback = 1.0 if n <= 1 else (n - 1 - i) / (n - 1)
        out.append(as_head_record(item, fallback))
    return out


def percentile_scores(records, key):
    sorted_records = sorted(records, key=lambda r: float(r.get(key, 0.0)), reverse=True)
    n = len(sorted_records)
    if n <= 1:
        return {f"{int(r['layer'])}-{int(r['head'])}": 1.0 for r in sorted_records}
    return {
        f"{int(r['layer'])}-{int(r['head'])}": (n - 1 - i) / (n - 1)
        for i, r in enumerate(sorted_records)
    }


def select_top(records, topk, key="score"):
    ordered = sorted(records, key=lambda r: float(r.get(key, 0.0)), reverse=True)
    return ordered[:topk]


def layer_matched_random(records, reference, seed):
    rng = random.Random(seed)
    by_layer = defaultdict(list)
    ref_counts = defaultdict(int)
    ref_keys = {f"{int(r['layer'])}-{int(r['head'])}" for r in reference}
    for r in records:
        by_layer[int(r["layer"])].append(r)
    for r in reference:
        ref_counts[int(r["layer"])] += 1
    selected = []
    for layer, count in sorted(ref_counts.items()):
        pool = [r for r in by_layer[layer] if f"{int(r['layer'])}-{int(r['head'])}" not in ref_keys]
        if len(pool) < count:
            pool = by_layer[layer]
        selected.extend(rng.sample(pool, min(count, len(pool))))
    return selected


def filter_records_by_layers(records, layers):
    if not layers:
        return list(records)
    allowed = {int(x) for x in layers}
    return [r for r in records if int(r["layer"]) in allowed]


def parse_layer_spec(spec):
    if spec is None or str(spec).strip() == "":
        return []
    text = str(spec).strip()
    if ":" in text:
        a, b = [int(x.strip()) for x in text.split(":", 1)]
        step = 1 if b >= a else -1
        return list(range(a, b + step, step))
    return [int(x.strip()) for x in text.replace(";", ",").split(",") if x.strip()]


def build_pools(
    records,
    topk,
    pool_names,
    random_seed,
    reproduced_adhh_records=None,
    adhh_alt_records=None,
    restrict_layers=None,
    adhh_topk=0,
):
    pools = {}
    proposed = select_top(records, topk, "score")
    reproduced_topk = adhh_topk if adhh_topk and adhh_topk > 0 else topk
    for name in pool_names:
        if name in ("proposed", "rank_fused"):
            pools[name] = proposed
        elif name == "text_only":
            pools[name] = select_top(records, topk, "front_raw")
        elif name == "contrast_only":
            pools[name] = select_top(records, topk, "back_raw")
        elif name == "signed_contrast_only":
            pools[name] = select_top(records, topk, "signed_back_raw")
        elif name == "layer_matched_random":
            pools[name] = layer_matched_random(records, proposed, random_seed)
        elif name == "score_shuffled":
            shuffled = deepcopy(proposed)
            rng = random.Random(random_seed)
            rng.shuffle(shuffled)
            pools[name] = shuffled
        elif name in ("reproduced_adhh", "adhh_reproduced", "adhh"):
            if reproduced_adhh_records is None:
                raise ValueError("Pool adhh/reproduced_adhh requires --reproduced-adhh-file")
            pools[name] = reproduced_adhh_records[:reproduced_topk]
        elif name in ("adhh_alt", "alternative_adhh"):
            if adhh_alt_records is None:
                raise ValueError("Pool adhh_alt requires --adhh-alt-file")
            pools[name] = adhh_alt_records[:topk]
        elif name in ("adhh_alt_restricted", "alternative_adhh_restricted"):
            if adhh_alt_records is None:
                raise ValueError("Pool adhh_alt_restricted requires --adhh-alt-file")
            restricted = filter_records_by_layers(adhh_alt_records, restrict_layers)
            pools[name] = restricted[:topk]
        else:
            raise ValueError(f"Unknown pool: {name}")
    return pools


def score_map_for_pool(pool):
    n = len(pool)
    out = {}
    for i, r in enumerate(pool):
        key = f"{int(r['layer'])}-{int(r['head'])}"
        out[key] = 1.0 if n <= 1 else (n - 1 - i) / (n - 1)
    return out


def make_prompt(question, model_config, conv_mode):
    qs = question
    if model_config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def concat_ids(prompt_ids, suffix_ids, device):
    if len(suffix_ids) == 0:
        return prompt_ids.to(device)
    suffix = torch.tensor(suffix_ids, dtype=prompt_ids.dtype, device=device).unsqueeze(0)
    return torch.cat([prompt_ids.to(device), suffix], dim=1)


def token_is_blank_piece(tokenizer, token_id):
    piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return piece.strip() == ""


def get_continuation_ids(tokenizer, text, add_leading_space=True, strip_leading_blank=False):
    if not text:
        return []
    value = (" " + text.strip()) if add_leading_space else text.strip()
    ids = tokenizer(value, add_special_tokens=False).input_ids
    if len(ids) == 0 and add_leading_space:
        ids = tokenizer(text.strip(), add_special_tokens=False).input_ids
    ids = [int(x) for x in ids]
    if strip_leading_blank:
        while ids and token_is_blank_piece(tokenizer, ids[0]):
            ids = ids[1:]
    return ids


def configure_intervention(model, heads=None, scores=None, img_start=35, img_length=576, enabled=False, text_scale=0.0, renorm=True, probe_thresholding=False, probe_threshold=0.4):
    if not enabled:
        model.config.intervention = "none"
        return
    model.config.intervention = "probe_hard"
    model.config.intervention_heads = [[int(r["layer"]), int(r["head"])] for r in heads]
    model.config.intervention_scores = scores or {f"{int(r['layer'])}-{int(r['head'])}": 1.0 for r in heads}
    model.config.hal_attention_heads = model.config.intervention_heads
    model.config.img_start_pos = int(img_start)
    model.config.img_length = int(img_length)
    model.config.probe_text_scale = float(text_scale)
    model.config.probe_thresholding = bool(probe_thresholding)
    model.config.probe_threshold = float(probe_threshold)
    model.config.text_threshold = float(probe_threshold)
    model.config.dynamic_renorm = bool(renorm)
    model.config.probe_renorm = bool(renorm)
    model.config.log_intervention_stats = False


def sequence_logprob(model, tokenizer, image_tensor, image_size, prompt_ids, prefix_text, target_text, aggregate_target_tokens):
    device = prompt_ids.device
    prefix_ids = get_continuation_ids(tokenizer, prefix_text, add_leading_space=True)
    target_ids = get_continuation_ids(tokenizer, target_text, add_leading_space=True, strip_leading_blank=True)
    if not target_ids:
        return None
    if not aggregate_target_tokens:
        target_ids = target_ids[:1]

    logprob = 0.0
    top1_first = None
    context_ids = prefix_ids[:]
    for step, target_id in enumerate(target_ids):
        input_ids = concat_ids(prompt_ids, context_ids, device)
        outputs = model(
            input_ids=input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        lp = F.log_softmax(logits.float(), dim=-1)[0, target_id]
        logprob += float(lp.item())
        if step == 0:
            top1_first = int(torch.argmax(logits, dim=-1).item())
        context_ids.append(target_id)
    return {
        "logprob": logprob,
        "target_token_ids": target_ids,
        "top1_first": top1_first,
        "top1_first_text": tokenizer.decode([top1_first]) if top1_first is not None else None,
    }


def summarize(records):
    by_pool_bucket = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_pool_bucket[r["pool"]][r["bucket"]].append(r)

    summary = {"by_pool": {}}
    for pool, buckets in by_pool_bucket.items():
        pool_out = {}
        for bucket, rows in buckets.items():
            drops = [float(x["logprob_drop"]) for x in rows]
            flips = [1.0 if x["top1_changed"] else 0.0 for x in rows]
            pool_out[bucket] = {
                "count": len(rows),
                "mean_logprob_drop": sum(drops) / max(len(drops), 1),
                "median_logprob_drop": sorted(drops)[len(drops)//2] if drops else None,
                "flip_rate": sum(flips) / max(len(flips), 1),
            }
        h = pool_out.get("hallucinated", {})
        g = pool_out.get("grounded", {})
        if h and g:
            pool_out["selectivity"] = {
                "logprob_gap_hall_minus_grounded": h["mean_logprob_drop"] - g["mean_logprob_drop"],
                "flip_gap_hall_minus_grounded": h["flip_rate"] - g["flip_rate"],
            }
        summary["by_pool"][pool] = pool_out
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", default=None)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--probe-file", required=True)
    parser.add_argument("--head-file", required=True)
    parser.add_argument("--reproduced-adhh-file", default=None)
    parser.add_argument("--adhh-alt-file", default=None)
    parser.add_argument("--restrict-layers", default="", help="Layer list/range used by adhh_alt_restricted, e.g. '9:16' or '9,10,11'.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conv-mode", default="vicuna_v1")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--adhh-topk", type=int, default=0, help="Optional top-k for the reproduced AD-HH pool only. Defaults to --topk.")
    parser.add_argument("--pools", default="proposed,text_only,contrast_only,adhh,adhh_alt,adhh_alt_restricted")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--img-start-pos", type=int, default=35)
    parser.add_argument("--img-length", type=int, default=576)
    parser.add_argument("--probe-text-scale", type=float, default=0.0)
    parser.add_argument("--probe-threshold", type=float, default=0.4, help="Text-mass threshold for paper-style hard probe triggering.")
    parser.add_argument("--no-probe-thresholding", dest="probe_thresholding", action="store_false", help="Disable thresholding and suppress every selected head unconditionally.")
    parser.set_defaults(probe_thresholding=True)
    parser.add_argument("--renorm", action="store_true", help="Renormalize rows after probe suppression. Default is no-renorm, matching the paper hard probe.")
    parser.add_argument("--aggregate-target-tokens", action="store_true", help="Sum log-prob over all non-blank target subword tokens instead of only the first object subword.")
    parser.add_argument("--max-probes", type=int, default=0)
    args = parser.parse_args()

    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)
    model.eval()

    probes = load_jsonl(args.probe_file)
    if args.max_probes and args.max_probes > 0:
        probes = probes[:args.max_probes]
    head_records = load_head_records(args.head_file)
    reproduced_adhh_records = load_head_records(args.reproduced_adhh_file) if args.reproduced_adhh_file else None
    adhh_alt_records = load_head_records(args.adhh_alt_file) if args.adhh_alt_file else None
    restrict_layers = parse_layer_spec(args.restrict_layers)
    pool_names = [x.strip() for x in args.pools.split(",") if x.strip()]
    pools = build_pools(
        head_records, args.topk, pool_names, args.random_seed,
        reproduced_adhh_records=reproduced_adhh_records,
        adhh_alt_records=adhh_alt_records,
        restrict_layers=restrict_layers,
        adhh_topk=args.adhh_topk,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    records_file = os.path.join(args.output_dir, "head_pool_probe_records.jsonl")
    summary_file = os.path.join(args.output_dir, "head_pool_probe_summary.json")
    pool_file = os.path.join(args.output_dir, "head_pool_probe_pools.json")

    with open(pool_file, "w", encoding="utf-8") as f:
        json.dump({
            name: [[int(r["layer"]), int(r["head"])] for r in rows]
            for name, rows in pools.items()
        }, f, indent=2)

    out_records = []
    with open(records_file, "w", encoding="utf-8") as writer:
        for probe in tqdm(probes, desc="probes"):
            prompt = make_prompt(probe.get("question", "Please describe this image in detail."), model.config, args.conv_mode)
            prompt_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
            image = Image.open(os.path.join(args.image_folder, probe["image"])).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(dtype=torch.float16, device="cuda")

            with torch.inference_mode():
                configure_intervention(model, enabled=False)
                base = sequence_logprob(
                    model, tokenizer, image_tensor, image.size, prompt_ids,
                    probe.get("prefix_text", ""), probe.get("target_text", ""),
                    args.aggregate_target_tokens,
                )
            if base is None:
                continue

            for pool_name, pool_heads in pools.items():
                with torch.inference_mode():
                    configure_intervention(
                        model,
                        heads=pool_heads,
                        scores=score_map_for_pool(pool_heads),
                        img_start=args.img_start_pos,
                        img_length=args.img_length,
                        enabled=True,
                        text_scale=args.probe_text_scale,
                        renorm=args.renorm,
                        probe_thresholding=args.probe_thresholding,
                        probe_threshold=args.probe_threshold,
                    )
                    probed = sequence_logprob(
                        model, tokenizer, image_tensor, image.size, prompt_ids,
                        probe.get("prefix_text", ""), probe.get("target_text", ""),
                        args.aggregate_target_tokens,
                    )
                if probed is None:
                    continue
                row = dict(probe)
                row.update({
                    "pool": pool_name,
                    "topk": args.topk,
                    "base_logprob": base["logprob"],
                    "probe_logprob": probed["logprob"],
                    "logprob_drop": base["logprob"] - probed["logprob"],
                    "base_top1_first": base["top1_first"],
                    "probe_top1_first": probed["top1_first"],
                    "base_top1_first_text": base["top1_first_text"],
                    "probe_top1_first_text": probed["top1_first_text"],
                    "top1_changed": base["top1_first"] != probed["top1_first"],
                    "target_token_ids": base["target_token_ids"],
                    "target_token_text": tokenizer.decode(base["target_token_ids"]),
                    "probe_text_scale": args.probe_text_scale,
                    "probe_thresholding": bool(args.probe_thresholding),
                    "probe_threshold": args.probe_threshold,
                    "renorm": bool(args.renorm),
                })
                out_records.append(row)
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                writer.flush()

    summary = summarize(out_records)
    summary["config"] = vars(args)
    summary["num_probe_records"] = len(probes)
    summary["num_output_rows"] = len(out_records)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
