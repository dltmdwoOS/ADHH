#!/usr/bin/env python3
"""Build a layer-matched random control head pool.

The proposed DEACT pool is ranked by a surrogate score. This script samples a
random control pool with the same per-layer head counts as the proposed top-k
pool. For online DEACT, the suppression strength also depends on head scores, so
the output assigns the proposed top-k score profile to the random heads by rank.
This isolates head identity from score-scale confounds.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_head_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("heads"), list):
        records = data["heads"]
    elif isinstance(data, dict) and isinstance(data.get("hal_heads"), list):
        records = data["hal_heads"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported head file format: {path}")

    out = []
    n = len(records)
    for idx, item in enumerate(records):
        fallback_score = 1.0 if n <= 1 else (n - 1 - idx) / (n - 1)
        if isinstance(item, dict):
            rec = dict(item)
            rec["layer"] = int(rec["layer"])
            rec["head"] = int(rec["head"])
            rec["score"] = float(rec.get("score", fallback_score))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            rec = {"layer": int(item[0]), "head": int(item[1]), "score": float(fallback_score)}
        else:
            raise ValueError(f"Unsupported head record: {item}")
        rec.setdefault("source_rank", idx + 1)
        out.append(rec)
    return out


def head_key(record):
    return f"{int(record['layer'])}-{int(record['head'])}"


def build_layer_matched_random(records, topk, seed, score_key):
    reference = records[:topk]
    rng = random.Random(seed)
    by_layer = defaultdict(list)
    for rec in records:
        by_layer[int(rec["layer"])].append(rec)

    ref_keys = {head_key(rec) for rec in reference}
    available_by_layer = {}
    fallback_by_layer = {}
    for layer, rows in by_layer.items():
        available = [rec for rec in rows if head_key(rec) not in ref_keys]
        rng.shuffle(available)
        fallback = list(rows)
        rng.shuffle(fallback)
        available_by_layer[layer] = available
        fallback_by_layer[layer] = fallback

    selected = []
    selected_keys = set()
    warnings = []

    for rank, ref in enumerate(reference, start=1):
        layer = int(ref["layer"])
        if available_by_layer[layer]:
            rec = available_by_layer[layer].pop()
        else:
            pool = [rec for rec in fallback_by_layer[layer] if head_key(rec) not in selected_keys]
            if not pool:
                pool = fallback_by_layer[layer]
            warnings.append(
                f"Layer {layer}: exhausted non-reference heads at proposed rank {rank}; "
                f"sampling from the full layer pool, so overlap with the proposed pool may occur."
            )
            rec = rng.choice(pool)
        selected.append(rec)
        selected_keys.add(head_key(rec))

    # Preserve the proposed score prior profile by rank, so the online
    # suppression magnitude is not weakened merely because the sampled heads
    # have lower surrogate scores.
    score_profile = [float(rec.get(score_key, rec.get("score", 1.0))) for rec in reference]
    out_heads = []
    selected_keys = set()
    for idx, rec in enumerate(selected):
        assigned_score = score_profile[idx] if idx < len(score_profile) else 1.0
        selected_keys.add(head_key(rec))
        out_heads.append({
            "layer": int(rec["layer"]),
            "head": int(rec["head"]),
            "head_id": f"L{int(rec['layer'])}H{int(rec['head'])}",
            "score": float(assigned_score),
            score_key: float(assigned_score),
            "assigned_score_from_proposed_rank": idx + 1,
            "original_score": float(rec.get(score_key, rec.get("score", 1.0))),
            "original_global_rank": rec.get("global_rank", rec.get("source_rank")),
            "control_pool": "layer_matched_random",
        })

    # Keep the output list length equal to the source ranked file. With
    # rank-percentile score normalization, eval_caption_dynamic computes scores
    # over the whole file, not only over top-k. Appending non-selected filler
    # records preserves the proposed run's percentile profile for the active
    # top-k random heads.
    for rec in records:
        if head_key(rec) in selected_keys:
            continue
        filler = dict(rec)
        filler["layer"] = int(filler["layer"])
        filler["head"] = int(filler["head"])
        filler.setdefault("head_id", f"L{int(filler['layer'])}H{int(filler['head'])}")
        filler["control_pool"] = "filler_not_selected"
        out_heads.append(filler)

    return reference, selected, out_heads, warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-file", required=True, help="Ranked proposed head JSON.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-key", default="score")
    args = parser.parse_args()

    records = load_head_records(args.head_file)
    if args.topk <= 0 or args.topk > len(records):
        raise ValueError(f"--topk must be in [1, {len(records)}], got {args.topk}")

    reference, selected_active, output_heads, warnings = build_layer_matched_random(
        records, args.topk, args.seed, args.score_key
    )
    ref_counts = Counter(int(rec["layer"]) for rec in reference)
    rand_counts = Counter(int(rec["layer"]) for rec in selected_active)
    overlap = sorted(set(head_key(rec) for rec in reference) & set(head_key(rec) for rec in selected_active))

    output = {
        "score_name": "layer_matched_random",
        "description": (
            "Layer-matched random control pool. Per-layer counts match the proposed top-k pool; "
            "the proposed score profile is copied by rank to control for suppression-prior strength."
        ),
        "source_head_file": args.head_file,
        "topk": int(args.topk),
        "num_total_records": len(output_heads),
        "num_active_random_heads": len(selected_active),
        "seed": int(args.seed),
        "score_key": args.score_key,
        "reference_layer_counts": {str(k): int(v) for k, v in sorted(ref_counts.items())},
        "random_layer_counts": {str(k): int(v) for k, v in sorted(rand_counts.items())},
        "overlap_with_reference": [[int(x.split("-")[0]), int(x.split("-")[1])] for x in overlap],
        "warnings": warnings,
        "heads": output_heads,
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(
        f"saved layer-matched random heads -> {out_path} "
        f"({len(selected_active)} active / {len(output_heads)} total heads, overlap={len(overlap)}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
