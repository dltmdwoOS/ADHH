#!/usr/bin/env python3
"""Export a saved probe head pool into the ranked-head format used by DEACT.

Historical head-pool control probes save pools as a compact mapping such as
{"layer_matched_random": [[9, 17], ...]}. DEACT's caption evaluator expects a
ranked JSON with a "heads" list and, for rank-percentile scoring, benefits from
having the full candidate list length preserved. This script places the saved
probe pool first and appends non-selected filler heads from a reference ranked
file.
"""

import argparse
import json
from pathlib import Path


def load_pool(pool_file, pool_name):
    with open(pool_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if pool_name not in data:
            raise ValueError(f"Pool '{pool_name}' not found in {pool_file}; keys={list(data.keys())}")
        rows = data[pool_name]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported pool file format: {pool_file}")

    pool = []
    for item in rows:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError(f"Unsupported pool item: {item}")
        pool.append({"layer": int(item[0]), "head": int(item[1])})
    return pool


def load_ranked_records(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("heads"), list):
        rows = data["heads"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"Unsupported reference ranked-head format: {path}")

    records = []
    n = len(rows)
    for idx, item in enumerate(rows):
        fallback_score = 1.0 if n <= 1 else (n - 1 - idx) / (n - 1)
        if isinstance(item, dict):
            rec = dict(item)
            rec["layer"] = int(rec["layer"])
            rec["head"] = int(rec["head"])
            rec.setdefault("score", float(rec.get("score", fallback_score)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            rec = {"layer": int(item[0]), "head": int(item[1]), "score": float(fallback_score)}
        else:
            raise ValueError(f"Unsupported reference item: {item}")
        rec.setdefault("source_rank", idx + 1)
        records.append(rec)
    return records


def head_key(record):
    return f"{int(record['layer'])}-{int(record['head'])}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-file", required=True, help="Saved probe pool JSON.")
    parser.add_argument("--pool-name", default="layer_matched_random")
    parser.add_argument("--reference-head-file", required=True, help="Ranked file used for score profile/filler heads.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--score-key", default="score")
    parser.add_argument("--source-summary-file", default=None)
    args = parser.parse_args()

    pool = load_pool(args.pool_file, args.pool_name)
    reference = load_ranked_records(args.reference_head_file)
    if args.topk <= 0 or args.topk > len(pool):
        raise ValueError(f"--topk must be in [1, {len(pool)}], got {args.topk}")
    if args.topk > len(reference):
        raise ValueError(f"--topk={args.topk} exceeds reference length {len(reference)}")

    active = pool[: args.topk]
    score_profile = [
        float(rec.get(args.score_key, rec.get("score", 1.0)))
        for rec in reference[: args.topk]
    ]
    reference_by_key = {head_key(rec): rec for rec in reference}

    active_keys = set()
    out_heads = []
    for idx, rec in enumerate(active):
        key = head_key(rec)
        active_keys.add(key)
        original = reference_by_key.get(key, {})
        assigned_score = score_profile[idx] if idx < len(score_profile) else 1.0
        out_heads.append({
            "layer": int(rec["layer"]),
            "head": int(rec["head"]),
            "head_id": f"L{int(rec['layer'])}H{int(rec['head'])}",
            "score": float(assigned_score),
            args.score_key: float(assigned_score),
            "assigned_score_from_reference_rank": idx + 1,
            "original_score": float(original.get(args.score_key, original.get("score", 1.0))),
            "original_global_rank": original.get("global_rank", original.get("source_rank")),
            "control_pool": args.pool_name,
        })

    for rec in reference:
        if head_key(rec) in active_keys:
            continue
        filler = dict(rec)
        filler["layer"] = int(filler["layer"])
        filler["head"] = int(filler["head"])
        filler.setdefault("head_id", f"L{int(filler['layer'])}H{int(filler['head'])}")
        filler["control_pool"] = "filler_not_selected"
        out_heads.append(filler)

    output = {
        "score_name": args.pool_name,
        "description": (
            "Saved probe pool exported for DEACT caption evaluation. The active top-k heads "
            "come from the historical probe pool; the reference score profile is copied by "
            "rank and non-selected reference heads are appended for rank-percentile normalization."
        ),
        "source_pool_file": args.pool_file,
        "source_summary_file": args.source_summary_file,
        "reference_head_file": args.reference_head_file,
        "pool_name": args.pool_name,
        "topk": int(args.topk),
        "num_active_heads": len(active),
        "num_total_records": len(out_heads),
        "score_key": args.score_key,
        "heads": out_heads,
    }

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(
        f"saved probe pool heads -> {out_path} "
        f"({len(active)} active / {len(out_heads)} total records, pool={args.pool_name})"
    )


if __name__ == "__main__":
    main()
