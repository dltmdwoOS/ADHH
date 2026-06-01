#!/usr/bin/env python3
import argparse
import json
import math
import os

BUCKETS = ("all", "object", "hallucinated", "non_hallucinated")


def finite(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def mean_metric(item, bucket, metric):
    return finite(item.get("buckets", {}).get(bucket, {}).get("metrics", {}).get(metric, {}).get("mean", 0.0))


def ratio_from_item(item, bucket, eps):
    text = mean_metric(item, bucket, "I_text")
    image = mean_metric(item, bucket, "image_attn")
    return max(0.0, min(text / (text + image + eps), 1.0))


def quantile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def round_tau(value, step, mode):
    step = float(step)
    if step <= 0:
        return value
    if mode == "floor":
        rounded = math.floor(value / step) * step
    elif mode == "ceil":
        rounded = math.ceil(value / step) * step
    else:
        rounded = round(value / step) * step
    return max(0.0, min(rounded, 1.0))


def format_tau(value):
    return f"{value:.2f}"


def estimate(args):
    with open(os.path.expanduser(args.summary_file), "r", encoding="utf-8") as f:
        summary = json.load(f)

    by_head = list(summary.get("by_head", {}).values())
    if not by_head:
        raise ValueError(f"No by_head entries found in {args.summary_file}")

    ratios = {
        bucket: [ratio_from_item(item, bucket, args.eps) for item in by_head]
        for bucket in BUCKETS
    }
    stats = {}
    for bucket, values in ratios.items():
        stats[bucket] = {
            "mean": sum(values) / max(len(values), 1),
            "q25": quantile(values, 0.25),
            "q50": quantile(values, 0.50),
            "q75": quantile(values, 0.75),
            "q90": quantile(values, 0.90),
        }

    hall_q50 = stats["hallucinated"]["q50"]
    tau = round_tau(hall_q50, args.round_step, args.round_mode)
    tau = max(float(args.min_tau), min(float(args.max_tau), tau))

    out = {
        "summary_file": args.summary_file,
        "num_heads": len(by_head),
        "method": "hallucinated_head_median_ratio",
        "ratio_definition": "I_text / (I_text + image_attn + eps), computed from per-head bucket means in txtattn_summary.json",
        "recommended_tau": tau,
        "recommended_tau_str": format_tau(tau),
        "round_step": args.round_step,
        "round_mode": args.round_mode,
        "bucket_ratio_stats": stats,
        "notes": [
            "This follows the earlier heuristic: choose tau near the hallucinated-object median ratio as the center of the high text-reliance regime.",
            "The value is rounded conservatively downward by default, matching the previous 0.916 -> 0.90 choice.",
            "This is not a validation-set hyperparameter search; it is a distributional rule derived from training txt-attn statistics.",
        ],
    }

    if args.output_file:
        output_file = os.path.expanduser(args.output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    if args.print_value_only:
        print(out["recommended_tau_str"])
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate dynamic tau from txt-attn summary bucket statistics.")
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--output-file", default="")
    parser.add_argument("--round-step", type=float, default=0.05)
    parser.add_argument("--round-mode", choices=["floor", "nearest", "ceil"], default="floor")
    parser.add_argument("--min-tau", type=float, default=0.0)
    parser.add_argument("--max-tau", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--print-value-only", action="store_true")
    estimate(parser.parse_args())
