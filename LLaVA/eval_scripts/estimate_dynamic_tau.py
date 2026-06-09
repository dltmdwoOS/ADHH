#!/usr/bin/env python3
import argparse
import json
import math
import os

BUCKETS = ("all", "object", "hallucinated", "non_hallucinated")
DEFAULT_QUANTILES = (
    ("q25", 0.25),
    ("q33", 1.0 / 3.0),
    ("q50", 0.50),
    ("q66", 2.0 / 3.0),
    ("q75", 0.75),
    ("q90", 0.90),
)


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


def load_selected_head_keys(head_file, topk):
    if not head_file:
        return None
    with open(os.path.expanduser(head_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    heads = data.get("heads", data if isinstance(data, list) else [])
    selected = []
    for item in heads[: max(int(topk), 0)]:
        if isinstance(item, dict):
            layer = int(item["layer"])
            head = int(item["head"])
        else:
            layer = int(item[0])
            head = int(item[1])
        selected.append(f"{layer}-{head}")
    return set(selected)


def summarize_ratios(by_head, eps, quantiles=DEFAULT_QUANTILES):
    ratios = {
        bucket: [ratio_from_item(item, bucket, eps) for item in by_head]
        for bucket in BUCKETS
    }
    stats = {}
    for bucket, values in ratios.items():
        bucket_stats = {
            "count": len(values),
            "mean": sum(values) / max(len(values), 1),
        }
        for name, q in quantiles:
            bucket_stats[name] = quantile(values, q)
        stats[bucket] = bucket_stats
    return stats


def parse_topk_list(value, default_topk):
    if value is None or str(value).strip() == "":
        return [int(default_topk)]
    topks = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        topks.append(int(part))
    if not topks:
        topks.append(int(default_topk))
    return topks


def select_heads(all_by_head_map, selected_keys):
    if selected_keys is None:
        return list(all_by_head_map.values()), []
    by_head = [all_by_head_map[k] for k in selected_keys if k in all_by_head_map]
    missing = sorted(selected_keys - set(all_by_head_map))
    if not by_head:
        raise ValueError("No selected heads from --head-file were found in the summary")
    return by_head, missing


def calibration_from_stats(stats, bucket, hi_quantile, lo_quantile, args):
    hi_raw = stats[bucket][hi_quantile]
    lo_raw = stats[bucket][lo_quantile]
    tau_hi = round_tau(hi_raw, args.round_step, args.round_mode)
    tau_lo = round_tau(lo_raw, args.round_step, args.round_mode)
    tau_hi = max(float(args.min_tau), min(float(args.max_tau), tau_hi))
    tau_lo = max(float(args.min_tau), min(float(args.max_tau), tau_lo))
    return {
        "bucket": bucket,
        "hi_quantile": hi_quantile,
        "lo_quantile": lo_quantile,
        "hi_raw": hi_raw,
        "lo_raw": lo_raw,
        "tau_hi": tau_hi,
        "tau_hi_str": format_tau(tau_hi),
        "tau_lo": tau_lo,
        "tau_lo_str": format_tau(tau_lo),
    }


def estimate(args):
    with open(os.path.expanduser(args.summary_file), "r", encoding="utf-8") as f:
        summary = json.load(f)

    all_by_head_map = summary.get("by_head", {})
    if not all_by_head_map:
        raise ValueError(f"No by_head entries found in {args.summary_file}")

    topk_list = parse_topk_list(args.topk_list, args.topk)
    all_stats = summarize_ratios(list(all_by_head_map.values()), args.eps)
    layer_all_calibration = calibration_from_stats(
        all_stats,
        args.calibration_bucket,
        args.hi_quantile,
        args.lo_quantile,
        args,
    )

    selected_by_topk = {}
    missing_by_topk = {}
    selected_stats = all_stats
    selected_calibration = layer_all_calibration
    by_head = list(all_by_head_map.values())
    missing = []
    selection_note = "all heads in summary"

    if args.head_file:
        for topk in topk_list:
            selected_keys = load_selected_head_keys(args.head_file, topk)
            selected_heads, missing_heads = select_heads(all_by_head_map, selected_keys)
            stats = summarize_ratios(selected_heads, args.eps)
            selected_by_topk[str(topk)] = {
                "topk": topk,
                "num_heads_used": len(selected_heads),
                "ratio_stats": stats,
                "calibration": calibration_from_stats(
                    stats,
                    args.calibration_bucket,
                    args.hi_quantile,
                    args.lo_quantile,
                    args,
                ),
            }
            missing_by_topk[str(topk)] = missing_heads

        selected_keys = load_selected_head_keys(args.head_file, args.topk)
        by_head, missing = select_heads(all_by_head_map, selected_keys)
        topk_key = str(args.topk)
        if topk_key not in selected_by_topk:
            stats = summarize_ratios(by_head, args.eps)
            selected_by_topk[topk_key] = {
                "topk": args.topk,
                "num_heads_used": len(by_head),
                "ratio_stats": stats,
                "calibration": calibration_from_stats(
                    stats,
                    args.calibration_bucket,
                    args.hi_quantile,
                    args.lo_quantile,
                    args,
                ),
            }
            missing_by_topk[topk_key] = missing
        selected_stats = selected_by_topk[topk_key]["ratio_stats"]
        selected_calibration = selected_by_topk[topk_key]["calibration"]
        selection_note = f"top-{args.topk} selected heads from {args.head_file}"

    if args.calibration_scope == "layer_all":
        main_calibration = layer_all_calibration
    else:
        main_calibration = selected_calibration

    out = {
        "summary_file": args.summary_file,
        "head_file": args.head_file,
        "topk": args.topk if args.head_file else None,
        "topk_list": topk_list if args.head_file else None,
        "num_heads_total": len(all_by_head_map),
        "num_heads_used": len(by_head),
        "method": f"{args.calibration_scope}_{args.calibration_bucket}_{args.hi_quantile}_{args.lo_quantile}_ratio",
        "calibration_bucket": args.calibration_bucket,
        "calibration_scope": args.calibration_scope,
        "hi_quantile": args.hi_quantile,
        "lo_quantile": args.lo_quantile,
        "ratio_definition": "I_text / (I_text + image_attn + eps), approximated from per-head bucket means in txtattn_summary.json",
        "recommended_tau_hi": main_calibration["tau_hi"],
        "recommended_tau_hi_str": main_calibration["tau_hi_str"],
        "recommended_tau_lo": main_calibration["tau_lo"],
        "recommended_tau_lo_str": main_calibration["tau_lo_str"],
        "recommended_tau": main_calibration["tau_hi"],
        "recommended_tau_str": main_calibration["tau_hi_str"],
        "round_step": args.round_step,
        "round_mode": args.round_mode,
        "selected_head_ratio_stats": selected_stats,
        "selected_head_by_topk": selected_by_topk,
        "all_head_ratio_stats": all_stats,
        "layer_all_calibration": layer_all_calibration,
        "selected_head_calibration": selected_calibration,
        "selection_note": selection_note,
        "notes": [
            "This script reports both selected-head and layer-all ratio statistics when --head-file is provided.",
            "Because txtattn_summary stores finalized per-head means rather than every raw step value, these quantiles are an approximation over head-level calibration means, not raw per-step quantiles.",
            "recommended_tau is kept as an alias for tau_hi for backward compatibility with older pipeline code.",
        ],
    }
    if args.head_file:
        out["missing_selected_heads"] = missing
        out["missing_selected_heads_by_topk"] = missing_by_topk

    if args.output_file:
        output_file = os.path.expanduser(args.output_file)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    if args.print_value_only:
        print(out["recommended_tau_hi_str"])
    elif args.print_late_value_only:
        print(out["recommended_tau_lo_str"])
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate dynamic tau_hi/tau_lo from txt-attn summary bucket statistics.")
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--output-file", default="")
    parser.add_argument("--head-file", default="", help="Optional ranked head JSON; if provided, report top-K selected actuator-head stats.")
    parser.add_argument("--topk", type=int, default=100, help="Top-k used for recommended selected-head stats.")
    parser.add_argument("--topk-list", default="", help="Comma/semicolon-separated top-k values to report, e.g. '20,50,100'.")
    parser.add_argument("--calibration-bucket", choices=BUCKETS, default="all")
    parser.add_argument("--calibration-scope", choices=["selected_head", "layer_all"], default="selected_head",
                        help="Which statistics define recommended_tau_hi/lo. Both are reported when --head-file is provided.")
    parser.add_argument("--hi-quantile", choices=[name for name, _ in DEFAULT_QUANTILES], default="q75")
    parser.add_argument("--lo-quantile", choices=[name for name, _ in DEFAULT_QUANTILES], default="q50")
    parser.add_argument("--round-step", type=float, default=0.01)
    parser.add_argument("--round-mode", choices=["floor", "nearest", "ceil"], default="floor")
    parser.add_argument("--min-tau", type=float, default=0.0)
    parser.add_argument("--max-tau", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--print-value-only", action="store_true")
    parser.add_argument("--print-late-value-only", action="store_true")
    estimate(parser.parse_args())
