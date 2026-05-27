import argparse
import json
import math
import os
from collections import defaultdict


METRICS = (
    "I_text",
    "generated_txt_attn",
    "text_image_fraction",
    "log_txt_img_ratio",
)
BUCKETS = ("all", "object", "hallucinated", "non_hallucinated")


def head_key(layer, head):
    return f"{int(layer)}-{int(head)}"


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(values) / len(values) if values else None


def variance(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return None
    m = sum(values) / len(values)
    return sum((x - m) ** 2 for x in values) / len(values)


def percentile(values, q):
    values = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def cohen_d(pos, neg):
    pos = [float(x) for x in pos if x is not None and math.isfinite(float(x))]
    neg = [float(x) for x in neg if x is not None and math.isfinite(float(x))]
    if len(pos) < 2 or len(neg) < 2:
        return None
    mp, mn = mean(pos), mean(neg)
    vp, vn = variance(pos), variance(neg)
    pooled = ((len(pos) - 1) * vp + (len(neg) - 1) * vn) / (len(pos) + len(neg) - 2)
    if pooled <= 0:
        return None
    return (mp - mn) / math.sqrt(pooled)


def auc_mann_whitney(pos, neg):
    pos = [float(x) for x in pos if x is not None and math.isfinite(float(x))]
    neg = [float(x) for x in neg if x is not None and math.isfinite(float(x))]
    if not pos or not neg:
        return None
    merged = [(x, 1) for x in pos] + [(x, 0) for x in neg]
    merged.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(merged):
        j = i + 1
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(label for _, label in merged[i:j])
        i = j
    n_pos = len(pos)
    n_neg = len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def safe_rate_above(values, threshold):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not values:
        return None
    return sum(1 for x in values if x >= threshold) / len(values)


def summarize_values(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {
        "count": len(values),
        "mean": mean(values),
        "std": None if variance(values) is None else math.sqrt(variance(values)),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
    }


def compare_buckets(pos, neg, thresholds):
    out = {
        "positive": summarize_values(pos),
        "negative": summarize_values(neg),
        "mean_diff": None,
        "cohen_d": cohen_d(pos, neg),
        "roc_auc": auc_mann_whitney(pos, neg),
    }
    if out["positive"]["mean"] is not None and out["negative"]["mean"] is not None:
        out["mean_diff"] = out["positive"]["mean"] - out["negative"]["mean"]
    out["threshold_rates"] = {
        str(t): {
            "positive_rate": safe_rate_above(pos, t),
            "negative_rate": safe_rate_above(neg, t),
            "rate_diff": None,
        }
        for t in thresholds
    }
    for item in out["threshold_rates"].values():
        if item["positive_rate"] is not None and item["negative_rate"] is not None:
            item["rate_diff"] = item["positive_rate"] - item["negative_rate"]
    return out


def load_proxy_heads(path, score_key):
    if not path:
        return {}
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "heads" in data:
        records = data["heads"]
        inferred_score_key = data.get("score_name") or score_key
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        records = data
        inferred_score_key = score_key
    elif isinstance(data, dict) and "hal_heads" in data:
        records = [{"layer": l, "head": h} for l, h in data["hal_heads"]]
        inferred_score_key = score_key
    elif isinstance(data, list) and data and isinstance(data[0], list):
        records = [{"layer": l, "head": h} for l, h in data]
        inferred_score_key = score_key
    else:
        raise ValueError(f"Unsupported head file format: {path}")

    out = {}
    for idx, item in enumerate(records, start=1):
        layer = int(item["layer"])
        head = int(item["head"])
        score = item.get(score_key, item.get(inferred_score_key, item.get("score", 1.0 / idx)))
        out[head_key(layer, head)] = {
            "layer": layer,
            "head": head,
            "rank": int(item.get("global_rank", item.get("candidate_global_rank", idx))),
            "score": float(score),
        }
    return out


def step_buckets(record):
    buckets = ["all"]
    if record.get("is_object"):
        buckets.append("object")
    if record.get("is_hallucinated"):
        buckets.append("hallucinated")
    if record.get("is_non_hallucinated"):
        buckets.append("non_hallucinated")
    return buckets


def head_metrics(item, eps):
    generated = float(item.get("generated_txt_attn", 0.0))
    image = float(item.get("image_attn", 0.0))
    ratio = generated / (image + eps)
    return {
        "I_text": float(item.get("I_text", 0.0)),
        "generated_txt_attn": generated,
        "text_image_fraction": generated / (generated + image + eps),
        "log_txt_img_ratio": math.log1p(max(ratio, 0.0)),
    }


def finalize_rank_bins(per_head_effects, num_bins):
    ranked = sorted(per_head_effects, key=lambda x: x["proxy_rank"])
    n = len(ranked)
    bins = []
    if n == 0:
        return bins
    for i in range(num_bins):
        start = round(i * n / num_bins)
        end = round((i + 1) * n / num_bins)
        chunk = ranked[start:end]
        if not chunk:
            continue
        item = {
            "bin": i + 1,
            "rank_start": chunk[0]["proxy_rank"],
            "rank_end": chunk[-1]["proxy_rank"],
            "count": len(chunk),
        }
        for metric in METRICS:
            effects = [x["metrics"][metric]["mean_diff"] for x in chunk]
            aucs = [x["metrics"][metric]["roc_auc"] for x in chunk]
            item[f"mean_{metric}_hall_minus_nonhall"] = mean(effects)
            item[f"mean_{metric}_auc"] = mean(aucs)
        bins.append(item)
    return bins


def analyze(args):
    proxy = load_proxy_heads(args.head_file, args.head_score_key)
    aggregate = {
        str(k): {bucket: {metric: [] for metric in METRICS} for bucket in BUCKETS}
        for k in args.topk
    }
    per_head = defaultdict(lambda: {bucket: {metric: [] for metric in METRICS} for bucket in BUCKETS})
    traced_heads = {}
    sample_ids = set()
    step_counts = {bucket: 0 for bucket in BUCKETS}

    with open(os.path.expanduser(args.trace_file), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_ids.add(record.get("question_id"))
            buckets = step_buckets(record)
            for bucket in buckets:
                step_counts[bucket] += 1

            values_by_head = {}
            for item in record.get("head_values", []):
                key = head_key(item["layer"], item["head"])
                if key not in traced_heads:
                    traced_heads[key] = {
                        "layer": int(item["layer"]),
                        "head": int(item["head"]),
                    }
                vals = head_metrics(item, args.eps)
                values_by_head[key] = vals
                for bucket in buckets:
                    for metric in METRICS:
                        per_head[key][bucket][metric].append(vals[metric])

            for k in args.topk:
                if proxy:
                    selected = [
                        key for key in values_by_head
                        if key in proxy and proxy[key]["rank"] <= k
                    ]
                else:
                    selected = list(values_by_head)[:k]
                if not selected:
                    continue
                for metric in METRICS:
                    value = sum(values_by_head[key][metric] for key in selected) / len(selected)
                    for bucket in buckets:
                        aggregate[str(k)][bucket][metric].append(value)

    per_head_effects = []
    for key, buckets in per_head.items():
        meta = proxy.get(key, {}) if proxy else {}
        layer, head = key.split("-")
        item = {
            "head_id": key,
            "layer": int(layer),
            "head": int(head),
            "proxy_rank": int(meta.get("rank", 10**9)),
            "proxy_score": float(meta.get("score", 0.0)),
            "metrics": {},
        }
        for metric in METRICS:
            item["metrics"][metric] = compare_buckets(
                buckets["hallucinated"][metric],
                buckets["non_hallucinated"][metric],
                args.thresholds,
            )
        per_head_effects.append(item)
    per_head_effects.sort(key=lambda x: x["proxy_rank"])

    topk_results = {}
    for k in args.topk:
        k_key = str(k)
        topk_results[k_key] = {"metrics": {}}
        for metric in METRICS:
            metric_out = {
                "hallucinated_vs_non_hallucinated": compare_buckets(
                    aggregate[k_key]["hallucinated"][metric],
                    aggregate[k_key]["non_hallucinated"][metric],
                    args.thresholds,
                ),
                "hallucinated_vs_all": compare_buckets(
                    aggregate[k_key]["hallucinated"][metric],
                    aggregate[k_key]["all"][metric],
                    args.thresholds,
                ),
                "object_vs_all": compare_buckets(
                    aggregate[k_key]["object"][metric],
                    aggregate[k_key]["all"][metric],
                    args.thresholds,
                ),
            }
            topk_results[k_key]["metrics"][metric] = metric_out

    return {
        "summary": {
            "num_samples": len(sample_ids),
            "num_traced_heads": len(traced_heads),
            "num_proxy_heads": len(proxy),
            "step_counts": step_counts,
            "interpretation": (
                "A hallucination-head proxy is supported when selected heads show high ROC-AUC, "
                "positive hall-minus-nonhall mean differences, and higher threshold exceedance rates "
                "for hallucinated object steps than grounded object steps."
            ),
        },
        "topk_behavior": topk_results,
        "per_head_behavior": per_head_effects,
        "rank_bin_behavior": finalize_rank_bins(per_head_effects, args.num_bins),
        "config": {
            "trace_file": args.trace_file,
            "head_file": args.head_file,
            "head_score_key": args.head_score_key,
            "topk": args.topk,
            "thresholds": args.thresholds,
            "num_bins": args.num_bins,
        },
    }


def write_json(path, obj):
    out_path = os.path.expanduser(path)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=str, required=True)
    parser.add_argument("--head-file", type=str, default="")
    parser.add_argument("--head-score-key", type=str, default="combo_mean_txtraw_Cratio")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.4, 0.5, 0.65, 0.9])
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    result = analyze(args)
    write_json(args.output_file, result)
    print(json.dumps(result["summary"], indent=2))
