import argparse
import json
import math
import os


def load_head_records(path, score_key):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "heads" in data:
        records = data["heads"]
        inferred_key = data.get("score_name") or score_key
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        records = data
        inferred_key = score_key
    else:
        raise ValueError(f"Unsupported head file format: {path}")

    out = {}
    for rank, item in enumerate(records, start=1):
        layer = int(item["layer"])
        head = int(item["head"])
        key = f"{layer}-{head}"
        if score_key in item:
            score = float(item[score_key])
        elif inferred_key in item:
            score = float(item[inferred_key])
        elif "score" in item:
            score = float(item["score"])
        else:
            score = 1.0 / rank
        out[key] = {
            "layer": layer,
            "head": head,
            "proxy_rank": int(item.get("global_rank", rank)),
            "proxy_score": score,
            "raw": item,
        }
    return out


def load_dla_records(path):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("heads", data if isinstance(data, list) else [])
    out = {}
    for item in records:
        layer = int(item["layer"])
        head = int(item["head"])
        out[f"{layer}-{head}"] = item
    return out, data.get("summary", {})


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(values) / len(values) if values else None


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def summarize_group(records):
    if not records:
        return {"count": 0}
    keys = [
        "proxy_score",
        "dla_contrast_hall_minus_nonhall",
        "hall_mean_dla",
        "nonhall_mean_dla",
        "object_mean_dla",
        "all_mean_dla",
    ]
    out = {"count": len(records)}
    for key in keys:
        out[f"mean_{key}"] = mean([x.get(key) for x in records])
    positives = [
        x.get("dla_contrast_hall_minus_nonhall")
        for x in records
        if x.get("dla_contrast_hall_minus_nonhall") is not None
    ]
    out["positive_dla_contrast_rate"] = (
        sum(1 for x in positives if x > 0) / len(positives) if positives else None
    )
    return out


def topk_enrichment(records, topk_values):
    by_proxy = sorted(records, key=lambda x: (x["proxy_rank"], -x["proxy_score"]))
    by_dla = sorted(
        records,
        key=lambda x: x.get("dla_contrast_hall_minus_nonhall", float("-inf")),
        reverse=True,
    )
    all_mean = mean([x.get("dla_contrast_hall_minus_nonhall") for x in records])
    out = {}
    for k in topk_values:
        k = min(int(k), len(records))
        if k <= 0:
            continue
        top_proxy = by_proxy[:k]
        rest_proxy = by_proxy[k:]
        top_dla_ids = {x["head_id"] for x in by_dla[:k]}
        top_proxy_ids = {x["head_id"] for x in top_proxy}
        top_mean = mean([x.get("dla_contrast_hall_minus_nonhall") for x in top_proxy])
        rest_mean = mean([x.get("dla_contrast_hall_minus_nonhall") for x in rest_proxy])
        out[str(k)] = {
            "top_proxy": summarize_group(top_proxy),
            "rest": summarize_group(rest_proxy),
            "dla_contrast_lift_vs_all": None if top_mean is None or all_mean is None else top_mean - all_mean,
            "dla_contrast_lift_vs_rest": None if top_mean is None or rest_mean is None else top_mean - rest_mean,
            "overlap_with_top_dla_count": len(top_proxy_ids & top_dla_ids),
            "overlap_with_top_dla_rate": len(top_proxy_ids & top_dla_ids) / k,
        }
    return out


def quantile_table(records, num_bins):
    by_proxy = sorted(records, key=lambda x: (x["proxy_rank"], -x["proxy_score"]))
    n = len(by_proxy)
    bins = []
    for i in range(num_bins):
        start = round(i * n / num_bins)
        end = round((i + 1) * n / num_bins)
        chunk = by_proxy[start:end]
        if not chunk:
            continue
        bins.append({
            "bin": i + 1,
            "rank_start": chunk[0]["proxy_rank"],
            "rank_end": chunk[-1]["proxy_rank"],
            **summarize_group(chunk),
        })
    return bins


def analyze(proxy_file, dla_file, score_key, topk_values, num_bins):
    proxy = load_head_records(proxy_file, score_key)
    dla, dla_summary = load_dla_records(dla_file)
    records = []
    for key, proxy_item in proxy.items():
        if key not in dla:
            continue
        item = {
            **proxy_item,
            **dla[key],
            "head_id": key,
        }
        records.append(item)

    records.sort(key=lambda x: (x["proxy_rank"], -x["proxy_score"]))
    xs = [x["proxy_score"] for x in records]
    ys = [x["dla_contrast_hall_minus_nonhall"] for x in records]

    return {
        "summary": {
            "num_proxy_heads": len(proxy),
            "num_dla_heads": len(dla),
            "num_matched_heads": len(records),
            "pearson_proxy_score_vs_dla_contrast": pearson(xs, ys),
            "spearman_proxy_score_vs_dla_contrast": spearman(xs, ys),
            "spearman_proxy_rank_vs_dla_contrast": spearman(
                [-x["proxy_rank"] for x in records],
                ys,
            ),
            "dla_summary": dla_summary,
        },
        "topk_enrichment": topk_enrichment(records, topk_values),
        "proxy_rank_quantiles": quantile_table(records, num_bins),
        "matched_heads": records,
        "config": {
            "proxy_file": proxy_file,
            "dla_file": dla_file,
            "score_key": score_key,
            "topk_values": topk_values,
            "num_bins": num_bins,
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
    parser.add_argument("--proxy-file", type=str, required=True)
    parser.add_argument("--dla-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--score-key", type=str, default="combo_mean_txtraw_Cratio")
    parser.add_argument("--topk", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--num-bins", type=int, default=10)
    args = parser.parse_args()

    result = analyze(args.proxy_file, args.dla_file, args.score_key, args.topk, args.num_bins)
    write_json(args.output_file, result)
    print(json.dumps(result["summary"], indent=2))
