#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import OrderedDict, defaultdict

BUCKETS = ("all", "object", "hallucinated", "non_hallucinated")

VERIFIED = {
    "verified_HH_git": [
        [16, 29], [26, 9], [13, 31], [15, 10], [20, 12], [30, 9], [19, 18], [17, 0], [18, 9], [26, 28],
        [19, 27], [18, 26], [15, 25], [14, 16], [31, 26], [15, 24], [31, 3], [22, 20], [27, 29], [17, 28],
        [20, 5], [12, 10], [20, 20], [15, 9], [22, 9], [19, 23], [16, 9], [29, 8], [13, 9], [14, 14],
    ],
    "verified_HH_yield": [
        [16, 29], [20, 12], [26, 9], [13, 31], [30, 9], [18, 9], [15, 10], [19, 18], [14, 16], [17, 0],
        [26, 28], [20, 5], [18, 26], [15, 24], [15, 25], [19, 27], [31, 26], [22, 20], [15, 9], [27, 29],
        [31, 3], [19, 12], [12, 10], [20, 20], [14, 4], [22, 17], [22, 9], [30, 5], [14, 14], [31, 27],
    ],
}


def finite(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def head_id(layer, head):
    return f"L{int(layer)}H{int(head)}"


def mean_metric(item, bucket, metric):
    return finite(item["buckets"].get(bucket, {}).get("metrics", {}).get(metric, {}).get("mean", 0.0))


def rank_percentiles(values, reverse=True):
    ordered = sorted(values, key=values.get, reverse=reverse)
    n = len(ordered)
    if n <= 1:
        return {hid: 1.0 for hid in ordered}
    return {hid: 1.0 - i / (n - 1) for i, hid in enumerate(ordered)}


def logtoi(text, image, eps):
    return math.log1p(max(text / (image + eps), 0.0))


def build_features(summary, eps):
    features = OrderedDict()
    for _, item in summary["by_head"].items():
        layer = int(item["layer"])
        head = int(item["head"])
        hid = head_id(layer, head)
        f = {"layer": layer, "head": head, "head_id": hid}
        for bucket in BUCKETS:
            text = mean_metric(item, bucket, "generated_txt_attn")
            itext = mean_metric(item, bucket, "I_text")
            image = mean_metric(item, bucket, "image_attn")
            raw_toi = mean_metric(item, bucket, "txt_img_ratio")
            f[f"T_{bucket}"] = text
            f[f"Itext_{bucket}"] = itext
            f[f"Img_{bucket}"] = image
            f[f"RawTOI_{bucket}"] = raw_toi
            f[f"LogTOI_{bucket}"] = logtoi(text, image, eps)
        features[hid] = f
    return features


def build_score_values(features):
    fronts = {
        "itext_all": {hid: f["Itext_all"] for hid, f in features.items()},
        "txt_attn_raw_all": {hid: f["T_all"] for hid, f in features.items()},
    }
    backs = {
        "C_toi_HminusG": {
            hid: max(0.0, f["RawTOI_hallucinated"] - f["RawTOI_non_hallucinated"])
            for hid, f in features.items()
        },
        "C_logtoi_HminusG": {
            hid: max(0.0, f["LogTOI_hallucinated"] - f["LogTOI_non_hallucinated"])
            for hid, f in features.items()
        },
        "C_itext_mass_HminusG": {
            hid: max(0.0, f["Itext_hallucinated"] - f["Itext_non_hallucinated"])
            for hid, f in features.items()
        },
    }
    return fronts, backs


def global_order(records, score_name):
    return sorted(records, key=lambda x: (x[score_name], x.get("front_percentile", 0.0), x.get("back_percentile", 0.0)), reverse=True)


def local_round_robin_order(records, score_name):
    global_ranked = global_order(records, score_name)
    layer_order = []
    seen_layers = set()
    for item in global_ranked:
        layer = int(item["layer"])
        if layer not in seen_layers:
            seen_layers.add(layer)
            layer_order.append(layer)

    by_layer = defaultdict(list)
    for item in global_ranked:
        by_layer[int(item["layer"])].append(item)
    for layer_items in by_layer.values():
        layer_items.sort(key=lambda x: (x[score_name], x.get("front_percentile", 0.0), x.get("back_percentile", 0.0)), reverse=True)

    ordered = []
    max_len = max(len(items) for items in by_layer.values()) if by_layer else 0
    for local_rank_idx in range(max_len):
        for layer_priority, layer in enumerate(layer_order, start=1):
            items = by_layer[layer]
            if local_rank_idx < len(items):
                item = dict(items[local_rank_idx])
                item["layer_priority"] = layer_priority
                item["local_rank_in_layer"] = local_rank_idx + 1
                ordered.append(item)
    return ordered, layer_order


def weighted_layer_order(records, score_field, topm=5, alpha=2.0, total_k=None, min_quota=1):
    layers = sorted({int(item["layer"]) for item in records})
    by_layer = {layer: [] for layer in layers}
    for item in records:
        by_layer[int(item["layer"])].append(item)
    layer_scores = {}
    for layer, items in by_layer.items():
        ranked = sorted(items, key=lambda x: (x[score_field], x.get("front_percentile", 0.0), x.get("back_percentile", 0.0)), reverse=True)
        top = ranked[:max(int(topm), 1)]
        layer_scores[layer] = sum(float(x[score_field]) for x in top) / max(len(top), 1)

    if total_k is None:
        total_k = len(records)
    total_k = min(int(total_k), len(records))
    min_quota = max(int(min_quota), 0)

    weights = {layer: max(score, 0.0) ** float(alpha) for layer, score in layer_scores.items()}
    if sum(weights.values()) <= 0:
        weights = {layer: 1.0 for layer in layers}
    weight_sum = sum(weights.values())

    quotas = {layer: min_quota for layer in layers}
    remaining = max(total_k - sum(quotas.values()), 0)
    raw_extra = {layer: remaining * weights[layer] / weight_sum for layer in layers}
    for layer in layers:
        quotas[layer] += int(math.floor(raw_extra[layer]))
    used = sum(quotas.values())
    remainders = sorted(layers, key=lambda layer: (raw_extra[layer] - math.floor(raw_extra[layer]), layer_scores[layer]), reverse=True)
    idx = 0
    while used < total_k and remainders:
        quotas[remainders[idx % len(remainders)]] += 1
        used += 1
        idx += 1

    # Clamp quotas to available heads and redistribute overflow if needed.
    overflow = 0
    for layer in layers:
        if quotas[layer] > len(by_layer[layer]):
            overflow += quotas[layer] - len(by_layer[layer])
            quotas[layer] = len(by_layer[layer])
    while overflow > 0:
        candidates = [layer for layer in sorted(layers, key=lambda l: layer_scores[l], reverse=True) if quotas[layer] < len(by_layer[layer])]
        if not candidates:
            break
        for layer in candidates:
            if overflow <= 0:
                break
            quotas[layer] += 1
            overflow -= 1

    layer_order = sorted(layers, key=lambda layer: layer_scores[layer], reverse=True)
    return by_layer, layer_order, quotas, layer_scores


def weighted_layer_ordered_heads(records, score_field, topm=5, alpha=2.0, total_k=None, min_quota=1):
    by_layer, layer_order, quotas, layer_scores = weighted_layer_order(records, score_field, topm, alpha, total_k, min_quota)
    out = []
    for layer_priority, layer in enumerate(layer_order, start=1):
        ranked = sorted(by_layer[layer], key=lambda x: (x[score_field], x.get("front_percentile", 0.0), x.get("back_percentile", 0.0)), reverse=True)
        for local_rank, item in enumerate(ranked[:quotas[layer]], start=1):
            copied = dict(item)
            copied["layer_priority"] = layer_priority
            copied["local_rank_in_layer"] = local_rank
            copied["layer_weighted_quota"] = quotas[layer]
            copied["layer_weighted_score"] = layer_scores[layer]
            out.append(copied)
    return out, layer_order, quotas, layer_scores


def annotate_ranks(ranked, score_field, selection_method):
    by_layer = defaultdict(list)
    for rank, item in enumerate(ranked, start=1):
        item["global_rank"] = rank
        item["selection_method"] = selection_method
        by_layer[int(item["layer"])].append(item)
    for items in by_layer.values():
        sorted_items = sorted(items, key=lambda x: (x[score_field], x.get("front_percentile", 0.0), x.get("back_percentile", 0.0)), reverse=True)
        for local_rank, item in enumerate(sorted_items, start=1):
            item["layer_rank"] = local_rank


def write_ranked(path, score_name, score_field, description, layer_range, ranked, selection_method, layer_order):
    annotate_ranks(ranked, score_field, selection_method)
    obj = {
        "score_name": score_name,
        "description": description,
        "layer_range": layer_range,
        "selection_method": selection_method,
        "layer_order": layer_order,
        "heads": ranked,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def overlap_stats(ranked):
    out = {}
    ranked_pairs = [(int(x["layer"]), int(x["head"])) for x in ranked]
    available = set(ranked_pairs)
    for verified_name, pairs in VERIFIED.items():
        verified = {(int(l), int(h)) for l, h in pairs}
        missing = sorted(verified - available)
        for k in (20, 50, 100):
            top = set(ranked_pairs[:k])
            hit = sorted(top & verified)
            out[(verified_name, k)] = {
                "overlap": len(hit),
                "hits": " ".join(head_id(l, h) for l, h in hit),
                "missing_verified": " ".join(head_id(l, h) for l, h in missing),
                "verified_available": len(verified & available),
                "verified_total": len(verified),
            }
    return out


def main(args):
    with open(os.path.expanduser(args.summary_file), "r", encoding="utf-8") as f:
        summary = json.load(f)
    features = build_features(summary, args.eps)
    fronts, backs = build_score_values(features)

    layer_values = [f["layer"] for f in features.values()]
    layer_range = [min(layer_values), max(layer_values)] if layer_values else None
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    overview_rows = []
    for front_name, front_values in fronts.items():
        front_pct = rank_percentiles(front_values)
        for back_name, back_values in backs.items():
            back_pct = rank_percentiles(back_values)
            combo_name = f"{front_name}__{back_name}"
            combo_values = {
                hid: 0.5 * front_pct[hid] + 0.5 * back_pct[hid]
                for hid in features
            }
            records = []
            for hid, f in features.items():
                records.append({
                    "layer": f["layer"],
                    "head": f["head"],
                    "head_id": hid,
                    combo_name: combo_values[hid],
                    "front_name": front_name,
                    "front_raw": front_values[hid],
                    "front_percentile": front_pct[hid],
                    "back_name": back_name,
                    "back_raw": back_values[hid],
                    "back_percentile": back_pct[hid],
                    "T_all": f["T_all"],
                    "Itext_all": f["Itext_all"],
                    "RawTOI_hallucinated": f["RawTOI_hallucinated"],
                    "RawTOI_non_hallucinated": f["RawTOI_non_hallucinated"],
                    "LogTOI_hallucinated": f["LogTOI_hallucinated"],
                    "LogTOI_non_hallucinated": f["LogTOI_non_hallucinated"],
                    "Itext_hallucinated": f["Itext_hallucinated"],
                    "Itext_non_hallucinated": f["Itext_non_hallucinated"],
                    "Img_hallucinated": f["Img_hallucinated"],
                    "Img_non_hallucinated": f["Img_non_hallucinated"],
                })

            for selection_method in ("global", "local", "layerweighted_top5mean_alpha2"):
                if selection_method == "global":
                    ranked = [dict(x) for x in global_order(records, combo_name)]
                    layer_order = []
                elif selection_method == "local":
                    ranked, layer_order = local_round_robin_order(records, combo_name)
                else:
                    ranked, layer_order, layer_quotas, layer_scores = weighted_layer_ordered_heads(
                        records, combo_name, topm=5, alpha=2.0, total_k=len(records), min_quota=1
                    )
                score_name = f"{selection_method}__{combo_name}"
                description = (
                    f"{selection_method} selection over rank-percentile combo: "
                    f"0.5*P({front_name}) + 0.5*P({back_name}). "
                    "local selection orders layers by first global occurrence, then round-robins layer-local ranks."
                )
                out_path = os.path.join(output_dir, f"ranked_heads_{score_name}.json")
                write_ranked(out_path, score_name, combo_name, description, layer_range, ranked, selection_method, layer_order)

                overlaps = overlap_stats(ranked)
                for verified_name in VERIFIED:
                    row = {
                        "score_name": score_name,
                        "selection_method": selection_method,
                        "front": front_name,
                        "back": back_name,
                        "path": out_path,
                        "verified_set": verified_name,
                        "top20": overlaps[(verified_name, 20)]["overlap"],
                        "top50": overlaps[(verified_name, 50)]["overlap"],
                        "top100": overlaps[(verified_name, 100)]["overlap"],
                        "verified_available": overlaps[(verified_name, 20)]["verified_available"],
                        "verified_total": overlaps[(verified_name, 20)]["verified_total"],
                        "missing_verified": overlaps[(verified_name, 20)]["missing_verified"],
                        "top20_hits": overlaps[(verified_name, 20)]["hits"],
                        "top50_hits": overlaps[(verified_name, 50)]["hits"],
                        "top100_hits": overlaps[(verified_name, 100)]["hits"],
                        "top20_heads": " ".join(item["head_id"] for item in ranked[:20]),
                    }
                    overview_rows.append(row)

    overview_path = os.path.join(output_dir, "surrogate_combo_overlap_summary.csv")
    fields = [
        "score_name", "selection_method", "front", "back", "verified_set", "top20", "top50", "top100",
        "verified_available", "verified_total", "missing_verified", "top20_hits", "top50_hits", "top100_hits",
        "top20_heads", "path",
    ]
    with open(overview_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(overview_rows)

    config = {
        "summary_file": args.summary_file,
        "output_dir": output_dir,
        "num_heads": len(features),
        "layers": sorted(set(layer_values)),
        "fronts": list(fronts),
        "backs": list(backs),
        "selection_methods": ["global", "local", "layerweighted_top5mean_alpha2"],
        "num_ranked_files": len(fronts) * len(backs) * 2,
        "overlap_summary": overview_path,
        "note": "Layer 12 is absent from the current all-head trace, so each verified set has one unavailable head L12H10.",
    }
    with open(os.path.join(output_dir, "surrogate_combo_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("\nTop overlap rows:")
    for row in sorted(overview_rows, key=lambda r: (r["verified_set"], int(r["top100"]), int(r["top50"]), int(r["top20"])), reverse=True)[:24]:
        print(f"{row['verified_set']} {row['score_name']} top20={row['top20']} top50={row['top50']} top100={row['top100']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eps", type=float, default=1e-12)
    main(parser.parse_args())
