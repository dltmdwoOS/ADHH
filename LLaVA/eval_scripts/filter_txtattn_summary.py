#!/usr/bin/env python3
import argparse
import json
import math
import os
from copy import deepcopy


def finite(value, default=0.0):
    try:
        value = float(value)
    except Exception:
        return default
    return value if math.isfinite(value) else default


def parse_layers(value):
    if value is None or str(value).strip() == "":
        return None
    layers = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, end = part.split(":", 1)
            start, end = int(start), int(end)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        elif "-" in part and not part.startswith("-"):
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            step = 1 if end >= start else -1
            layers.extend(range(start, end + step, step))
        else:
            layers.append(int(part))
    seen = set()
    out = []
    for layer in layers:
        if layer not in seen:
            out.append(layer)
            seen.add(layer)
    return out


def layer_slug(layers):
    layers = [int(x) for x in layers]
    if not layers:
        return "empty"
    if len(layers) == 1:
        return f"l{layers[0]}"
    return "l" + "_l".join(str(x) for x in layers)


def init_acc(metrics):
    return {
        "count": 0,
        "sum": {metric: 0.0 for metric in metrics},
        "sumsq": {metric: 0.0 for metric in metrics},
        "min": {metric: float("inf") for metric in metrics},
        "max": {metric: float("-inf") for metric in metrics},
    }


def add_finalized_bucket(acc, bucket):
    count = int(bucket.get("count", 0))
    if count <= 0:
        return
    acc["count"] += count
    for metric, values in bucket.get("metrics", {}).items():
        mean = finite(values.get("mean", 0.0))
        var = max(finite(values.get("var", 0.0)), 0.0)
        acc["sum"][metric] += mean * count
        acc["sumsq"][metric] += (var + mean * mean) * count
        vmin = values.get("min")
        vmax = values.get("max")
        if vmin is not None:
            acc["min"][metric] = min(acc["min"][metric], finite(vmin))
        if vmax is not None:
            acc["max"][metric] = max(acc["max"][metric], finite(vmax))


def finalize_acc(acc):
    count = int(acc["count"])
    denom = max(count, 1)
    out = {"count": count, "metrics": {}}
    for metric in acc["sum"]:
        mean = acc["sum"][metric] / denom
        var = max(acc["sumsq"][metric] / denom - mean * mean, 0.0)
        out["metrics"][metric] = {
            "mean": mean,
            "var": var,
            "min": None if count == 0 or acc["min"][metric] == float("inf") else acc["min"][metric],
            "max": None if count == 0 or acc["max"][metric] == float("-inf") else acc["max"][metric],
        }
    return out


def resolve_requested_layers(args):
    layers = parse_layers(args.layers)
    if layers is not None:
        return layers
    if args.min_layer is None or args.max_layer is None:
        raise ValueError("Provide either --layers or both --min-layer and --max-layer")
    step = 1 if args.max_layer >= args.min_layer else -1
    return list(range(args.min_layer, args.max_layer + step, step))


def filter_summary(args):
    with open(os.path.expanduser(args.summary_file), "r", encoding="utf-8") as f:
        summary = json.load(f)

    requested_layers = resolve_requested_layers(args)
    requested_set = set(int(x) for x in requested_layers)

    by_head = summary.get("by_head", {})
    filtered_by_head = {}
    for key, item in by_head.items():
        layer = int(item["layer"])
        if layer in requested_set:
            filtered_by_head[key] = item

    if not filtered_by_head:
        raise ValueError(f"No heads found in requested layers {requested_layers}")

    found_layers = sorted({int(item["layer"]) for item in filtered_by_head.values()})
    missing_layers = [layer for layer in requested_layers if layer not in found_layers]
    if missing_layers and not args.allow_missing_layers:
        raise ValueError(f"Requested layers absent from summary: {missing_layers}")

    bucket_names = list(next(iter(filtered_by_head.values())).get("buckets", {}).keys())
    metric_names = list(
        next(iter(next(iter(filtered_by_head.values())).get("buckets", {}).values()))
        .get("metrics", {})
        .keys()
    )

    bucket_acc = {bucket: init_acc(metric_names) for bucket in bucket_names}
    for item in filtered_by_head.values():
        for bucket in bucket_names:
            add_finalized_bucket(bucket_acc[bucket], item["buckets"].get(bucket, {}))

    filtered_heads = [
        [int(item["layer"]), int(item["head"])]
        for item in filtered_by_head.values()
    ]
    filtered_heads.sort()

    out = deepcopy(summary)
    out["heads"] = filtered_heads
    out["buckets"] = {bucket: finalize_acc(acc) for bucket, acc in bucket_acc.items()}
    out["by_head"] = dict(
        sorted(
            filtered_by_head.items(),
            key=lambda kv: (int(kv[1]["layer"]), int(kv[1]["head"])),
        )
    )
    out["config"] = dict(summary.get("config", {}))
    out["config"].update(
        {
            "source_summary_file": args.summary_file,
            "layer_filter": found_layers,
            "requested_layer_filter": requested_layers,
            "missing_requested_layers": missing_layers,
            "layer_filter_slug": layer_slug(found_layers),
            "num_filtered_heads": len(filtered_heads),
            "note": "Filtered from an existing txtattn_summary.json; overall bucket stats are recomputed from selected by_head finalized means/vars.",
        }
    )

    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    layers = [layer for layer, _ in filtered_heads]
    print(
        json.dumps(
            {
                "output_file": output_file,
                "num_heads": len(filtered_heads),
                "layers": found_layers,
                "requested_layers": requested_layers,
                "missing_layers": missing_layers,
                "layer_slug": layer_slug(found_layers),
                "layer_counts": {str(layer): layers.count(layer) for layer in sorted(set(layers))},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter txtattn_summary.json to selected layers and recompute overall bucket stats.")
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--layers", default="", help="Comma-separated layer list. Supports ranges like 9:16 or 9-16 inside the list.")
    parser.add_argument("--min-layer", type=int, default=None, help="Backward-compatible inclusive minimum layer.")
    parser.add_argument("--max-layer", type=int, default=None, help="Backward-compatible inclusive maximum layer.")
    parser.add_argument("--allow-missing-layers", action="store_true")
    filter_summary(parser.parse_args())
