#!/usr/bin/env python3
import argparse
import json
import os


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def row(name, bucket):
    quantiles = bucket.get("suppression_quantiles", {}) or {}
    return {
        "name": name,
        "count": bucket.get("count", 0),
        "scaled_rate": bucket.get("scaled_rate"),
        "near_zero_rate": bucket.get("near_zero_rate"),
        "saturation_rate": bucket.get("saturation_rate"),
        "mean_delta": bucket.get("mean_suppression"),
        "q50_delta": quantiles.get("q50"),
        "q75_delta": quantiles.get("q75"),
        "q90_delta": quantiles.get("q90"),
        "q95_delta": quantiles.get("q95"),
        "mean_row_sum_after": bucket.get("mean_row_sum_after"),
    }


def print_table(title, rows, limit=None):
    if limit:
        rows = rows[:limit]
    if not rows:
        return
    fields = list(rows[0].keys())
    print(f"\n## {title}")
    print("| " + " | ".join(fields) + " |")
    print("| " + " | ".join(["---"] * len(fields)) + " |")
    for item in rows:
        print("| " + " | ".join(fmt(item.get(field)) for field in fields) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_file")
    parser.add_argument("--position-limit", type=int, default=20)
    args = parser.parse_args()

    with open(os.path.expanduser(args.stats_file), "r", encoding="utf-8") as f:
        stats = json.load(f)

    overall_rows = [
        row(mode, bucket)
        for mode, bucket in sorted((stats.get("overall") or {}).items())
    ]
    print_table("Overall", overall_rows)

    for mode, by_label in sorted((stats.get("by_token_bucket") or {}).items()):
        rows = [row(label, bucket) for label, bucket in sorted(by_label.items())]
        print_table(f"Token Buckets: {mode}", rows)

    for mode, by_position in sorted((stats.get("by_position") or {}).items()):
        rows = [
            row(str(position), bucket)
            for position, bucket in sorted(by_position.items(), key=lambda kv: int(kv[0]))
        ]
        print_table(f"Positions: {mode}", rows, limit=args.position_limit)


if __name__ == "__main__":
    main()
