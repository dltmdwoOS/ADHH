#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import OrderedDict

BUCKETS = ("all", "object", "hallucinated", "non_hallucinated")


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


def global_order(records, score_name):
    return sorted(
        records,
        key=lambda x: (x[score_name], x.get("front_percentile", 0.0), x.get("back_percentile", 0.0)),
        reverse=True,
    )


def annotate_ranks(ranked, score_field):
    by_layer = {}
    for rank, item in enumerate(ranked, start=1):
        item["global_rank"] = rank
        item["selection_method"] = "global"
        by_layer.setdefault(int(item["layer"]), []).append(item)
    for items in by_layer.values():
        sorted_items = sorted(
            items,
            key=lambda x: (x[score_field], x.get("front_percentile", 0.0), x.get("back_percentile", 0.0)),
            reverse=True,
        )
        for local_rank, item in enumerate(sorted_items, start=1):
            item["layer_rank"] = local_rank


def write_ranked(path, score_name, description, layer_range, ranked):
    annotate_ranks(ranked, score_name)
    obj = {
        "score_name": score_name,
        "description": description,
        "layer_range": layer_range,
        "selection_method": "global",
        "layer_order": [],
        "heads": ranked,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_records(features, score_name, score_values, front_values, back_values, front_pct, back_pct, back_name):
    records = []
    for hid, f in features.items():
        signed_toi = f["RawTOI_hallucinated"] - f["RawTOI_non_hallucinated"]
        signed_itext = f["Itext_hallucinated"] - f["Itext_non_hallucinated"]
        rel_itext = signed_itext / (f["Itext_all"] + 1e-12)
        text_minus_img_h = f["Itext_hallucinated"] - f["Img_hallucinated"]
        text_minus_img_g = f["Itext_non_hallucinated"] - f["Img_non_hallucinated"]
        records.append({
            "layer": f["layer"],
            "head": f["head"],
            "head_id": hid,
            score_name: score_values[hid],
            "score": score_values[hid],
            "front_name": "itext_all",
            "front_raw": front_values[hid],
            "front_percentile": front_pct[hid],
            "back_name": back_name,
            "back_raw": back_values[hid],
            "back_percentile": back_pct[hid],
            "signed_back_raw": back_values[hid],
            "signed_toi_back_raw": signed_toi,
            "signed_itext_back_raw": signed_itext,
            "rel_itext_back_raw": rel_itext,
            "textminusimg_back_raw": text_minus_img_h - text_minus_img_g,
            "clipped_back_raw": max(0.0, back_values[hid]),
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
    return global_order(records, score_name)


def main(args):
    with open(os.path.expanduser(args.summary_file), "r", encoding="utf-8") as f:
        summary = json.load(f)
    features = build_features(summary, args.eps)
    layer_values = [f["layer"] for f in features.values()]
    layer_range = [min(layer_values), max(layer_values)] if layer_values else None

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Keep this directory compact: future runs emit only the TOI rankings used
    # by the paper-facing dynamic intervention and control probes.
    for name in os.listdir(out_dir):
        if name.startswith("ranked_heads_") and name.endswith(".json"):
            os.remove(os.path.join(out_dir, name))
        elif name in {
            "surrogate_combo_overlap_summary.csv",
            "surrogate_combo_config.json",
            "surrogate_score_zoo_overview.csv",
            "surrogate_score_zoo_config.json",
        }:
            os.remove(os.path.join(out_dir, name))

    front_values = {hid: f["Itext_all"] for hid, f in features.items()}
    toi_hminusg = {
        hid: f["RawTOI_hallucinated"] - f["RawTOI_non_hallucinated"]
        for hid, f in features.items()
    }
    clipped_toi_hminusg = {hid: max(0.0, value) for hid, value in toi_hminusg.items()}
    front_pct = rank_percentiles(front_values)
    clipped_toi_pct = rank_percentiles(clipped_toi_hminusg)
    signed_toi_pct = rank_percentiles(toi_hminusg)
    def mean_combo(back_pct):
        return {hid: 0.5 * front_pct[hid] + 0.5 * back_pct[hid] for hid in features}

    score_specs = [
        {
            "score_name": "global__itext_all__C_toi_HminusG",
            "back_name": "C_toi_HminusG",
            "back_values": clipped_toi_hminusg,
            "back_pct": clipped_toi_pct,
            "score_values": mean_combo(clipped_toi_pct),
            "description": "Global rank-percentile combo: 0.5*P(Itext_all) + 0.5*P(max(0, RawTOI_hallucinated - RawTOI_non_hallucinated)).",
        },
        {
            "score_name": "global__itext_all__C_toi_HminusG_signed",
            "back_name": "C_toi_HminusG_signed",
            "back_values": toi_hminusg,
            "back_pct": signed_toi_pct,
            "score_values": mean_combo(signed_toi_pct),
            "description": "Global signed rank-percentile combo: 0.5*P(Itext_all) + 0.5*P(RawTOI_hallucinated - RawTOI_non_hallucinated), without clipping negative contrast to zero.",
        },
    ]

    overview_rows = []
    for spec in score_specs:
        ranked = make_records(
            features=features,
            score_name=spec["score_name"],
            score_values=spec["score_values"],
            front_values=front_values,
            back_values=spec["back_values"],
            front_pct=front_pct,
            back_pct=spec["back_pct"],
            back_name=spec["back_name"],
        )
        out_path = os.path.join(out_dir, f"ranked_heads_{spec['score_name']}.json")
        write_ranked(out_path, spec["score_name"], spec["description"], layer_range, ranked)
        overview_rows.append({
            "score_name": spec["score_name"],
            "description": spec["description"],
            "path": out_path,
            "top1": ranked[0]["head_id"] if ranked else "",
            "top20": " ".join(item["head_id"] for item in ranked[:20]),
            "top100_negative_back_raw": sum(1 for item in ranked[:100] if item.get("back_raw", 0.0) < 0),
        })

    overview_path = os.path.join(out_dir, "surrogate_score_zoo_overview.csv")
    fields = ["score_name", "description", "path", "top1", "top20", "top100_negative_back_raw"]
    with open(overview_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(overview_rows)

    config = {
        "summary_file": args.summary_file,
        "output_dir": out_dir,
        "num_heads": len(features),
        "num_scores": len(score_specs),
        "scores": [spec["score_name"] for spec in score_specs],
        "note": "This compact zoo emits only the TOI rank-fused rankings used by the paper-facing head-pool controls and dynamic intervention.",
    }
    with open(os.path.join(out_dir, "surrogate_score_zoo_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"overview -> {overview_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate compact surrogate head rankings from txtattn_summary.json.")
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eps", type=float, default=1e-12)
    main(parser.parse_args())
