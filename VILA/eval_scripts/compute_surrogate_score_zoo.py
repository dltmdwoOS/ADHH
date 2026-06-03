#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from collections import OrderedDict


BUCKETS = ("all", "object", "hallucinated", "non_hallucinated")
METRICS = ("I_text", "generated_txt_attn", "image_attn", "txt_img_ratio")


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


def var_metric(item, bucket, metric):
    return max(finite(item["buckets"].get(bucket, {}).get("metrics", {}).get(metric, {}).get("var", 0.0)), 0.0)


def frac(text, image, eps):
    return min(max(text / (text + image + eps), 0.0), 1.0)


def rel_delta(pos, neg, eps):
    delta = max(0.0, pos - neg)
    return delta / (pos + neg + eps)


def safe_ratio(num, den, eps):
    return num / (den + eps)


def rank_percentiles(values, reverse=True):
    ordered = sorted(values, key=values.get, reverse=reverse)
    n = len(ordered)
    if n <= 1:
        return {hid: 1.0 for hid in ordered}
    return {hid: 1.0 - i / (n - 1) for i, hid in enumerate(ordered)}


def normalize_minmax(values):
    vals = list(values.values())
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if abs(span) < 1e-12:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def write_ranked(path, score_name, description, layer_range, records, score_key):
    ranked = sorted(records, key=lambda x: (x[score_key], x.get("tie_break", 0.0)), reverse=True)
    by_layer = {}
    for rank, item in enumerate(ranked, start=1):
        item["global_rank"] = rank
        by_layer.setdefault(item["layer"], []).append(item)
    for layer_items in by_layer.values():
        layer_items.sort(key=lambda x: (x[score_key], x.get("tie_break", 0.0)), reverse=True)
        for rank, item in enumerate(layer_items, start=1):
            item["layer_rank"] = rank
    for item in ranked:
        item.pop("tie_break", None)
    obj = {
        "score_name": score_name,
        "description": description,
        "layer_range": layer_range,
        "heads": ranked,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return ranked


def build_head_features(summary, eps):
    features = OrderedDict()
    for key, item in summary["by_head"].items():
        layer = int(item["layer"])
        head = int(item["head"])
        hid = head_id(layer, head)
        f = {
            "layer": layer,
            "head": head,
            "head_id": hid,
        }
        for bucket in BUCKETS:
            t = mean_metric(item, bucket, "generated_txt_attn")
            itext = mean_metric(item, bucket, "I_text")
            img = mean_metric(item, bucket, "image_attn")
            ratio_mean = mean_metric(item, bucket, "txt_img_ratio")
            f[f"T_{bucket}"] = t
            f[f"Itext_{bucket}"] = itext
            f[f"Img_{bucket}"] = img
            f[f"R_{bucket}"] = frac(t, img, eps)
            f[f"LogTOI_{bucket}"] = math.log1p(max(safe_ratio(t, img, eps), 0.0))
            f[f"RawTOI_{bucket}"] = ratio_mean
            f[f"T_std_{bucket}"] = math.sqrt(var_metric(item, bucket, "generated_txt_attn"))
            f[f"Img_std_{bucket}"] = math.sqrt(var_metric(item, bucket, "image_attn"))
        # Core contrasts.
        for metric in ("T", "Itext", "R", "LogTOI"):
            h = f[f"{metric}_hallucinated"]
            g = f[f"{metric}_non_hallucinated"]
            a = f[f"{metric}_all"]
            o = f[f"{metric}_object"]
            f[f"Delta_{metric}_HminusG"] = h - g
            f[f"PosDelta_{metric}_HminusG"] = max(0.0, h - g)
            f[f"RelDelta_{metric}_HminusG"] = rel_delta(h, g, eps)
            f[f"Delta_{metric}_HminusAll"] = h - a
            f[f"Delta_{metric}_ObjminusAll"] = o - a
        f["ImgDrop_HminusG"] = max(0.0, f["Img_non_hallucinated"] - f["Img_hallucinated"])
        f["ImgDropRel_HminusG"] = f["ImgDrop_HminusG"] / (f["Img_non_hallucinated"] + f["Img_hallucinated"] + eps)
        features[hid] = f
    return features


def compute_score_maps(features, eps):
    ids = list(features)
    raw = OrderedDict()

    def add(name, values, desc):
        raw[name] = (values, desc)

    add("text_all", {hid: features[hid]["T_all"] for hid in ids}, "Mean generated-text-prefix attention over all generated steps.")
    add("itext_all", {hid: features[hid]["Itext_all"] for hid in ids}, "Mean intervention text-slice attention over all generated steps.")
    add("text_hall", {hid: features[hid]["T_hallucinated"] for hid in ids}, "Mean generated-text-prefix attention on hallucinated object steps.")
    add("text_object", {hid: features[hid]["T_object"] for hid in ids}, "Mean generated-text-prefix attention on object steps.")
    add("ratio_all", {hid: features[hid]["R_all"] for hid in ids}, "Mean-level T/(T+I) over all generated steps.")
    add("ratio_hall", {hid: features[hid]["R_hallucinated"] for hid in ids}, "Mean-level T/(T+I) on hallucinated object steps.")
    add("logtoi_hall", {hid: features[hid]["LogTOI_hallucinated"] for hid in ids}, "Log text-over-image ratio on hallucinated object steps, from mean masses.")
    add("low_image_hall", {hid: 1.0 - features[hid]["Img_hallucinated"] for hid in ids}, "Heads that pay little image attention on hallucinated object steps.")

    add("C_text_hall_minus_nonhall", {hid: features[hid]["PosDelta_T_HminusG"] for hid in ids}, "Positive hallucinated minus grounded object contrast in text-prefix attention.")
    add("C_ratio_hall_minus_nonhall", {hid: features[hid]["PosDelta_R_HminusG"] for hid in ids}, "Positive hallucinated minus grounded object contrast in T/(T+I).")
    add("C_logtoi_hall_minus_nonhall", {hid: features[hid]["PosDelta_LogTOI_HminusG"] for hid in ids}, "Positive hallucinated minus grounded object contrast in log text-over-image.")
    add("C_itext_hall_minus_nonhall", {hid: features[hid]["PosDelta_Itext_HminusG"] for hid in ids}, "Positive hallucinated minus grounded object contrast in intervention text-slice mass.")
    add("C_image_drop_nonhall_minus_hall", {hid: features[hid]["ImgDrop_HminusG"] for hid in ids}, "Image-attention drop from grounded to hallucinated object steps.")

    add("RelC_text_hall_minus_nonhall", {hid: features[hid]["RelDelta_T_HminusG"] for hid in ids}, "Relative contrast in text-prefix attention: delta/(hall+grounded).")
    add("RelC_ratio_hall_minus_nonhall", {hid: features[hid]["RelDelta_R_HminusG"] for hid in ids}, "Relative contrast in T/(T+I): delta/(hall+grounded).")
    add("RelC_logtoi_hall_minus_nonhall", {hid: features[hid]["RelDelta_LogTOI_HminusG"] for hid in ids}, "Relative contrast in log text-over-image.")
    add("RelC_image_drop", {hid: features[hid]["ImgDropRel_HminusG"] for hid in ids}, "Relative image-attention drop on hallucinated object steps.")

    add("Specific_text_delta_x_hall", {hid: features[hid]["PosDelta_T_HminusG"] * features[hid]["T_hallucinated"] for hid in ids}, "Absolute text contrast gated by hallucinated-step text strength.")
    add("Specific_ratio_delta_x_hall", {hid: features[hid]["PosDelta_R_HminusG"] * features[hid]["R_hallucinated"] for hid in ids}, "Absolute ratio contrast gated by hallucinated-step ratio strength.")
    add("Specific_logtoi_delta_x_hall", {hid: features[hid]["PosDelta_LogTOI_HminusG"] * features[hid]["LogTOI_hallucinated"] for hid in ids}, "Absolute log text-over-image contrast gated by hallucinated-step strength.")
    add("Specific_text_threefactor", {hid: features[hid]["PosDelta_T_HminusG"] * features[hid]["RelDelta_T_HminusG"] * features[hid]["T_hallucinated"] for hid in ids}, "Text mass three-factor score: delta * relative contrast * hall strength.")
    add("Specific_ratio_threefactor", {hid: features[hid]["PosDelta_R_HminusG"] * features[hid]["RelDelta_R_HminusG"] * features[hid]["R_hallucinated"] for hid in ids}, "Ratio three-factor score: delta * relative contrast * hall strength.")

    add("Object_text_minus_all", {hid: max(0.0, features[hid]["Delta_T_ObjminusAll"]) for hid in ids}, "Object-token text-prefix excess over all generated steps.")
    add("Hall_text_minus_all", {hid: max(0.0, features[hid]["Delta_T_HminusAll"]) for hid in ids}, "Hallucinated-token text-prefix excess over all generated steps.")
    add("Object_ratio_minus_all", {hid: max(0.0, features[hid]["Delta_R_ObjminusAll"]) for hid in ids}, "Object-token T/(T+I) excess over all generated steps.")
    add("Hall_ratio_minus_all", {hid: max(0.0, features[hid]["Delta_R_HminusAll"]) for hid in ids}, "Hallucinated-token T/(T+I) excess over all generated steps.")

    # Rank-fusion families. These are fixed hypothesis families, not fitted to any verified HH list.
    pct = {name: rank_percentiles(vals) for name, (vals, _) in raw.items()}
    add("Fusion_text70_ratioC30", {hid: 0.70 * pct["text_all"][hid] + 0.30 * pct["C_ratio_hall_minus_nonhall"][hid] for hid in ids}, "Rank fusion: 70% global text-prefix reliance, 30% hallucination contrast in T/(T+I).")
    add("Fusion_text50_ratioC30_imgdrop20", {hid: 0.50 * pct["text_all"][hid] + 0.30 * pct["C_ratio_hall_minus_nonhall"][hid] + 0.20 * pct["C_image_drop_nonhall_minus_hall"][hid] for hid in ids}, "Rank fusion: text reliance + ratio contrast + image-attention drop.")
    add("Fusion_halltext50_logC30_imgdrop20", {hid: 0.50 * pct["text_hall"][hid] + 0.30 * pct["C_logtoi_hall_minus_nonhall"][hid] + 0.20 * pct["C_image_drop_nonhall_minus_hall"][hid] for hid in ids}, "Rank fusion: hallucinated-step text reliance + log text-over-image contrast + image drop.")
    add("Fusion_text40_absC30_relC30", {hid: 0.40 * pct["text_all"][hid] + 0.30 * pct["C_text_hall_minus_nonhall"][hid] + 0.30 * pct["RelC_text_hall_minus_nonhall"][hid] for hid in ids}, "Rank fusion balancing global text reliance, absolute text contrast, and relative text contrast.")
    add("Fusion_ratio_policy", {hid: 0.40 * pct["ratio_hall"][hid] + 0.30 * pct["C_ratio_hall_minus_nonhall"][hid] + 0.30 * pct["RelC_ratio_hall_minus_nonhall"][hid] for hid in ids}, "Policy-style ratio fusion using hall ratio, absolute ratio contrast, and relative ratio contrast.")

    return raw


def main(args):
    with open(os.path.expanduser(args.summary_file), "r", encoding="utf-8") as f:
        summary = json.load(f)
    features = build_head_features(summary, args.eps)
    layer_values = [f["layer"] for f in features.values()]
    layer_range = [min(layer_values), max(layer_values)] if layer_values else None
    score_maps = compute_score_maps(features, args.eps)

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    overview_rows = []
    for score_name, (values, desc) in score_maps.items():
        norm_values = normalize_minmax(values)
        records = []
        for hid, f in features.items():
            records.append({
                "layer": f["layer"],
                "head": f["head"],
                "head_id": hid,
                score_name: norm_values[hid],
                f"{score_name}_raw": values[hid],
                "T_all": f["T_all"],
                "T_hallucinated": f["T_hallucinated"],
                "T_non_hallucinated": f["T_non_hallucinated"],
                "R_hallucinated": f["R_hallucinated"],
                "R_non_hallucinated": f["R_non_hallucinated"],
                "Img_hallucinated": f["Img_hallucinated"],
                "Img_non_hallucinated": f["Img_non_hallucinated"],
                "tie_break": f["T_all"],
            })
        path = os.path.join(out_dir, f"ranked_heads_{score_name}.json")
        ranked = write_ranked(path, score_name, desc, layer_range, records, score_name)
        overview_rows.append({
            "score_name": score_name,
            "description": desc,
            "path": path,
            "top1": ranked[0]["head_id"] if ranked else "",
            "top20": " ".join(item["head_id"] for item in ranked[:20]),
        })

    overview_path = os.path.join(out_dir, "surrogate_score_zoo_overview.csv")
    with open(overview_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["score_name", "description", "path", "top1", "top20"])
        writer.writeheader()
        writer.writerows(overview_rows)

    config = {
        "summary_file": args.summary_file,
        "output_dir": out_dir,
        "num_heads": len(features),
        "num_scores": len(score_maps),
        "note": "These scores are hypothesis-driven surrogate families generated without fitting to any verified top-k HH list.",
    }
    with open(os.path.join(out_dir, "surrogate_score_zoo_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"overview -> {overview_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a diverse zoo of hallucination-head surrogate scores from txtattn_summary.json.")
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eps", type=float, default=1e-12)
    main(parser.parse_args())
