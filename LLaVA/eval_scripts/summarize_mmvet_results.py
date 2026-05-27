import argparse
import csv
import json
import os
from collections import defaultdict


def load_json(path):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        return json.load(f)


def load_payload(path):
    rows = load_json(path)
    return {str(row["question_id"]): row for row in rows}


def load_scores(path):
    raw = load_json(path)
    scores = {}
    for qid, item in raw.items():
        vals = item.get("score", [])
        if vals:
            scores[str(qid)] = sum(float(x) for x in vals) / len(vals)
    return scores


def mean(values):
    return sum(values) / len(values) if values else 0.0


def add_bucket(buckets, name, qid, score, base_score=None):
    bucket = buckets[name]
    bucket["count"] += 1
    bucket["scores"].append(score)
    if base_score is not None:
        bucket["deltas"].append(score - base_score)


def summarize(payload, scores, baseline_scores=None):
    buckets = {
        "overall": defaultdict(lambda: {"count": 0, "scores": [], "deltas": []}),
        "by_capability": defaultdict(lambda: {"count": 0, "scores": [], "deltas": []}),
        "by_capability_combo": defaultdict(lambda: {"count": 0, "scores": [], "deltas": []}),
        "by_answer_operator": defaultdict(lambda: {"count": 0, "scores": [], "deltas": []}),
        "by_imagesource": defaultdict(lambda: {"count": 0, "scores": [], "deltas": []}),
    }

    for qid, score in scores.items():
        meta = payload.get(str(qid), {})
        base_score = baseline_scores.get(str(qid)) if baseline_scores else None
        add_bucket(buckets["overall"], "total", qid, score, base_score)

        caps = meta.get("capability") or ["unknown"]
        for cap in caps:
            add_bucket(buckets["by_capability"], cap, qid, score, base_score)

        combo = meta.get("capability_combo") or "+".join(sorted(caps)) or "unknown"
        add_bucket(buckets["by_capability_combo"], combo, qid, score, base_score)
        add_bucket(buckets["by_answer_operator"], meta.get("answer_operator", "unknown"), qid, score, base_score)
        add_bucket(buckets["by_imagesource"], meta.get("imagesource", "unknown") or "unknown", qid, score, base_score)

    result = {}
    for section, section_buckets in buckets.items():
        result[section] = {}
        for name, bucket in sorted(section_buckets.items()):
            entry = {
                "count": bucket["count"],
                "score": mean(bucket["scores"]) * 100,
            }
            if bucket["deltas"]:
                entry["delta_vs_baseline"] = mean(bucket["deltas"]) * 100
            result[section][name] = entry
    return result


def write_csv(summary, path):
    rows = []
    for section, buckets in summary.items():
        for name, metrics in buckets.items():
            rows.append({
                "section": section,
                "bucket": name,
                "count": metrics["count"],
                "score": metrics["score"],
                "delta_vs_baseline": metrics.get("delta_vs_baseline", ""),
            })
    with open(os.path.expanduser(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "bucket", "count", "score", "delta_vs_baseline"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-file", required=True, help="mmvet_eval_payload.json from the evaluated run")
    parser.add_argument("--grade-file", required=True, help="official MM-Vet *-grade-*runs.json")
    parser.add_argument("--baseline-grade-file", default="", help="optional baseline official grade JSON for deltas")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    payload = load_payload(args.payload_file)
    scores = load_scores(args.grade_file)
    baseline_scores = load_scores(args.baseline_grade_file) if args.baseline_grade_file else None
    summary = summarize(payload, scores, baseline_scores=baseline_scores)

    with open(os.path.expanduser(args.output_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if args.output_csv:
        write_csv(summary, args.output_csv)


if __name__ == "__main__":
    main()
