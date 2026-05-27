import argparse
import json
import os
from collections import OrderedDict


def canonical_generated_objects(sentence):
    objects = set()
    for key in ("mscoco_non_hallucinated_words", "mscoco_hallucinated_words"):
        for item in sentence.get(key, []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                objects.add(item[1])
    return objects


def compute_sentence_metrics(sentence):
    gt_objects = set(sentence.get("mscoco_gt_words", []))
    generated_objects = canonical_generated_objects(sentence)
    correct_objects = generated_objects.intersection(gt_objects)

    precision = len(correct_objects) / len(generated_objects) if generated_objects else 0.0
    recall = len(correct_objects) / len(gt_objects) if gt_objects else 0.0
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, len(correct_objects), len(generated_objects), len(gt_objects)


def insert_after_chairi(overall, additions):
    ordered = OrderedDict()
    inserted = False
    for key, value in overall.items():
        ordered[key] = value
        if key == "CHAIRi":
            for metric_name, metric_value in additions.items():
                ordered[metric_name] = metric_value
            inserted = True
    if not inserted:
        for metric_name, metric_value in additions.items():
            ordered[metric_name] = metric_value
    return dict(ordered)


def backfill(path):
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f, object_pairs_hook=OrderedDict)

    tp_total = 0.0
    pred_total = 0.0
    gt_total = 0.0
    for sentence in result.get("sentences", []):
        precision, recall, f1, tp, pred, gt = compute_sentence_metrics(sentence)
        metrics = sentence.setdefault("metrics", OrderedDict())
        metrics["ObjectPrecision"] = precision
        metrics["ObjectRecall"] = recall
        metrics["ObjectF1"] = f1
        tp_total += tp
        pred_total += pred
        gt_total += gt

    overall_precision = tp_total / pred_total if pred_total > 0 else 0.0
    overall_recall = tp_total / gt_total if gt_total > 0 else 0.0
    overall_f1 = 0.0
    if overall_precision + overall_recall > 0:
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)

    additions = OrderedDict([
        ("ObjectPrecision", overall_precision),
        ("ObjectRecall", overall_recall),
        ("ObjectF1", overall_f1),
    ])
    overall = result.setdefault("overall_metrics", OrderedDict())
    result["overall_metrics"] = insert_after_chairi(overall, additions)
    result["object_f1_config"] = {
        "definition": "unique canonical MSCOCO objects; precision=|generated∩GT|/|generated|, recall=|generated∩GT|/|GT|, F1=harmonic mean",
        "num_samples": len(result.get("sentences", [])),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return additions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        if os.path.isdir(path):
            path = os.path.join(path, "captions_eval_results.json")
        print(f"[object-f1] {path}")
        scores = backfill(path)
        print("  scores:", json.dumps(scores, ensure_ascii=False))
