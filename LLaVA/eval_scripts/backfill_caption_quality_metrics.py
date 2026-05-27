import argparse
import json
import os
from collections import OrderedDict

from pycocotools.coco import COCO
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

JAVA17_SPICE_OPENS = " ".join([
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/java.io=ALL-UNNAMED",
    "--add-opens=java.base/java.nio=ALL-UNNAMED",
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
    "--add-opens=java.base/java.math=ALL-UNNAMED",
    "--add-opens=java.base/java.net=ALL-UNNAMED",
    "--add-opens=java.base/java.text=ALL-UNNAMED",
])

SCORERS = {
    "CIDEr": Cider,
    "SPICE": Spice,
    "METEOR": Meteor,
}
SPICE_PREFLIGHT_ERROR = None
SPICE_PREFLIGHT_DONE = False


def ensure_spice_java_options():
    existing = os.environ.get("JAVA_TOOL_OPTIONS", "")
    if "--add-opens=java.base/java.lang=" not in existing:
        os.environ["JAVA_TOOL_OPTIONS"] = (existing + " " + JAVA17_SPICE_OPENS).strip()


def spice_preflight_error():
    global SPICE_PREFLIGHT_DONE, SPICE_PREFLIGHT_ERROR
    if SPICE_PREFLIGHT_DONE:
        return SPICE_PREFLIGHT_ERROR
    SPICE_PREFLIGHT_DONE = True
    ensure_spice_java_options()
    try:
        scorer = Spice()
        scorer.compute_score({0: ["a dog on grass"]}, {0: ["a dog on grass"]})
    except Exception as exc:
        SPICE_PREFLIGHT_ERROR = f"{type(exc).__name__}: {exc}"
    return SPICE_PREFLIGHT_ERROR


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_answers(path, fallback_eval_file=None):
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                rows.append({
                    "image_id": int(item["question_id"]),
                    "caption": item["text"],
                })
    if rows or not fallback_eval_file:
        return rows

    with open(fallback_eval_file, "r", encoding="utf-8") as f:
        result = json.load(f)
    for item in result.get("sentences", []):
        rows.append({
            "image_id": int(item["image_id"]),
            "caption": item["caption"],
        })
    return rows


def build_eval_inputs(annotation_file, answers):
    coco = COCO(annotation_file)
    refs = {}
    cands = {}
    for idx, item in enumerate(answers):
        ann_ids = coco.getAnnIds(imgIds=item["image_id"])
        anns = coco.loadAnns(ann_ids)
        refs[idx] = [{"caption": ann["caption"]} for ann in anns]
        cands[idx] = [{"caption": item["caption"]}]
    return refs, cands


def insert_metrics_after_bleu(overall, additions):
    ordered = OrderedDict()
    inserted = False
    for key, value in overall.items():
        ordered[key] = value
        if key == "Bleu":
            for metric_name, metric_value in additions.items():
                ordered[metric_name] = metric_value
            inserted = True
    if not inserted:
        for metric_name, metric_value in additions.items():
            ordered[metric_name] = metric_value
    return dict(ordered)


def evaluate(annotation_file, answers_file, metrics, fallback_eval_file=None):
    answers = load_answers(answers_file, fallback_eval_file=fallback_eval_file)
    if not answers:
        raise ValueError(f"No captions found in {answers_file} or fallback eval file")
    refs, cands = build_eval_inputs(annotation_file, answers)
    tokenizer = PTBTokenizer()
    refs_tok = tokenizer.tokenize(refs)
    cands_tok = tokenizer.tokenize(cands)

    scores = OrderedDict()
    errors = OrderedDict()
    for metric in metrics:
        try:
            if metric == "SPICE":
                preflight_error = spice_preflight_error()
                if preflight_error:
                    errors[metric] = preflight_error
                    continue
            scorer = SCORERS[metric]()
            score, _ = scorer.compute_score(refs_tok, cands_tok)
            scores[metric] = json_safe(score)
        except Exception as exc:
            errors[metric] = f"{type(exc).__name__}: {exc}"
    return scores, errors, len(answers)


def backfill(result_dir, annotation_file, answers_name, metrics):
    answers_file = os.path.join(result_dir, answers_name)
    eval_file = os.path.join(result_dir, "captions_eval_results.json")
    if not os.path.exists(result_dir):
        raise FileNotFoundError(result_dir)
    if not os.path.exists(answers_file):
        raise FileNotFoundError(answers_file)
    if not os.path.exists(eval_file):
        raise FileNotFoundError(eval_file)

    scores, errors, num_samples = evaluate(annotation_file, answers_file, metrics, fallback_eval_file=eval_file)
    with open(eval_file, "r", encoding="utf-8") as f:
        result = json.load(f, object_pairs_hook=OrderedDict)

    overall = result.setdefault("overall_metrics", OrderedDict())
    result["overall_metrics"] = insert_metrics_after_bleu(overall, scores)
    if errors:
        result["quality_metric_errors"] = errors
    elif "quality_metric_errors" in result:
        for metric in metrics:
            result["quality_metric_errors"].pop(metric, None)
        if not result["quality_metric_errors"]:
            result.pop("quality_metric_errors")

    result["quality_metric_config"] = {
        "annotation_file": annotation_file,
        "answers_file": answers_file,
        "metrics_requested": metrics,
        "num_samples": num_samples,
    }

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return scores, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--answers-name", default="captions.jsonl")
    parser.add_argument("--metrics", nargs="+", default=["CIDEr", "METEOR"], choices=sorted(SCORERS))
    parser.add_argument("result_dirs", nargs="+")
    args = parser.parse_args()

    for result_dir in args.result_dirs:
        print(f"[backfill] {result_dir}")
        try:
            scores, errors = backfill(result_dir, args.annotation_file, args.answers_name, args.metrics)
            print("  scores:", json.dumps(scores, ensure_ascii=False))
            if errors:
                print("  errors:", json.dumps(errors, ensure_ascii=False))
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            raise
