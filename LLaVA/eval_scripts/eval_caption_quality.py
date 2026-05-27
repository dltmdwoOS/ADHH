import argparse
import json
import os

try:
    from pycocotools.coco import COCO
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.spice.spice import Spice
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "eval_caption_quality.py requires pycocotools and pycocoevalcap. "
        "Use the same environment as eval_scripts/eval_utils/eval_bleu.py, "
        "or install those packages before running CIDEr/SPICE/METEOR evaluation."
    ) from exc


SCORERS = {
    "CIDEr": Cider,
    "METEOR": Meteor,
    "SPICE": Spice,
}


def json_safe_score(value):
    if isinstance(value, dict):
        return {k: json_safe_score(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_score(v) for v in value]
    return float(value)


def load_answers(path):
    answers = []
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                answers.append({
                    "image_id": int(item["question_id"]),
                    "caption": item["text"],
                    "raw": item,
                })
    return answers


def build_coco_references(annotation_file, image_ids):
    coco = COCO(os.path.expanduser(annotation_file))
    refs = {}
    for idx, image_id in enumerate(image_ids):
        ann_ids = coco.getAnnIds(imgIds=int(image_id))
        anns = coco.loadAnns(ann_ids)
        refs[idx] = [{"caption": ann["caption"]} for ann in anns]
    return refs


def evaluate_quality(annotation_file, answers_file, metrics):
    answers = load_answers(answers_file)
    image_ids = [x["image_id"] for x in answers]
    refs = build_coco_references(annotation_file, image_ids)
    cands = {
        idx: [{"caption": item["caption"]}]
        for idx, item in enumerate(answers)
    }

    tokenizer = PTBTokenizer()
    refs_tok = tokenizer.tokenize(refs)
    cands_tok = tokenizer.tokenize(cands)

    overall = {}
    per_image = []
    for metric in metrics:
        scorer = SCORERS[metric]()
        score, scores = scorer.compute_score(refs_tok, cands_tok)
        overall[metric] = json_safe_score(score)
        for idx, image_score in enumerate(scores):
            if len(per_image) <= idx:
                per_image.append({
                    "image_id": image_ids[idx],
                    "caption": answers[idx]["caption"],
                    "metrics": {},
                })
            per_image[idx]["metrics"][metric] = json_safe_score(image_score)

    return {
        "overall_metrics": overall,
        "per_image": per_image,
        "config": {
            "annotation_file": annotation_file,
            "answers_file": answers_file,
            "metrics": metrics,
            "num_samples": len(answers),
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
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["CIDEr", "SPICE", "METEOR"],
        choices=sorted(SCORERS),
    )
    args = parser.parse_args()

    output_file = args.output_file
    if not output_file:
        output_file = args.answers_file.replace(".jsonl", "_quality_metrics.json")
        if output_file == args.answers_file:
            output_file = args.answers_file + ".quality_metrics.json"

    result = evaluate_quality(args.annotation_file, args.answers_file, args.metrics)
    write_json(output_file, result)
    print(json.dumps(result["overall_metrics"], indent=2))
