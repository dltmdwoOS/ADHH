import argparse
import json
import os
from collections import OrderedDict


def _load_spacy_model():
    try:
        import spacy

        return spacy.load("en_core_web_lg")
    except Exception:
        return None


def _extract_nouns(text):
    try:
        import nltk
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        return [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith("NN")]
    except Exception:
        import re

        return re.findall(r"[A-Za-z][A-Za-z_-]*", text.lower())


def _similar_enough(nlp, word1, word2, threshold):
    if nlp is None:
        return False
    return nlp(word1).similarity(nlp(word2)) > threshold


def _expand_terms(terms, association):
    expanded = set(terms)
    for term in terms:
        expanded.update(association.get(term, []))
    return expanded


def _canonical_mentions(response, association):
    known_words = set(association)
    for values in association.values():
        known_words.update(values)
    return [noun for noun in _extract_nouns(response) if noun in known_words]


def _match_any(nlp, noun, candidates, global_safe_words, similarity_score):
    if noun in global_safe_words:
        return True
    if noun in candidates:
        return True
    for candidate in candidates:
        if _similar_enough(nlp, noun, candidate, similarity_score):
            return True
    return False


def compute_amber_generative_object_metrics(
    response_file,
    amber_root,
    official_metrics=None,
    similarity_score=0.8,
):
    """
    Compute CHAIR-style object precision/recall/F1 for AMBER generative outputs.

    This is not AMBER's official discriminative Acc/Prec/Rec/F1.  It uses the
    generative annotation object sets: predicted nouns matched to truth objects
    are counted as true positives, other AMBER-known object nouns as false
    positives, and uncovered truth objects as false negatives.
    """
    amber_root = os.path.abspath(os.path.expanduser(amber_root))
    with open(os.path.expanduser(response_file), "r", encoding="utf-8") as f:
        responses = json.load(f)
    with open(os.path.join(amber_root, "data", "annotations.json"), "r", encoding="utf-8") as f:
        annotations = json.load(f)
    with open(os.path.join(amber_root, "data", "relation.json"), "r", encoding="utf-8") as f:
        association = json.load(f)
    with open(os.path.join(amber_root, "data", "safe_words.txt"), "r", encoding="utf-8") as f:
        global_safe_words = {line.strip() for line in f if line.strip()}

    nlp = _load_spacy_model()
    response_by_id = {int(row["id"]): row.get("response", "") for row in responses}

    tp_total = 0
    pred_total = 0
    gt_total = 0
    evaluated = 0
    per_sample = []

    for idx, gt in enumerate(annotations, start=1):
        if gt.get("type") != "generative" or idx not in response_by_id:
            continue

        truth_objects = set(gt.get("truth", []))
        truth_terms = _expand_terms(truth_objects, association)
        mentions = _canonical_mentions(response_by_id[idx], association)

        predicted_objects = set()
        correct_objects = set()
        false_objects = set()

        for noun in mentions:
            predicted_objects.add(noun)
            if _match_any(nlp, noun, truth_terms, global_safe_words, similarity_score):
                correct_objects.add(noun)
            else:
                false_objects.add(noun)

        covered_truth = set()
        for truth in truth_objects:
            candidates = _expand_terms({truth}, association)
            for noun in predicted_objects:
                if _match_any(nlp, noun, candidates, global_safe_words, similarity_score):
                    covered_truth.add(truth)
                    break

        tp = len(correct_objects)
        pred = len(predicted_objects)
        gt_count = len(truth_objects)
        tp_total += tp
        pred_total += pred
        gt_total += gt_count
        evaluated += 1

        precision = tp / pred if pred else 0.0
        recall = len(covered_truth) / gt_count if gt_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_sample.append(
            {
                "id": idx,
                "num_pred_objects": pred,
                "num_truth_objects": gt_count,
                "num_correct_pred_objects": tp,
                "num_covered_truth_objects": len(covered_truth),
                "ObjectPrecision": precision,
                "ObjectRecall": recall,
                "ObjectF1": f1,
                "false_objects": sorted(false_objects),
            }
        )

    precision = tp_total / pred_total if pred_total else 0.0
    recall = tp_total / gt_total if gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics = OrderedDict(
        [
            ("ObjectPrecision", precision),
            ("ObjectRecall", recall),
            ("ObjectF1", f1),
        ]
    )
    if official_metrics and "CHAIR" in official_metrics:
        chair = float(official_metrics["CHAIR"]) / 100.0
        metrics["AMBER"] = (1.0 - chair + f1) / 2.0

    return {
        "metrics": metrics,
        "counts": {
            "num_samples": evaluated,
            "tp_total": tp_total,
            "pred_total": pred_total,
            "gt_total": gt_total,
        },
        "config": {
            "definition": "generative object-set metric; not AMBER discriminative Acc/Prec/Rec/F1",
            "similarity_score": similarity_score,
            "semantic_similarity_enabled": nlp is not None,
            "amber_formula": "(1 - CHAIR + ObjectF1) / 2, with CHAIR and ObjectF1 in [0, 1]",
        },
        "per_sample": per_sample,
    }


def augment_metrics_file(metrics_file, response_file=None, amber_root="../third_party/AMBER"):
    with open(os.path.expanduser(metrics_file), "r", encoding="utf-8") as f:
        result = json.load(f, object_pairs_hook=OrderedDict)

    if response_file is None:
        command = result.get("official_command", [])
        for i, token in enumerate(command):
            if token == "--inference_data" and i + 1 < len(command):
                response_file = command[i + 1]
                break
    if response_file is None:
        raise ValueError("Could not infer response_file; pass it explicitly.")

    object_result = compute_amber_generative_object_metrics(
        response_file=response_file,
        amber_root=amber_root,
        official_metrics=result.get("metrics", {}),
    )
    result["metrics"].update(object_result["metrics"])
    result["generative_object_metrics"] = {
        key: value for key, value in object_result.items() if key != "per_sample"
    }

    with open(os.path.expanduser(metrics_file), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_file")
    parser.add_argument("--response-file", type=str, default=None)
    parser.add_argument("--amber-root", type=str, default="../third_party/AMBER")
    args = parser.parse_args()
    updated = augment_metrics_file(args.metrics_file, args.response_file, args.amber_root)
    print(json.dumps(updated["metrics"], indent=2, ensure_ascii=False))
