#!/usr/bin/env python3
"""Build object-token probe records from CHAIR caption evaluation output.

The output is JSONL. Each record stores the generated caption prefix immediately
before a COCO object occurrence and the object surface token to probe. This is
used by probe_head_pool_actuation.py to measure how much a head-pool hard
text-side suppression changes the target object's log-probability.
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import random
from collections import defaultdict

import nltk

from eval_scripts.eval_utils.chair import CHAIR


def load_eval(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "sentences" not in data:
        raise ValueError(f"{path} does not look like a CHAIR captions_eval_results.json")
    return data["sentences"]


def normalize_pair(item):
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return str(item[0]), str(item[1])
    return str(item), str(item)


def _find_token_span(text_lower, token, start_pos):
    # nltk.word_tokenize represents double quotes as `` / ''. Map those back
    # to the original quote character for char-span reconstruction.
    surface = {'``': '"', "''": '"'}.get(token, token)
    idx = text_lower.find(surface, start_pos)
    if idx < 0:
        return None
    return idx, idx + len(surface)


def align_word_token_spans(caption, lower_tokens):
    text_lower = caption.lower()
    spans = []
    cursor = 0
    for token in lower_tokens:
        span = _find_token_span(text_lower, token, cursor)
        if span is None:
            return None
        spans.append(span)
        cursor = span[1]
    return spans


def surface_prefix_target(caption, lower_tokens, raw_idx, target_text):
    spans = align_word_token_spans(caption, lower_tokens)
    if spans is None or raw_idx < 0 or raw_idx >= len(spans):
        return None, None, False

    target_len = max(1, len(str(target_text).split()))
    end_idx = min(raw_idx + target_len - 1, len(spans) - 1)
    start_char = spans[raw_idx][0]
    end_char = spans[end_idx][1]
    return caption[:start_char], caption[start_char:end_char], True


def build_records(sentences, coco_path, max_per_bucket, seed, question):
    rng = random.Random(seed)
    imids = [int(x["image_id"]) for x in sentences]
    chair = CHAIR(imids, coco_path)

    by_bucket = defaultdict(list)
    seen = set()
    for sent in sentences:
        caption = sent.get("caption", "")
        if not caption:
            continue
        try:
            # CHAIR returns idxs in the original nltk-tokenized caption space, but
            # its fourth return value is the post-merged object list. Use fresh
            # raw_tokens for prefix/target reconstruction to avoid index drift after
            # double-word merges such as "sports ball" or "traffic light".
            words, node_words, idxs, _ = chair.caption_to_words(caption)
            raw_tokens = nltk.word_tokenize(caption.lower())
        except Exception as exc:
            print(f"[warn] caption_to_words failed for image_id={sent.get('image_id')}: {exc}")
            continue

        buckets = [
            ("hallucinated", sent.get("mscoco_hallucinated_words", []), sent.get("hallucination_idxs", [])),
            ("grounded", sent.get("mscoco_non_hallucinated_words", []), sent.get("non_hallucination_idxs", [])),
        ]
        for bucket, pairs, occurrence_idxs in buckets:
            for pair, raw_idx in zip(pairs, occurrence_idxs):
                raw_idx = int(raw_idx)
                if raw_idx < 0 or raw_idx >= len(raw_tokens):
                    continue
                word, node_word = normalize_pair(pair)
                object_tokens = word.split()
                if len(object_tokens) > 1 and raw_idx + len(object_tokens) <= len(raw_tokens):
                    candidate = " ".join(raw_tokens[raw_idx:raw_idx + len(object_tokens)])
                    target_text = candidate if candidate == word else raw_tokens[raw_idx]
                else:
                    target_text = raw_tokens[raw_idx]
                target_text = target_text.strip()
                if not target_text or all(ch in ".,;:!?" for ch in target_text):
                    continue

                normalized_prefix_text = " ".join(raw_tokens[:raw_idx]).strip()
                normalized_target_text = target_text
                prefix_text, surface_target_text, span_ok = surface_prefix_target(
                    caption, raw_tokens, raw_idx, target_text
                )
                if not span_ok:
                    # Conservative fallback: keep the old normalized behavior but
                    # mark it explicitly for downstream diagnostics.
                    prefix_text = normalized_prefix_text
                    surface_target_text = normalized_target_text
                    span_source = "normalized_fallback"
                else:
                    span_source = "exact_caption_span"

                surface_target_text = surface_target_text.strip()
                if not surface_target_text or all(ch in ".,;:!?" for ch in surface_target_text):
                    continue
                key = (int(sent["image_id"]), bucket, raw_idx, surface_target_text.lower())
                if key in seen:
                    continue
                seen.add(key)
                by_bucket[bucket].append({
                    "probe_id": f"{sent['image_id']}:{bucket}:{raw_idx}:{len(by_bucket[bucket])}",
                    "image_id": int(sent["image_id"]),
                    "image": sent.get("image", ""),
                    "question": question,
                    "caption": caption,
                    "bucket": bucket,
                    "prefix_text": prefix_text,
                    "target_text": surface_target_text,
                    "normalized_prefix_text": normalized_prefix_text,
                    "normalized_target_text": normalized_target_text,
                    "span_source": span_source,
                    "chair_word": word,
                    "chair_node_word": node_word,
                    "raw_word_idx": raw_idx,
                    "mscoco_gt_words": sent.get("mscoco_gt_words", []),
                })

    records = []
    for bucket in ("hallucinated", "grounded"):
        candidates = by_bucket[bucket]
        rng.shuffle(candidates)
        if max_per_bucket and max_per_bucket > 0:
            candidates = candidates[:max_per_bucket]
        records.extend(candidates)
    records.sort(key=lambda x: (x["image_id"], x["bucket"], x["raw_word_idx"]))
    return records, {k: len(v) for k, v in by_bucket.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-eval", required=True)
    parser.add_argument("--coco-path", default="../dataset/coco/annotations")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--max-per-bucket", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", default="Please describe this image in detail.")
    args = parser.parse_args()

    sentences = load_eval(args.caption_eval)
    records, available = build_records(
        sentences,
        coco_path=args.coco_path,
        max_per_bucket=args.max_per_bucket,
        seed=args.seed,
        question=args.question,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = os.path.splitext(args.output_file)[0] + "_summary.json"
    summary = {
        "caption_eval": args.caption_eval,
        "output_file": args.output_file,
        "available_before_sampling": available,
        "written": len(records),
        "written_by_bucket": dict(defaultdict(int, {b: sum(1 for r in records if r["bucket"] == b) for b in ("hallucinated", "grounded")})),
        "max_per_bucket": args.max_per_bucket,
        "seed": args.seed,
        "target_definition": "exact caption surface span aligned from nltk-tokenized CHAIR object occurrence index; double-word objects keep the full phrase when the raw span matches",
        "span_source_counts": dict(defaultdict(int, {
            source: sum(1 for r in records if r.get("span_source") == source)
            for source in sorted({r.get("span_source", "unknown") for r in records})
        })),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
