#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import random
import shutil
from pathlib import Path

import shortuuid
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoProcessor, set_seed

from eval_scripts.eval_utils.chair import CHAIR
from qwen_dynamic.attention_patch import (
    configure_txtattn_trace,
    disable_txtattn_trace,
    get_txtattn_trace_buffer,
    install_qwen25_attention_patch,
    patch_qwen_attention_modules,
)

TXTATTN_TRACE_BUCKETS = ("all", "object", "hallucinated", "non_hallucinated")
TXTATTN_TRACE_METRICS = ("I_text", "generated_txt_attn", "image_attn", "txt_img_ratio")


def load_txtattn_heads(head_file, topk):
    with open(os.path.expanduser(head_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("heads", data.get("hal_heads")) if isinstance(data, dict) else data
    if records is None:
        raise ValueError(f"Unsupported txt-attn head file format: {head_file}")
    selected = records if topk is None or int(topk) <= 0 else records[: int(topk)]
    heads = []
    for item in selected:
        if isinstance(item, dict):
            heads.append([int(item["layer"]), int(item["head"])])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            heads.append([int(item[0]), int(item[1])])
        else:
            raise ValueError(f"Unsupported head record: {item}")
    return heads


class TxtAttnTraceStats:
    def __init__(self, heads):
        self.heads = [[int(layer), int(head)] for layer, head in heads]
        self.overall = {bucket: self._new_bucket() for bucket in TXTATTN_TRACE_BUCKETS}
        self.by_head = {
            f"{layer}-{head}": {
                "layer": int(layer),
                "head": int(head),
                "buckets": {bucket: self._new_bucket() for bucket in TXTATTN_TRACE_BUCKETS},
            }
            for layer, head in self.heads
        }

    @staticmethod
    def _new_bucket():
        return {
            "count": 0,
            "sum": {metric: 0.0 for metric in TXTATTN_TRACE_METRICS},
            "sumsq": {metric: 0.0 for metric in TXTATTN_TRACE_METRICS},
            "min": {metric: float("inf") for metric in TXTATTN_TRACE_METRICS},
            "max": {metric: float("-inf") for metric in TXTATTN_TRACE_METRICS},
        }

    @staticmethod
    def _update_bucket(bucket, values):
        bucket["count"] += 1
        for metric in TXTATTN_TRACE_METRICS:
            value = float(values.get(metric, 0.0))
            bucket["sum"][metric] += value
            bucket["sumsq"][metric] += value * value
            bucket["min"][metric] = min(bucket["min"][metric], value)
            bucket["max"][metric] = max(bucket["max"][metric], value)

    def update(self, buckets, head_values):
        for values in head_values:
            key = f"{int(values['layer'])}-{int(values['head'])}"
            if key not in self.by_head:
                continue
            for bucket in buckets:
                self._update_bucket(self.overall[bucket], values)
                self._update_bucket(self.by_head[key]["buckets"][bucket], values)

    @staticmethod
    def _finalize_bucket(bucket):
        count = int(bucket["count"])
        out = {"count": count, "metrics": {}}
        denom = max(count, 1)
        for metric in TXTATTN_TRACE_METRICS:
            mean = bucket["sum"][metric] / denom
            var = max(bucket["sumsq"][metric] / denom - mean * mean, 0.0)
            out["metrics"][metric] = {
                "mean": mean,
                "var": var,
                "min": None if count == 0 else bucket["min"][metric],
                "max": None if count == 0 else bucket["max"][metric],
            }
        return out

    def to_dict(self):
        return {
            "heads": self.heads,
            "buckets": {bucket: self._finalize_bucket(stats) for bucket, stats in self.overall.items()},
            "by_head": {
                key: {
                    "layer": item["layer"],
                    "head": item["head"],
                    "buckets": {bucket: self._finalize_bucket(stats) for bucket, stats in item["buckets"].items()},
                }
                for key, item in self.by_head.items()
            },
        }


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    return split_list(lst, n)[k]


def load_completed_question_ids(path, require_txtattn=False):
    completed = set()
    if not path or not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                item = json.loads(line)
            except Exception:
                print(f"[resume] ignoring malformed answer line {line_no} in {path}")
                continue
            qid = item.get("question_id")
            if qid is None:
                continue
            if require_txtattn:
                trace_meta = item.get("metadata", {}).get("txtattn_trace", {})
                if int(trace_meta.get("num_records", 0) or 0) <= 0:
                    continue
            completed.add(int(qid))
    return completed



def prune_incomplete_txtattn_answers(path):
    if not path or not os.path.exists(path):
        return 0
    kept = []
    removed = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                kept.append(line)
                continue
            trace_meta = item.get("metadata", {}).get("txtattn_trace", {})
            if int(trace_meta.get("num_records", 0) or 0) <= 0:
                removed += 1
                continue
            kept.append(line)
    if removed:
        backup = path + ".zero_trace.bak"
        os.replace(path, backup)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(kept)
        print(f"[resume] pruned {removed} zero-trace answer lines from {path}; backup -> {backup}")
    return removed

def buckets_from_label(label):
    buckets = ["all"]
    if label["is_object"]:
        buckets.append("object")
    if label["is_hallucinated"]:
        buckets.append("hallucinated")
    if label["is_non_hallucinated"]:
        buckets.append("non_hallucinated")
    return buckets


def buckets_from_txtattn_record(record):
    buckets = ["all"]
    if record.get("is_object"):
        buckets.append("object")
    if record.get("is_hallucinated"):
        buckets.append("hallucinated")
    if record.get("is_non_hallucinated"):
        buckets.append("non_hallucinated")
    return buckets

def replay_txtattn_trace(trace_file, stats):
    if not trace_file or not os.path.exists(trace_file):
        return 0
    count = 0
    with open(trace_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except Exception:
                print(f"[resume] ignoring malformed trace line {line_no} in {trace_file}")
                continue
            head_values = record.get("head_values", [])
            if head_values:
                stats.update(
                    ["all"]
                    + (["object"] if record.get("is_object") else [])
                    + (["hallucinated"] if record.get("is_hallucinated") else [])
                    + (["non_hallucinated"] if record.get("is_non_hallucinated") else []),
                    head_values,
                )
                count += 1
    return count


def safe_token_label(tokenizer, token_id):
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return f"<id:{int(token_id)}>"


def classify_generation_steps(tokenizer, generated_ids_only, chair_evaluator, gt_objects):
    labels = []
    previous_object_count = 0
    token_ids = generated_ids_only.detach().cpu().tolist()
    for step_idx, token_id in enumerate(token_ids):
        prefix_text = tokenizer.decode(token_ids[: step_idx + 1], skip_special_tokens=True)
        token_text = safe_token_label(tokenizer, token_id)
        objects, hallucinated, non_hallucinated = [], [], []
        if chair_evaluator is not None:
            try:
                words, node_words, _, _ = chair_evaluator.caption_to_words(prefix_text)
                for word, node_word in list(zip(words, node_words))[previous_object_count:]:
                    item = {"word": word, "node_word": node_word}
                    objects.append(item)
                    if node_word in gt_objects:
                        non_hallucinated.append(item)
                    else:
                        hallucinated.append(item)
                previous_object_count = len(node_words)
            except Exception:
                pass
        labels.append(
            {
                "step_idx": int(step_idx),
                "token_id": int(token_id),
                "token_text": token_text,
                "is_object": bool(objects),
                "is_hallucinated": bool(hallucinated),
                "is_non_hallucinated": bool(non_hallucinated),
                "objects": objects,
                "hallucinated_objects": hallucinated,
                "non_hallucinated_objects": non_hallucinated,
            }
        )
    return labels


def build_questions_from_existing_sample_file(args):
    with open(os.path.expanduser(args.existing_sample_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("sentences", data.get("samples")) if isinstance(data, dict) else data
    if not entries:
        raise ValueError(f"No sample entries found in {args.existing_sample_file}")
    coco = COCO(args.caption_file_path)
    id_to_img = {int(img["id"]): img for img in coco.dataset["images"]}
    questions, meta = [], []
    for idx, entry in enumerate(entries):
        if isinstance(entry, int):
            qid = int(entry)
            image_file = id_to_img[qid]["file_name"]
        else:
            qid = entry.get("question_id", entry.get("image_id"))
            image_file = entry.get("image")
        if qid is None or image_file is None:
            raise ValueError(f"Bad sample entry {idx}: {entry}")
        item = {"question_id": int(qid), "image": image_file, "text": args.prompt_text}
        questions.append(item)
        meta.append({**item, "source_index": int(idx), "source_file": args.existing_sample_file})
    if args.save_sample_ids:
        os.makedirs(os.path.dirname(args.save_sample_ids), exist_ok=True)
        with open(args.save_sample_ids, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    print(f"Loaded {len(questions)} existing samples from {args.existing_sample_file}")
    return questions


def build_questions(args):
    if args.use_existing_sample_file:
        return build_questions_from_existing_sample_file(args)
    coco = COCO(args.caption_file_path)
    random.seed(args.seed)
    img_ids = random.sample(coco.getImgIds(), args.num_samples)
    questions, meta = [], []
    for img_id in img_ids:
        image_file = coco.loadImgs(img_id)[0]["file_name"]
        item = {"question_id": int(img_id), "image": image_file, "text": args.prompt_text}
        questions.append(item)
        meta.append(item)
    if args.save_sample_ids:
        os.makedirs(os.path.dirname(args.save_sample_ids), exist_ok=True)
        with open(args.save_sample_ids, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return questions


def token_id(tokenizer, candidates):
    for token in candidates:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            return int(tid)
    return None


def locate_qwen_spans(input_ids, tokenizer):
    ids = input_ids.detach().cpu().tolist()
    vision_start_id = token_id(tokenizer, ["<|vision_start|>"])
    vision_end_id = token_id(tokenizer, ["<|vision_end|>"])
    if vision_start_id is None or vision_end_id is None:
        raise ValueError("Could not resolve Qwen vision boundary token ids.")
    vision_start = ids.index(vision_start_id)
    vision_end = len(ids) - 1 - ids[::-1].index(vision_end_id)
    if vision_end <= vision_start:
        raise ValueError("Invalid Qwen vision span.")
    img_start = vision_start
    img_length = vision_end - vision_start + 1
    generated_start = len(ids)
    return img_start, img_length, generated_start


def write_last_row_trace(question_id, image_file, generated_ids_only, trace_steps, labels, writer, stats):
    num_steps = min(len(trace_steps), int(generated_ids_only.numel()), len(labels))
    for step_idx in range(num_steps):
        trace_step = trace_steps[step_idx] or {}
        head_values = trace_step.get("head_values", [])
        label = labels[step_idx]
        buckets = buckets_from_label(label)
        stats.update(buckets, head_values)
        i_text_values = [float(x.get("I_text", 0.0)) for x in head_values]
        record = {
            "question_id": int(question_id),
            "image": image_file,
            "step_idx": int(step_idx),
            "token_id": label["token_id"],
            "token_text": label["token_text"],
            "is_object": label["is_object"],
            "is_hallucinated": label["is_hallucinated"],
            "is_non_hallucinated": label["is_non_hallucinated"],
            "objects": label["objects"],
            "hallucinated_objects": label["hallucinated_objects"],
            "non_hallucinated_objects": label["non_hallucinated_objects"],
            "layout": trace_step.get("layout", {}),
            "mean_I_text": float(sum(i_text_values) / max(len(i_text_values), 1)),
            "max_I_text": float(max(i_text_values)) if i_text_values else None,
            "head_values": head_values,
            "trace_mode": "last_row",
        }
        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"num_steps": int(num_steps), "num_records": int(num_steps), "trace_mode": "last_row"}


def build_messages(image_obj, prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def eval_model(args):
    install_qwen25_attention_patch()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
    except Exception:
        try:
            from transformers import AutoModelForImageTextToText as QwenVLModel
        except Exception as exc:
            raise RuntimeError("This environment does not provide Qwen2.5-VL model classes. Install a recent transformers build.") from exc

    model = QwenVLModel.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    patch_qwen_attention_modules(model)

    questions = get_chunk(build_questions(args), args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if args.resume and args.enable_txtattn_trace:
        prune_incomplete_txtattn_answers(answers_file)
    completed = load_completed_question_ids(answers_file, require_txtattn=args.enable_txtattn_trace) if args.resume else set()
    if completed:
        before = len(questions)
        questions = [q for q in questions if int(q["question_id"]) not in completed]
        print(f"[resume] found {len(completed)} completed; running {len(questions)}/{before} remaining.")

    chair_evaluator = CHAIR([q["question_id"] for q in questions], args.annotation_dir or os.path.dirname(args.caption_file_path))
    chair_evaluator.get_annotations()

    txtattn_heads = load_txtattn_heads(args.txtattn_head_file, args.txtattn_topk) if args.enable_txtattn_trace else []
    txtattn_stats = TxtAttnTraceStats(txtattn_heads) if args.enable_txtattn_trace else None
    txtattn_writer = None
    if args.enable_txtattn_trace:
        txtattn_output = os.path.expanduser(args.txtattn_output_file or os.path.join(os.path.dirname(answers_file), "txtattn_trace.jsonl"))
        os.makedirs(os.path.dirname(txtattn_output), exist_ok=True)
        if args.resume:
            replayed = replay_txtattn_trace(txtattn_output, txtattn_stats)
            if replayed:
                print(f"[resume] replayed {replayed} existing trace records from {txtattn_output}")
        txtattn_writer = open(txtattn_output, "a" if args.resume else "w", encoding="utf-8")
        print(f"[txtattn] tracing {len(txtattn_heads)} heads -> {txtattn_output}")

    ans_file = open(answers_file, "a" if args.resume else "w", encoding="utf-8")
    for sample_idx, line in tqdm(list(enumerate(questions, start=1)), total=len(questions)):
        question_id = int(line["question_id"])
        image_file = line["image"]
        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        messages = build_messages(image, line["text"])
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(model.device)
        input_ids = inputs["input_ids"][0]
        img_start, img_length, generated_start = locate_qwen_spans(input_ids, tokenizer)

        if args.enable_txtattn_trace:
            configure_txtattn_trace(model, txtattn_heads, img_start, img_length, generated_start)

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=bool(args.temperature > 0),
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=False,
            )

        gen_only = generated[0, input_ids.shape[0] :]
        output_text = processor.batch_decode([gen_only], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        print(f"[{sample_idx}/{len(questions)}] question_id={question_id}\n{output_text}")

        metadata = {}
        if args.enable_txtattn_trace:
            gt_objects = set(chair_evaluator.imid_to_objects.get(question_id, set()))
            labels = classify_generation_steps(tokenizer, gen_only, chair_evaluator, gt_objects)
            trace_steps = list(get_txtattn_trace_buffer(model) or [])
            metadata["txtattn_trace"] = write_last_row_trace(
                question_id, image_file, gen_only, trace_steps, labels, txtattn_writer, txtattn_stats
            )
            disable_txtattn_trace(model)

        ans_file.write(
            json.dumps(
                {
                    "question_id": question_id,
                    "image": image_file,
                    "prompt": line["text"],
                    "text": output_text,
                    "answer_id": shortuuid.uuid(),
                    "model_id": args.model_path,
                    "metadata": metadata,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        ans_file.flush()

        del inputs, generated, gen_only, input_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ans_file.close()
    if txtattn_writer is not None:
        txtattn_writer.close()

    if args.enable_txtattn_trace:
        summary_file = os.path.expanduser(args.txtattn_summary_file or os.path.join(os.path.dirname(answers_file), "txtattn_summary.json"))
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        summary = txtattn_stats.to_dict()
        summary["config"] = {
            "model_path": args.model_path,
            "txtattn_head_file": args.txtattn_head_file,
            "txtattn_topk": args.txtattn_topk,
            "txtattn_trace_mode": "last_row",
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "note": "Qwen I_text is attention over text-side tokens after <|vision_end|>.",
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--caption_file_path", required=True)
    parser.add_argument("--annotation-dir", default="")
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-text", default="Please describe this image in detail.")
    parser.add_argument("--save-sample-ids", default="")
    parser.add_argument("--use-existing-sample-file", action="store_true")
    parser.add_argument("--existing-sample-file", default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--enable-txtattn-trace", action="store_true")
    parser.add_argument("--txtattn-head-file", default="")
    parser.add_argument("--txtattn-topk", type=int, default=0)
    parser.add_argument("--txtattn-output-file", default="")
    parser.add_argument("--txtattn-summary-file", default="")
    args = parser.parse_args()
    if args.dataset != "coco":
        raise ValueError("Qwen head analysis currently supports --dataset coco")
    set_seed(args.seed)
    eval_model(args)


if __name__ == "__main__":
    main()
