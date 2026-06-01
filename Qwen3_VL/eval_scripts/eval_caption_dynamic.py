#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import random

import shortuuid
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoProcessor, set_seed

from qwen_dynamic.attention_patch import (
    configure_dynamic_intervention,
    install_qwen25_attention_patch,
    patch_qwen_attention_modules,
    get_intervention_stats,
)
from eval_scripts.eval_caption import locate_qwen_spans, build_messages


def load_completed_question_ids(path):
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
            if qid is not None:
                completed.add(int(qid))
    return completed


def normalize_head_scores(raw_scores, mode):
    if mode == "raw":
        return [min(max(float(s), 0.0), 1.0) for s in raw_scores]
    if mode == "minmax":
        lo, hi = min(raw_scores), max(raw_scores)
        if abs(hi - lo) < 1e-8:
            return [1.0 for _ in raw_scores]
        return [(float(s) - lo) / (hi - lo) for s in raw_scores]
    if mode == "logminmax":
        logged = [math.log1p(max(float(s), 0.0)) for s in raw_scores]
        lo, hi = min(logged), max(logged)
        if abs(hi - lo) < 1e-8:
            return [1.0 for _ in logged]
        return [(s - lo) / (hi - lo) for s in logged]
    if mode == "rank_percentile":
        n = len(raw_scores)
        if n <= 1:
            return [1.0 for _ in raw_scores]
        return [(n - 1 - i) / (n - 1) for i in range(n)]
    raise ValueError(f"Unsupported head_score_normalize: {mode}")


def score_from_head_record(record, score_key):
    if score_key in record:
        return float(record[score_key])
    # Combo files use names like ranked_heads_global__A__B.json, while each
    # record stores the raw combo field as A__B. Keep both forms usable.
    parts = str(score_key).split("__", 1)
    if len(parts) == 2 and parts[0] in {"global", "local"}:
        stripped = parts[1]
        if stripped in record:
            return float(record[stripped])
    if str(score_key).startswith("layerweighted_"):
        stripped = str(score_key).split("__", 1)[1] if "__" in str(score_key) else ""
        if stripped in record:
            return float(record[stripped])
    if "score" in record:
        return float(record["score"])
    return 1.0


def load_selected_heads(head_file, topk, score_key="score", score_normalize="minmax"):
    with open(os.path.expanduser(head_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "heads" in data:
        records = data["heads"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported head file format: {head_file}")

    top = records[:topk]
    heads = [[int(x["layer"]), int(x["head"])] if isinstance(x, dict) else [int(x[0]), int(x[1])] for x in top]
    if top and isinstance(top[0], dict):
        score_records = records if score_normalize in ("logminmax", "rank_percentile") else top
        raw_scores = [score_from_head_record(x, score_key) for x in score_records]
        norm_scores = normalize_head_scores(raw_scores, score_normalize)
        norm_by_head = {
            f"{int(x['layer'])}-{int(x['head'])}": float(ns)
            for x, ns in zip(score_records, norm_scores)
        }
        score_map = {f"{l}-{h}": norm_by_head.get(f"{l}-{h}", 1.0) for l, h in heads}
    else:
        score_map = {f"{l}-{h}": 1.0 for l, h in heads}
    return heads, score_map


def load_or_sample_ids(args):
    if args.sample_id_file and os.path.exists(args.sample_id_file):
        with open(args.sample_id_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data and isinstance(data[0], dict):
            return [int(x.get("question_id", x.get("image_id"))) for x in data]
        return [int(x) for x in data]

    coco = COCO(args.caption_file_path)
    sampled_ids = random.sample(coco.getImgIds(), args.num_samples)
    if args.save_sample_id_file:
        os.makedirs(os.path.dirname(args.save_sample_id_file), exist_ok=True)
        with open(args.save_sample_id_file, "w", encoding="utf-8") as f:
            json.dump(sampled_ids, f, indent=2)
    return sampled_ids


def build_questions(args, sampled_ids):
    coco = COCO(args.caption_file_path)
    id_to_img = {int(img["id"]): img for img in coco.dataset["images"]}
    return [
        {
            "question_id": int(img_id),
            "image": id_to_img[int(img_id)]["file_name"],
            "text": args.prompt_text,
        }
        for img_id in sampled_ids
    ]


def finalize_intervention_stats(raw_stats):
    def finalize_bucket(bucket):
        out = dict(bucket)
        count = max(int(out.get("count", 0)), 1)
        for key in list(out.keys()):
            if key.startswith("sum_"):
                metric = key[len("sum_"):]
                out[f"mean_{metric}"] = out[key] / count
                del out[key]
        out["scaled_rate"] = out.get("scaled_count", 0) / count
        out["near_zero_rate"] = out.get("near_zero_count", 0) / count
        return out

    if not raw_stats:
        return {"overall": {}, "by_head": {}}
    return {
        "overall": {mode: finalize_bucket(bucket) for mode, bucket in raw_stats.get("overall", {}).items()},
        "by_head": {key: finalize_bucket(bucket) for key, bucket in raw_stats.get("by_head", {}).items()},
    }




def total_intervention_count(stats):
    total = 0
    for bucket in (stats or {}).get("overall", {}).values():
        total += int(bucket.get("count", 0))
    return total


def merge_intervention_bucket(old, new):
    old = old or {}
    new = new or {}
    old_count = int(old.get("count", 0))
    new_count = int(new.get("count", 0))
    total_count = old_count + new_count
    if total_count <= 0:
        return dict(new or old)

    merged = {}
    for key in sorted(set(old) | set(new)):
        if key == "count":
            continue
        old_value = old.get(key)
        new_value = new.get(key)
        if key.startswith("mean_"):
            old_sum = float(old_value or 0.0) * old_count
            new_sum = float(new_value or 0.0) * new_count
            merged[key] = (old_sum + new_sum) / total_count
        elif key.startswith("min_"):
            vals = [v for v in (old_value, new_value) if v is not None]
            if vals:
                merged[key] = min(vals)
        elif key.startswith("max_"):
            vals = [v for v in (old_value, new_value) if v is not None]
            if vals:
                merged[key] = max(vals)
        elif key in ("scaled_count", "near_zero_count"):
            merged[key] = int(old_value or 0) + int(new_value or 0)
        elif key in ("scaled_rate", "near_zero_rate"):
            continue
        elif new_value is not None:
            merged[key] = new_value
        elif old_value is not None:
            merged[key] = old_value

    merged["count"] = total_count
    merged["scaled_rate"] = merged.get("scaled_count", 0) / total_count
    merged["near_zero_rate"] = merged.get("near_zero_count", 0) / total_count
    return merged


def merge_intervention_stats(old_stats, new_stats):
    if not old_stats:
        return new_stats
    if not new_stats or total_intervention_count(new_stats) == 0:
        return old_stats
    merged = {"overall": {}, "by_head": {}}
    for section in ("overall", "by_head"):
        old_section = old_stats.get(section, {})
        new_section = new_stats.get(section, {})
        for key in sorted(set(old_section) | set(new_section)):
            merged[section][key] = merge_intervention_bucket(old_section.get(key), new_section.get(key))
    return merged


def maybe_merge_resume_intervention_stats(stats, out_file, args):
    if not getattr(args, "resume", False) or not os.path.exists(out_file):
        return stats
    if total_intervention_count(stats) == 0:
        print(f"[resume] no new intervention stats; keeping existing stats at {out_file}")
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)
    with open(out_file, "r", encoding="utf-8") as f:
        existing = json.load(f)
    existing_count = total_intervention_count(existing)
    new_count = total_intervention_count(stats)
    merged = merge_intervention_stats(existing, stats)
    print(
        f"[resume] merged intervention stats at {out_file}: "
        f"existing_count={existing_count}, new_count={new_count}, "
        f"merged_count={total_intervention_count(merged)}"
    )
    return merged


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.expanduser(path)), exist_ok=True)
    with open(os.path.expanduser(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_run_config(args, heads):
    out_dir = os.path.dirname(os.path.expanduser(args.answers_file))
    save_json(os.path.join(out_dir, "run_config.json"), {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "resume": args.resume,
        "intervention": args.intervention,
        "topk": args.topk,
        "head_source": args.head_source,
        "head_file": args.head_file,
        "head_score_key": args.head_score_key,
        "head_score_normalize": args.head_score_normalize,
        "use_head_scores": args.use_head_scores,
        "dynamic_strength": args.dynamic_strength,
        "dynamic_ratio_power": args.dynamic_ratio_power,
        "dynamic_score_power": args.dynamic_score_power,
        "dynamic_tau": args.dynamic_tau,
        "dynamic_exp_sharpness": args.dynamic_exp_sharpness,
        "dynamic_context_mode": args.dynamic_context_mode,
        "dynamic_redistribute": args.dynamic_redistribute,
        "log_dynamic_trace": args.log_dynamic_trace,
        "dynamic_trace_topn": args.dynamic_trace_topn,
        "dynamic_trace_every": args.dynamic_trace_every,
        "selected_heads": heads,
    })


def save_intervention_stats(model, args):
    if not args.log_intervention_stats:
        return
    out_file = args.intervention_stats_file or os.path.join(os.path.dirname(os.path.expanduser(args.answers_file)), "intervention_stats.json")
    out_file = os.path.expanduser(out_file)
    if getattr(args, "resume", False) and getattr(args, "_resume_remaining_questions", 1) == 0 and os.path.exists(out_file):
        print(f"[resume] no remaining questions; keeping existing intervention stats at {out_file}")
        return

    stats = finalize_intervention_stats(get_intervention_stats(model))
    stats = maybe_merge_resume_intervention_stats(stats, out_file, args)
    stats["config"] = {
        "intervention": args.intervention,
        "topk": args.topk,
        "dynamic_strength": args.dynamic_strength,
        "dynamic_ratio_power": args.dynamic_ratio_power,
        "dynamic_score_power": args.dynamic_score_power,
        "dynamic_tau": args.dynamic_tau,
        "dynamic_exp_sharpness": args.dynamic_exp_sharpness,
        "dynamic_context_mode": args.dynamic_context_mode,
        "dynamic_redistribute": args.dynamic_redistribute,
        "use_head_scores": args.use_head_scores,
        "head_file": args.head_file,
        "head_score_key": args.head_score_key,
        "head_score_normalize": args.head_score_normalize,
    }
    save_json(out_file, stats)


def eval_model(args):
    install_qwen25_attention_patch()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
    except Exception:
        from transformers import AutoModelForImageTextToText as QwenVLModel

    model = QwenVLModel.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    patch_qwen_attention_modules(model)

    if args.head_source != "file" or not args.head_file:
        raise ValueError("Qwen dynamic currently requires --head-source file --head-file")
    heads, score_map = load_selected_heads(args.head_file, args.topk, args.head_score_key, args.head_score_normalize)
    save_run_config(args, heads)

    sampled_ids = load_or_sample_ids(args)
    questions = build_questions(args, sampled_ids)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    completed = load_completed_question_ids(answers_file) if args.resume else set()
    if completed:
        before = len(questions)
        questions = [q for q in questions if int(q["question_id"]) not in completed]
        print(f"[resume] {len(completed)} completed answers found; running {len(questions)}/{before} remaining.")
    args._resume_remaining_questions = len(questions)

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
        img_start, img_length, _ = locate_qwen_spans(input_ids, tokenizer)
        configure_dynamic_intervention(model, heads, score_map, img_start, img_length, args)

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

        gen_only = generated[0, input_ids.shape[0]:]
        output_text = processor.batch_decode([gen_only], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        print(f"[{sample_idx}/{len(questions)}] question_id={question_id}\n{output_text}")
        ans_file.write(json.dumps({
            "question_id": question_id,
            "image": image_file,
            "prompt": line["text"],
            "text": output_text,
            "answer_id": shortuuid.uuid(),
            "model_id": args.model_path,
            "metadata": {"intervention": args.intervention},
        }, ensure_ascii=False) + "\n")
        ans_file.flush()
        del inputs, generated, gen_only, input_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ans_file.close()
    save_intervention_stats(model, args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--caption_file_path", required=True)
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
    parser.add_argument("--sample-id-file", default="")
    parser.add_argument("--save-sample-id-file", default="")
    parser.add_argument("--intervention", choices=["none", "dynamic"], default="dynamic")
    parser.add_argument("--head-source", choices=["file"], default="file")
    parser.add_argument("--head-file", default="")
    parser.add_argument("--head-score-key", default="score")
    parser.add_argument("--head-score-normalize", choices=["minmax", "raw", "logminmax", "rank_percentile"], default="rank_percentile")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--use-head-scores", action="store_true")
    parser.add_argument("--dynamic-strength", type=float, default=1.0)
    parser.add_argument("--dynamic-ratio-power", type=float, default=1.0)
    parser.add_argument("--dynamic-score-power", type=float, default=1.0)
    parser.add_argument("--dynamic-context-mode", choices=["ratio_exp", "ratio_power", "text_exp", "text_power"], default="ratio_exp")
    parser.add_argument("--dynamic-tau", type=float, default=0.9)
    parser.add_argument("--dynamic-exp-sharpness", type=float, default=8.0)
    parser.add_argument("--dynamic-redistribute", choices=["renorm", "system", "system_only", "vision", "vision_only"], default="renorm")
    parser.add_argument("--log-dynamic-trace", action="store_true")
    parser.add_argument("--dynamic-trace-topn", type=int, default=10)
    parser.add_argument("--dynamic-trace-every", type=int, default=5)
    parser.add_argument("--log-intervention-stats", action="store_true")
    parser.add_argument("--intervention-stats-file", default="")
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)


if __name__ == "__main__":
    main()
