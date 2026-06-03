import argparse
import copy
import json
import math
import os

os.environ.setdefault("VILA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")

import random
import shutil
import uuid
from collections import defaultdict

import torch
from PIL import Image as PILImage
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import set_seed

import llava
from llava import conversation as clib
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.media import Image
from llava.mm_utils import process_image, process_images
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation

from eval_scripts.eval_utils.chair import CHAIR

TXTATTN_TRACE_BUCKETS = (
    "all",
    "object",
    "hallucinated",
    "non_hallucinated",
)
TXTATTN_TRACE_METRICS = (
    "I_text",
    "generated_txt_attn",
    "image_attn",
    "txt_img_ratio",
)


def split_list(items, n):
    chunk_size = math.ceil(len(items) / n)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items, n, k):
    return split_list(items, n)[k]


def load_txtattn_heads(head_file, topk):
    with open(os.path.expanduser(head_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "heads" in data and isinstance(data["heads"], list):
            records = data["heads"]
        elif "hal_heads" in data and isinstance(data["hal_heads"], list):
            records = data["hal_heads"]
        else:
            raise ValueError(f"Unsupported txt-attn head file format: {head_file}")
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"Unsupported txt-attn head file format: {head_file}")
    selected = records if topk is None or int(topk) <= 0 else records[: int(topk)]
    heads = []
    for item in selected:
        if isinstance(item, dict):
            heads.append([int(item["layer"]), int(item["head"])])
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            heads.append([int(item[0]), int(item[1])])
        else:
            raise ValueError(f"Unsupported head record in {head_file}: {item}")
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
            value = float(values[metric])
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
                    "buckets": {
                        bucket: self._finalize_bucket(stats)
                        for bucket, stats in item["buckets"].items()
                    },
                }
                for key, item in self.by_head.items()
            },
        }


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
    num_records = 0
    with open(trace_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[resume] ignoring malformed trace line {line_no} in {trace_file}")
                continue
            head_values = record.get("head_values", [])
            if head_values:
                stats.update(buckets_from_txtattn_record(record), head_values)
                num_records += 1
    return num_records


def load_completed_question_ids(answers_file):
    completed = set()
    if not answers_file or not os.path.exists(answers_file):
        return completed
    with open(answers_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[resume] ignoring malformed answer line {line_no} in {answers_file}")
                continue
            question_id = item.get("question_id")
            if question_id is not None:
                completed.add(int(question_id))
    return completed


def safe_token_label(tokenizer, token_id):
    if token_id < 0:
        return f"<special:{token_id}>"
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return f"<id:{int(token_id)}>"


def extract_generated_ids(output_ids, input_ids):
    prompt_len = int(input_ids.shape[1])
    output_len = int(output_ids.shape[1])
    if output_len >= prompt_len and torch.equal(output_ids[:, :prompt_len].detach().cpu(), input_ids.detach().cpu()):
        return output_ids[:, prompt_len:]
    return output_ids


def get_special_token_ids(tokenizer):
    special_ids = set()
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        if token_id is not None:
            special_ids.add(int(token_id))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    for token_id in getattr(tokenizer, "media_token_ids", {}).values():
        if token_id is not None:
            special_ids.add(int(token_id))
    return special_ids


def get_image_token_id(tokenizer):
    media_ids = getattr(tokenizer, "media_token_ids", {}) or {}
    if "image" in media_ids:
        return int(media_ids["image"])
    token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    if token_id is None or token_id < 0:
        raise ValueError("Could not resolve VILA image token id")
    return int(token_id)


def infer_step_masks(input_ids, generated_prefix_ids, att_seq_len, special_token_ids, image_token_id=None):
    prompt_ids = input_ids[0].detach().cpu().tolist()
    generated_prefix_ids = [int(token_id) for token_id in generated_prefix_ids]
    generated_prefix_len = len(generated_prefix_ids)
    if image_token_id is None:
        image_positions = [idx for idx, token_id in enumerate(prompt_ids) if token_id < 0]
    else:
        image_positions = [idx for idx, token_id in enumerate(prompt_ids) if int(token_id) == int(image_token_id)]
    if len(image_positions) != 1:
        raise ValueError(f"Expected exactly one image token, found {len(image_positions)}")

    image_start = image_positions[0]
    visible_len = len(prompt_ids) + generated_prefix_len
    image_len = att_seq_len - (visible_len - 1)
    if image_len <= 0:
        raise ValueError(
            f"Invalid inferred image span: image_len={image_len}, visible_len={visible_len}, "
            f"att_seq_len={att_seq_len}"
        )
    image_end = image_start + image_len
    if image_end > att_seq_len:
        raise ValueError(f"Invalid image span: image_start={image_start}, image_end={image_end}, att_seq_len={att_seq_len}")

    image_mask = torch.zeros(att_seq_len, dtype=torch.bool)
    image_mask[image_start:image_end] = True

    prompt_after_image_len = len(prompt_ids) - image_start - 1
    generated_start = image_end + prompt_after_image_len

    sys_mask = torch.zeros(att_seq_len, dtype=torch.bool)
    sys_mask[:image_start] = True
    sys_mask[image_end:generated_start] = True

    txt_mask = torch.zeros(att_seq_len, dtype=torch.bool)
    available_generated_len = max(0, min(len(generated_prefix_ids), att_seq_len - generated_start))
    for offset, token_id in enumerate(generated_prefix_ids[:available_generated_len]):
        att_pos = generated_start + offset
        if token_id in special_token_ids:
            sys_mask[att_pos] = True
        else:
            txt_mask[att_pos] = True

    layout = {
        "prompt_visible_len": int(len(prompt_ids)),
        "generated_prefix_len": int(generated_prefix_len),
        "att_seq_len": int(att_seq_len),
        "image_start": int(image_start),
        "image_end": int(image_end),
        "image_len": int(image_len),
        "prompt_after_image_len": int(prompt_after_image_len),
        "generated_start": int(generated_start),
        "sys_len": int(sys_mask.sum().item()),
        "txt_len": int(txt_mask.sum().item()),
        "generated_special_len": int(sum(1 for token_id in generated_prefix_ids[:available_generated_len] if token_id in special_token_ids)),
    }
    return sys_mask, txt_mask, image_mask, layout


def classify_generation_steps(tokenizer, generated_ids_only, chair_evaluator, gt_objects):
    labels = []
    previous_object_count = 0
    token_ids = generated_ids_only[0].detach().cpu().tolist()
    for step_idx, token_id in enumerate(token_ids):
        prefix_ids = token_ids[: step_idx + 1]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
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


def extract_txtattn_trace_for_sample(
    question_id,
    image_file,
    caption,
    input_ids,
    generated_ids_only,
    attentions,
    step_labels,
    special_token_ids,
    heads,
    writer,
    stats,
    image_token_id,
):
    if attentions is None:
        return {"num_steps": 0, "num_records": 0}
    num_steps = min(len(attentions), int(generated_ids_only.shape[1]), len(step_labels))
    generated_ids_list = generated_ids_only[0].detach().cpu().tolist()
    num_records = 0
    eps = 1e-12

    for step_idx in range(num_steps):
        step_attentions = attentions[step_idx]
        att_seq_len = int(step_attentions[0].shape[-1])
        _, txt_mask, image_mask, layout = infer_step_masks(
            input_ids,
            generated_ids_list[:step_idx],
            att_seq_len,
            special_token_ids,
            image_token_id=image_token_id,
        )
        image_end = int(layout["image_end"])
        label = step_labels[step_idx]
        buckets = buckets_from_txtattn_record(label)
        head_values = []
        for layer_idx, head_idx in heads:
            layer_idx = int(layer_idx)
            head_idx = int(head_idx)
            if layer_idx >= len(step_attentions):
                continue
            att = step_attentions[layer_idx].detach().cpu()
            if att.shape[0] != 1 or head_idx >= att.shape[1]:
                continue
            q_idx = int(att.shape[-2] - 1)
            row = att[0, head_idx, q_idx, :].float()
            row_sum = row.sum()
            if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3):
                row = torch.softmax(row, dim=-1)
            i_text = float(row[image_end:].sum().item())
            generated_txt_attn = float(row[txt_mask].sum().item())
            image_attn = float(row[image_mask].sum().item())
            head_values.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "I_text": i_text,
                    "generated_txt_attn": generated_txt_attn,
                    "image_attn": image_attn,
                    "txt_img_ratio": generated_txt_attn / (image_attn + eps),
                }
            )
        if not head_values:
            continue
        stats.update(buckets, head_values)
        i_text_values = [x["I_text"] for x in head_values]
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
            "layout": layout,
            "mean_I_text": float(sum(i_text_values) / max(len(i_text_values), 1)),
            "max_I_text": float(max(i_text_values)) if i_text_values else None,
            "head_values": head_values,
        }
        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
        num_records += 1
    return {"num_steps": int(num_steps), "num_records": int(num_records)}


def build_questions_from_existing_sample_file(args):
    sample_path = os.path.expanduser(args.existing_sample_file)
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        entries = data.get("sentences") or data.get("samples")
    elif isinstance(data, list):
        entries = data
    else:
        entries = None
    if not entries:
        raise ValueError(f"No sample entries found in existing sample file: {sample_path}")
    questions, sampled_meta = [], []
    for idx, entry in enumerate(entries[: args.num_samples]):
        question_id = entry.get("question_id", entry.get("image_id"))
        image_file = entry.get("image")
        if question_id is None or image_file is None:
            raise ValueError(f"Entry {idx} must contain image_id/question_id and image fields: {entry}")
        item = {"question_id": int(question_id), "image": image_file, "text": args.prompt_text}
        questions.append(item)
        sampled_meta.append({**item, "prompt": args.prompt_text, "source_index": idx, "source_file": sample_path})
    if args.save_sample_ids:
        os.makedirs(os.path.dirname(args.save_sample_ids), exist_ok=True)
        with open(args.save_sample_ids, "w", encoding="utf-8") as f:
            json.dump(sampled_meta, f, indent=2, ensure_ascii=False)
    print(f"Loaded {len(questions)} existing samples from {sample_path}")
    return questions


def build_questions(args):
    if args.dataset != "coco":
        raise ValueError("Only --dataset coco is supported for this adapter.")
    if args.use_existing_sample_file:
        if not args.existing_sample_file:
            raise ValueError("--existing-sample-file is required when --use-existing-sample-file is set")
        return build_questions_from_existing_sample_file(args)
    coco = COCO(args.caption_file_path)
    sampled_img_ids = random.sample(coco.getImgIds(), args.num_samples)
    questions, sampled_meta = [], []
    dest_image_folder = os.path.join(
        os.path.split(os.path.split(os.path.dirname(args.answers_file))[0])[0],
        "images",
        f"seed{args.seed}_{args.num_samples}",
    )
    os.makedirs(dest_image_folder, exist_ok=True)
    for image_id in sampled_img_ids:
        image_file = coco.loadImgs(image_id)[0]["file_name"]
        item = {"question_id": int(image_id), "image": image_file, "text": args.prompt_text}
        questions.append(item)
        src = os.path.join(args.image_folder, image_file)
        dst = os.path.join(dest_image_folder, image_file)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)
        sampled_meta.append({**item, "prompt": args.prompt_text})
    if args.save_sample_ids:
        os.makedirs(os.path.dirname(args.save_sample_ids), exist_ok=True)
        with open(args.save_sample_ids, "w", encoding="utf-8") as f:
            json.dump(sampled_meta, f, indent=2, ensure_ascii=False)
    return questions


def build_chair_evaluator(args, questions):
    if not args.annotation_dir:
        return None
    imids = [int(q["question_id"]) for q in questions]
    chair = CHAIR(imids, args.annotation_dir)
    chair.get_annotations()
    return chair


def prepare_vila_inputs(model, image_path, prompt_text):
    conversation = [{"from": "human", "value": [Image(image_path), prompt_text]}]
    media_config = defaultdict(dict)
    media = extract_media(conversation, config=model.config)
    for name in media:
        if name == "image":
            if len(media["image"]) == 1 and model.config.image_aspect_ratio in ["dynamic", "dynamic_s2"]:
                model.config.image_processor = model.vision_tower.image_processor
                if model.config.image_aspect_ratio == "dynamic":
                    images = process_image(media["image"][0], model.config, None, enable_dynamic_res=True).half()
                    conversation[0]["value"] = conversation[0]["value"].replace(
                        DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                    )
                else:
                    if type(model.config.s2_scales) is str:
                        model.config.s2_scales = list(map(int, model.config.s2_scales.split(",")))
                    images, block_sizes = process_image(media["image"][0], model.config, None, enable_dynamic_s2=True)
                    images = images.half()
                    media_config[name]["block_sizes"] = [block_sizes]
            else:
                images = process_images(media["image"], model.vision_tower.image_processor, model.config).half()
            media[name] = [image for image in images]
        else:
            raise ValueError(f"Unsupported media type for caption eval: {name}")
    input_ids = tokenize_conversation(conversation, model.tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)
    inputs_embeds, _, attention_mask = model._embed(input_ids, media, media_config, None, None)
    return input_ids, inputs_embeds, attention_mask


def generate_with_optional_attentions(model, image_path, prompt_text, generation_config, output_attentions):
    input_ids, inputs_embeds, attention_mask = prepare_vila_inputs(model, image_path, prompt_text)
    output = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        use_cache=True,
        output_attentions=output_attentions,
        output_scores=False,
        output_hidden_states=False,
        return_dict_in_generate=True,
    )
    output_ids = output.sequences
    generated_ids_only = extract_generated_ids(output_ids, input_ids)
    caption = model.tokenizer.decode(generated_ids_only[0], skip_special_tokens=True).strip()
    return caption, output, input_ids, generated_ids_only


def write_summary(summary_file, stats, args, num_records):
    if not summary_file:
        return
    summary = stats.to_dict()
    summary["config"] = {
        "txtattn_head_file": args.txtattn_head_file,
        "txtattn_topk": args.txtattn_topk,
        "num_trace_records": num_records,
        "note": "VILA trace. I_text is sum of attention over image_end:, matching the dynamic intervention text-side slice.",
    }
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def eval_model(args):
    model = llava.load(
        os.path.expanduser(args.model_path),
        model_base=args.model_base,
        attn_implementation=os.environ.get("VILA_ATTN_IMPLEMENTATION", "eager"),
    )
    model.eval()
    if args.conv_mode != "auto":
        clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    questions_all = build_questions(args)
    chair_evaluator = build_chair_evaluator(args, questions_all) if (args.enable_attention_analysis or args.enable_txtattn_trace) else None
    questions = get_chunk(questions_all, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    completed = load_completed_question_ids(answers_file) if args.resume else set()
    if completed:
        before = len(questions)
        questions = [q for q in questions if int(q["question_id"]) not in completed]
        print(f"[resume] {len(completed)} completed answers found; running {len(questions)}/{before} remaining in this chunk.")

    generation_config = copy.deepcopy(model.default_generation_config)
    updates = {
        "do_sample": bool(args.temperature > 0),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
    }
    generation_config.update(**{k: v for k, v in updates.items() if v is not None})

    heads, trace_stats, trace_writer = [], None, None
    trace_count = 0
    output_attentions = args.enable_attention_analysis or args.enable_txtattn_trace
    if args.enable_txtattn_trace:
        if not args.txtattn_head_file:
            raise ValueError("--txtattn-head-file is required with --enable-txtattn-trace")
        heads = load_txtattn_heads(args.txtattn_head_file, args.txtattn_topk)
        trace_stats = TxtAttnTraceStats(heads)
        if args.resume and args.txtattn_output_file:
            trace_count += replay_txtattn_trace(args.txtattn_output_file, trace_stats)
        if args.txtattn_output_file:
            os.makedirs(os.path.dirname(args.txtattn_output_file), exist_ok=True)
            trace_writer = open(args.txtattn_output_file, "a" if args.resume else "w", encoding="utf-8")

    image_token_id = get_image_token_id(model.tokenizer)
    special_token_ids = get_special_token_ids(model.tokenizer)
    mode = "a" if args.resume else "w"
    try:
        with open(answers_file, mode, encoding="utf-8") as ans_file:
            for sample_idx, line in tqdm(enumerate(questions, start=1), total=len(questions)):
                question_id = int(line["question_id"])
                image_path = os.path.join(args.image_folder, line["image"])
                with PILImage.open(image_path) as img:
                    img.verify()
                with torch.inference_mode():
                    caption, output_dict, input_ids, generated_ids_only = generate_with_optional_attentions(
                        model,
                        image_path,
                        line["text"],
                        generation_config,
                        output_attentions=output_attentions,
                    )
                print(f"[{sample_idx}/{len(questions)}] question_id={question_id}")
                print(caption)

                metadata = {"model_path": args.model_path}
                if args.enable_txtattn_trace:
                    gt_objects = set(chair_evaluator.imid_to_objects.get(question_id, set())) if chair_evaluator else set()
                    step_labels = classify_generation_steps(model.tokenizer, generated_ids_only, chair_evaluator, gt_objects)
                    trace_info = extract_txtattn_trace_for_sample(
                        question_id=question_id,
                        image_file=line["image"],
                        caption=caption,
                        input_ids=input_ids,
                        generated_ids_only=generated_ids_only,
                        attentions=output_dict.attentions,
                        step_labels=step_labels,
                        special_token_ids=special_token_ids,
                        heads=heads,
                        writer=trace_writer,
                        stats=trace_stats,
                        image_token_id=image_token_id,
                    )
                    trace_count += int(trace_info.get("num_records", 0))
                    metadata["txtattn_trace"] = trace_info
                    if trace_writer:
                        trace_writer.flush()

                ans_file.write(
                    json.dumps(
                        {
                            "question_id": question_id,
                            "image": line["image"],
                            "prompt": line["text"],
                            "text": caption,
                            "answer_id": str(uuid.uuid4()),
                            "model_id": args.model_name,
                            "metadata": metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ans_file.flush()
    finally:
        if trace_writer is not None:
            trace_writer.close()
    if args.enable_txtattn_trace:
        write_summary(args.txtattn_summary_file, trace_stats, args, trace_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="vila")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--caption_file_path", type=str, required=True)
    parser.add_argument("--annotation-dir", type=str, default="")
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-text", type=str, default="Please describe this image in detail.")
    parser.add_argument("--save-sample-ids", type=str, default="")
    parser.add_argument("--use-existing-sample-file", action="store_true")
    parser.add_argument("--existing-sample-file", type=str, default="")
    parser.add_argument("--enable-attention-analysis", action="store_true")
    parser.add_argument("--enable-pre-token-analysis", action="store_true")
    parser.add_argument("--enable-txtattn-trace", action="store_true")
    parser.add_argument("--txtattn-head-file", type=str, default="")
    parser.add_argument("--txtattn-topk", type=int, default=0)
    parser.add_argument("--txtattn-output-file", type=str, default="")
    parser.add_argument("--txtattn-summary-file", type=str, default="")
    args = parser.parse_args()
    random.seed(args.seed)
    set_seed(args.seed)
    eval_model(args)
