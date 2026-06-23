import os
import json
import math
import shutil
import random
import argparse

import shortuuid
import torch
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from transformers import LogitsProcessor, set_seed
from transformers.generation.logits_process import LogitsProcessorList
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path


class PAITextCFGLogitsProcessor(LogitsProcessor):
    def __init__(self, guidance_scale, uncond_input_ids, model):
        self.guidance_scale = float(guidance_scale)
        self.uncond_input_ids = uncond_input_ids
        self.model = model
        self.uncond_outputs = None

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1.0:
            return scores

        if self.uncond_outputs is None:
            uncond = self.uncond_input_ids
            if uncond.shape[0] != input_ids.shape[0]:
                repeat = int(math.ceil(input_ids.shape[0] / uncond.shape[0]))
                uncond = uncond.repeat(repeat, 1)[: input_ids.shape[0]]
            try:
                self.model.config.pai_cfg_active = True
                self.uncond_outputs = self.model(uncond, use_cache=True)
            finally:
                self.model.config.pai_cfg_active = False
        else:
            try:
                self.model.config.pai_cfg_active = True
                self.uncond_outputs = self.model(
                    input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=self.uncond_outputs.past_key_values,
                )
            finally:
                self.model.config.pai_cfg_active = False

        unconditional_logits = F.log_softmax(self.uncond_outputs.logits[:, -1, :], dim=-1)
        cutoff = torch.log(torch.tensor(0.1, device=scores.device, dtype=scores.dtype)) + scores.max(dim=-1, keepdim=True).values
        refined = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return refined.masked_fill(scores < cutoff, -float("inf"))


def build_text_only_input_ids(input_ids):
    rows = []
    for row in input_ids:
        rows.append(row[row != IMAGE_TOKEN_INDEX])
    max_len = max(int(row.shape[0]) for row in rows)
    if all(int(row.shape[0]) == max_len for row in rows):
        return torch.stack(rows, dim=0)
    pad_id = 0
    padded = []
    for row in rows:
        if int(row.shape[0]) < max_len:
            pad = torch.full((max_len - int(row.shape[0]),), pad_id, dtype=row.dtype, device=row.device)
            row = torch.cat([pad, row], dim=0)
        padded.append(row)
    return torch.stack(padded, dim=0)


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

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def default_head_setup(model_path):
    model_path_l = str(model_path).lower()
    if model_path == "liuhaotian/llava-v1.5-7b" or "llava-v1.5-7b" in model_path_l:
        return {
            "heads": [
                [16, 29], [26, 9], [13, 31], [15, 10], [20, 12],
                [30, 9], [19, 18], [17, 0], [18, 9], [26, 28],
                [19, 27], [18, 26], [15, 25], [14, 16], [31, 26],
                [15, 24], [31, 3], [22, 20], [27, 29], [17, 28]
            ],
            "img_start_pos": 35,
            "img_length": 576,
        }
    elif model_path == "liuhaotian/llava-v1.5-13b" or "llava-v1.5-13b" in model_path_l:
        return {
            "heads": [
                [0, 8], [29, 27], [23, 18], [20, 11], [36, 26], [19, 37], [22, 16], [22, 34], [21, 31], [20, 34],
                [37, 11], [17, 25], [35, 10], [17, 5], [15, 26], [0, 22], [19, 5], [19, 0], [14, 1], [23, 20],
                [21, 6], [30, 24], [26, 27], [21, 32], [15, 28], [15, 31], [19, 30], [20, 8], [19, 14], [14, 9],
                [39, 26], [25, 1], [18, 32], [17, 27], [39, 32]
            ],
            "img_start_pos": 35,
            "img_length": 576,
        }
    elif model_path == "liuhaotian/llava-v1.6-34b" or "llava-v1.6-34b" in model_path_l:
        return {
            "heads": [
                [45, 34], [43, 4], [43, 48], [44, 29], [35, 47],
                [40, 27], [54, 34], [37, 48], [43, 2], [41, 34]
            ],
            "img_start_pos": 33,
            "img_length": 1948,
        }
    else:
        raise ValueError(f"Unknown default head setup for model_path={model_path}")


def normalize_head_scores(raw_scores, mode):
    if mode == "raw":
        return [min(max(float(s), 0.0), 1.0) for s in raw_scores]

    if mode == "minmax":
        s_min, s_max = min(raw_scores), max(raw_scores)
        if abs(s_max - s_min) < 1e-8:
            return [1.0 for _ in raw_scores]
        return [(float(s) - s_min) / (s_max - s_min) for s in raw_scores]

    if mode == "logminmax":
        positive = [max(float(s), 0.0) for s in raw_scores]
        logged = [math.log1p(s) for s in positive]
        l_min, l_max = min(logged), max(logged)
        if abs(l_max - l_min) < 1e-8:
            return [1.0 for _ in logged]
        return [(s - l_min) / (l_max - l_min) for s in logged]

    if mode == "rank_percentile":
        n = len(raw_scores)
        if n <= 1:
            return [1.0 for _ in raw_scores]
        return [(n - 1 - i) / (n - 1) for i in range(n)]


    raise ValueError(f"Unsupported head_score_normalize: {mode}")


def score_from_head_record(record, score_key):
    if score_key in record:
        return float(record[score_key])
    if "score" in record:
        return float(record["score"])
    if "txt_attn_raw" in record:
        return float(record["txt_attn_raw"])
    return 1.0


def load_selected_heads(
    head_file,
    topk,
    score_key="score",
    score_normalize="minmax",
    min_back_raw=0.0,
):
    with open(head_file, "r") as f:
        data = json.load(f)

    def keep_record(record):
        if not isinstance(record, dict) or min_back_raw <= 0:
            return True
        if "back_raw" not in record:
            return True
        return float(record["back_raw"]) >= float(min_back_raw)

    def select_records(records):
        top_records = records[:topk]
        selected = [x for x in top_records if keep_record(x)]
        if len(selected) < len(top_records):
            print(
                f"[head-filter] kept {len(selected)}/{len(top_records)} heads "
                f"with back_raw >= {min_back_raw}"
            )
        return selected

    if isinstance(data, dict):
        if "heads" in data and isinstance(data["heads"], list):
            records = data["heads"]
            top = select_records(records)
            heads = [[int(x["layer"]), int(x["head"])] for x in top]
            score_records = records if score_normalize in ("logminmax", "rank_percentile") else top
            raw_scores = [score_from_head_record(x, score_key) for x in score_records]
            norm_scores = normalize_head_scores(raw_scores, score_normalize)
            norm_by_head = {
                f"{int(x['layer'])}-{int(x['head'])}": float(ns)
                for x, ns in zip(score_records, norm_scores)
            }
            score_map = {
                f"{int(x['layer'])}-{int(x['head'])}": norm_by_head[f"{int(x['layer'])}-{int(x['head'])}"]
                for x in top
            }
            return heads, score_map
        elif "topk_sets" in data:
            heads = data["topk_sets"].get(str(topk), data.get("hal_heads", []))[:topk]
            score_map = {f"{int(l)}-{int(h)}": 1.0 for l, h in heads}
            return heads, score_map
        elif "hal_heads" in data:
            heads = data["hal_heads"][:topk]
            score_map = {f"{int(l)}-{int(h)}": 1.0 for l, h in heads}
            return heads, score_map
        else:
            raise ValueError(f"Unsupported head file format: {head_file}")

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        records = data
        top = select_records(records)
        heads = [[int(x["layer"]), int(x["head"])] for x in top]
        score_records = records if score_normalize in ("logminmax", "rank_percentile") else top
        raw_scores = [score_from_head_record(x, score_key) for x in score_records]
        norm_scores = normalize_head_scores(raw_scores, score_normalize)
        norm_by_head = {
            f"{int(x['layer'])}-{int(x['head'])}": float(ns)
            for x, ns in zip(score_records, norm_scores)
        }
        score_map = {
            f"{int(x['layer'])}-{int(x['head'])}": norm_by_head[f"{int(x['layer'])}-{int(x['head'])}"]
            for x in top
        }
        return heads, score_map

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
        heads = [[int(x[0]), int(x[1])] for x in data[:topk]]
        score_map = {f"{int(l)}-{int(h)}": 1.0 for l, h in heads}
        return heads, score_map

    raise ValueError(f"Unsupported head file format: {head_file}")


def resolve_head_config(args):
    base = default_head_setup(args.model_path)

    if args.head_source == "default":
        heads = base["heads"][:args.topk]
        score_map = {f"{int(l)}-{int(h)}": 1.0 for l, h in heads}
    elif args.head_source == "file":
        if not args.head_file:
            raise ValueError("--head-source file requires --head-file")
        heads, score_map = load_selected_heads(
            args.head_file,
            args.topk,
            score_key=args.head_score_key,
            score_normalize=args.head_score_normalize,
            min_back_raw=args.min_head_back_raw,
        )
    else:
        raise ValueError(f"Unsupported head_source: {args.head_source}")

    return {
        "heads": heads,
        "scores": score_map,
        "img_start_pos": base["img_start_pos"],
        "img_length": base["img_length"],
    }


def load_or_sample_ids(args):
    if args.sample_id_file and os.path.exists(args.sample_id_file):
        with open(args.sample_id_file, "r") as f:
            return json.load(f)

    if args.dataset == "coco":
        coco = COCO(args.caption_file_path)
        img_ids = coco.getImgIds()
        sampled_ids = random.sample(img_ids, args.num_samples)
    elif args.dataset == "nocaps":
        val_caps = json.load(open(args.caption_file_path))
        image_infos = val_caps["images"]
        out_infos = [x for x in image_infos if x["domain"] == "out-domain"]
        sampled = random.sample(out_infos, args.num_samples)
        sampled_ids = [x["id"] for x in sampled]
    else:
        raise ValueError(args.dataset)

    if args.save_sample_id_file:
        os.makedirs(os.path.dirname(args.save_sample_id_file), exist_ok=True)
        with open(args.save_sample_id_file, "w") as f:
            json.dump(sampled_ids, f, indent=2)

    return sampled_ids


def build_questions(args, sampled_ids):
    questions = []

    if args.dataset == "coco":
        coco = COCO(args.caption_file_path)
        id_to_img = {img["id"]: img for img in coco.dataset["images"]}

        dest_image_folder = args.copy_image_folder or os.path.join(
            os.path.split(os.path.split(os.path.dirname(args.answers_file))[0])[0],
            "images",
            f"seed{args.seed}_{args.num_samples}"
        )
        os.makedirs(dest_image_folder, exist_ok=True)

        for sampled_img_id in sampled_ids:
            image_file = id_to_img[sampled_img_id]["file_name"]
            questions.append({
                "question_id": sampled_img_id,
                "image": image_file,
                "text": "Please describe this image in detail.",
            })
            src = os.path.join(args.image_folder, image_file)
            dst = os.path.join(dest_image_folder, image_file)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)

    elif args.dataset == "nocaps":
        val_caps = json.load(open(args.caption_file_path))
        image_infos = {x["id"]: x for x in val_caps["images"]}

        dest_image_folder = args.copy_image_folder or os.path.join(
            os.path.split(os.path.split(os.path.dirname(args.answers_file))[0])[0],
            "images",
            f"seed{args.seed}_{args.num_samples}"
        )
        os.makedirs(dest_image_folder, exist_ok=True)

        for image_id in sampled_ids:
            info = image_infos[image_id]
            image_file = info["file_name"]
            questions.append({
                "question_id": image_id,
                "image": image_file,
                "text": "Please describe this image in detail.",
            })
            src = os.path.join(args.image_folder, image_file)
            dst = os.path.join(dest_image_folder, f"{image_id}_{image_file}")
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    return questions


def merge_raw_intervention_bucket(dst, src):
    if not src:
        return dst
    for key, value in src.items():
        if key == "suppression_hist_counts":
            old = dst.get(key)
            if old is None or len(old) != len(value):
                old = [0 for _ in range(len(value))]
            for i, count in enumerate(value):
                old[i] += int(count)
            dst[key] = old
        elif key == "suppression_hist_bins":
            dst[key] = int(value)
        elif key.startswith("min_"):
            dst[key] = min(float(dst.get(key, float("inf"))), float(value))
        elif key.startswith("max_"):
            dst[key] = max(float(dst.get(key, float("-inf"))), float(value))
        elif isinstance(value, int):
            dst[key] = int(dst.get(key, 0)) + int(value)
        elif isinstance(value, float):
            dst[key] = float(dst.get(key, 0.0)) + float(value)
        else:
            dst[key] = value
    return dst


def _find_token_span(text_lower, token, start_pos):
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


def decode_prefix(tokenizer, token_ids):
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def generated_token_char_spans(tokenizer, output_token_ids):
    spans = []
    prev = ""
    for idx in range(len(output_token_ids)):
        cur = decode_prefix(tokenizer, output_token_ids[: idx + 1])
        start = len(prev)
        end = len(cur)
        if end > start:
            spans.append((idx, start, end))
        prev = cur
    return spans


def build_intervention_step_labels(
    tokenizer,
    output_token_ids,
    caption,
    question_id,
    image_file,
    chair,
    window,
):
    if chair is None or not caption:
        return {}
    try:
        import nltk
        chair_out = chair.compute_chair([{
            "image_id": int(question_id),
            "image": image_file,
            "caption": caption,
        }])
        sent = chair_out["sentences"][0]
        raw_tokens = nltk.word_tokenize(caption.lower())
    except Exception as exc:
        print(f"[intervention-stats] token labeling failed for image_id={question_id}: {exc}")
        return {}

    word_spans = align_word_token_spans(caption, raw_tokens)
    if word_spans is None:
        return {}
    token_spans = generated_token_char_spans(tokenizer, output_token_ids)
    if not token_spans:
        return {}

    def overlapping_steps(raw_idxs):
        steps = set()
        for raw_idx in raw_idxs:
            raw_idx = int(raw_idx)
            if raw_idx < 0 or raw_idx >= len(word_spans):
                continue
            start, end = word_spans[raw_idx]
            for step, tok_start, tok_end in token_spans:
                if tok_start < end and tok_end > start:
                    steps.add(step)
        return steps

    hall_exact = overlapping_steps(sent.get("hallucination_idxs", []))
    ground_exact = overlapping_steps(sent.get("non_hallucination_idxs", []))
    labels = {}

    def add_label(step, label):
        labels.setdefault(int(step), set()).add(label)

    for step in hall_exact:
        add_label(step, "hallucinated_exact")
    for step in ground_exact:
        add_label(step, "grounded_exact")

    window = max(int(window), 0)
    if window > 0:
        for source, label in ((hall_exact, "hallucinated_window"), (ground_exact, "grounded_window")):
            for step in source:
                for near in range(max(0, step - window), step + window + 1):
                    add_label(near, label)
                    add_label(near, "object_window")
    return labels


def update_intervention_token_bucket_stats(raw_stats, step_labels):
    if not raw_stats or "_current_sample_steps" not in raw_stats:
        return
    target = raw_stats.setdefault("by_token_bucket", {})
    for mode, by_step in raw_stats.get("_current_sample_steps", {}).items():
        mode_target = target.setdefault(mode, {})
        for step_text, bucket in by_step.items():
            labels = step_labels.get(int(step_text), None)
            if not labels:
                labels = {"other"}
            for label in sorted(labels):
                merge_raw_intervention_bucket(mode_target.setdefault(label, {}), bucket)
    raw_stats["_current_sample_steps"] = {}


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return input_ids, image_tensor, image.size, prompt

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, model_input_prompts = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, model_input_prompts


def create_data_loader(
    questions, image_folder, tokenizer, image_processor, model_config, conv_mode,
    batch_size=1, num_workers=4
):
    assert batch_size == 1
    dataset = CustomDataset(
        questions, image_folder, tokenizer, image_processor, model_config, conv_mode
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def attach_intervention_config(model, args):
    head_cfg = resolve_head_config(args)

    model.config.intervention = args.intervention
    model.config.intervention_heads = head_cfg["heads"]
    model.config.intervention_scores = head_cfg["scores"]
    model.config.img_start_pos = head_cfg["img_start_pos"]
    model.config.img_length = head_cfg["img_length"]

    model.config.text_threshold = args.text_threshold
    model.config.text_scale = args.text_scale
    model.config.dynamic_strength = args.dynamic_strength
    model.config.dynamic_ratio_power = args.dynamic_ratio_power
    model.config.dynamic_score_power = args.dynamic_score_power
    model.config.dynamic_tau = args.dynamic_tau
    model.config.dynamic_exp_sharpness = args.dynamic_exp_sharpness
    model.config.dynamic_late_boost_start = args.dynamic_late_boost_start
    model.config.dynamic_late_boost_end = args.dynamic_late_boost_end if args.dynamic_late_boost_end > 0 else args.max_new_tokens
    model.config.dynamic_late_boost_mode = args.dynamic_late_boost_mode
    model.config.dynamic_late_tau = args.dynamic_late_tau
    model.config.dynamic_context_mode = args.dynamic_context_mode
    model.config.dynamic_redistribute = args.dynamic_redistribute
    model.config.dynamic_renorm = args.dynamic_renorm
    model.config.use_head_scores = args.use_head_scores
    model.config.log_dynamic_trace = args.log_dynamic_trace
    model.config.dynamic_trace_topn = args.dynamic_trace_topn
    model.config.dynamic_trace_every = args.dynamic_trace_every
    model.config.intervention_stats_bins = args.intervention_stats_bins
    model.config.log_intervention_position_stats = args.log_intervention_position_stats
    model.config.log_intervention_token_bucket_stats = args.log_intervention_token_bucket_stats
    model.config.intervention_stats_device_accum = args.intervention_stats_device_accum
    model.config._dynamic_trace_step = 0
    model.config._dynamic_trace_buffer = []
    model.config.baseline_start_layer = args.baseline_start_layer
    model.config.baseline_end_layer = args.baseline_end_layer
    model.config.pai_alpha = args.pai_alpha
    model.config.pai_gamma = args.pai_gamma
    model.config.pai_use_cfg = args.pai_use_cfg
    model.config.pai_cfg_active = bool(args.pai_use_cfg)
    model.config.vaf_enh_para = args.vaf_enh_para
    model.config.vaf_sup_para = args.vaf_sup_para
    model.config.tarac_alpha = args.tarac_alpha
    model.config.tarac_beta = args.tarac_beta
    model.config.tarac_start_layer = args.tarac_start_layer
    model.config.tarac_end_layer = args.tarac_end_layer
    model.config._tarac_attn_memory = {}

    # backward compatibility
    model.config.hal_attention_heads = head_cfg["heads"]
    model.config.adhh_threshold = args.text_threshold
    model.config.adaptive_deactivate = (args.intervention == "adhh")

    return head_cfg

def save_run_config(args, head_cfg):
    run_cfg = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "resume": args.resume,
        "intervention": args.intervention,
        "head_source": args.head_source,
        "topk": args.topk,
        "text_threshold": args.text_threshold,
        "text_scale": args.text_scale,
        "dynamic_strength": args.dynamic_strength,
        "dynamic_ratio_power": args.dynamic_ratio_power,
        "dynamic_score_power": args.dynamic_score_power,
        "dynamic_tau": args.dynamic_tau,
        "dynamic_exp_sharpness": args.dynamic_exp_sharpness,
        "dynamic_late_boost_start": args.dynamic_late_boost_start,
        "dynamic_late_boost_end": args.dynamic_late_boost_end if args.dynamic_late_boost_end > 0 else args.max_new_tokens,
        "dynamic_late_boost_mode": args.dynamic_late_boost_mode,
        "dynamic_late_tau": args.dynamic_late_tau,
        "dynamic_context_mode": args.dynamic_context_mode,
        "dynamic_redistribute": args.dynamic_redistribute,
        "dynamic_renorm": args.dynamic_renorm,
        "use_head_scores": args.use_head_scores,
        "head_file": args.head_file,
        "head_score_key": args.head_score_key,
        "head_score_normalize": args.head_score_normalize,
        "min_head_back_raw": args.min_head_back_raw,
        "selected_head_count": len(head_cfg["heads"]),
        "baseline_start_layer": args.baseline_start_layer,
        "baseline_end_layer": args.baseline_end_layer,
        "pai_alpha": args.pai_alpha,
        "pai_gamma": args.pai_gamma,
        "pai_use_cfg": args.pai_use_cfg,
        "vaf_enh_para": args.vaf_enh_para,
        "vaf_sup_para": args.vaf_sup_para,
        "tarac_alpha": args.tarac_alpha,
        "tarac_beta": args.tarac_beta,
        "tarac_start_layer": args.tarac_start_layer,
        "tarac_end_layer": args.tarac_end_layer,
        "log_intervention_stats": args.log_intervention_stats,
        "intervention_stats_file": args.intervention_stats_file,
        "intervention_stats_bins": args.intervention_stats_bins,
        "log_intervention_position_stats": args.log_intervention_position_stats,
        "log_intervention_token_bucket_stats": args.log_intervention_token_bucket_stats,
        "intervention_stats_token_window": args.intervention_stats_token_window,
        "intervention_stats_device_accum": args.intervention_stats_device_accum,
        "output_attentions": args.output_attentions,
        "log_dynamic_trace": args.log_dynamic_trace,
        "dynamic_trace_topn": args.dynamic_trace_topn,
        "dynamic_trace_every": args.dynamic_trace_every,
        "selected_heads": head_cfg["heads"],
    }
    out_dir = os.path.dirname(os.path.expanduser(args.answers_file))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)


def quantiles_from_hist(counts, bins, quantiles=(0.25, 0.50, 0.75, 0.90, 0.95, 0.99)):
    if not counts or bins <= 0:
        return {}
    total = sum(int(x) for x in counts)
    if total <= 0:
        return {}
    out = {}
    cumulative = 0
    q_items = sorted((float(q), f"q{int(round(float(q) * 100)):02d}") for q in quantiles)
    q_idx = 0
    for idx, count in enumerate(counts):
        cumulative += int(count)
        while q_idx < len(q_items) and cumulative >= q_items[q_idx][0] * total:
            q, name = q_items[q_idx]
            out[name] = min(max(idx / float(bins), 0.0), 1.0)
            q_idx += 1
    for _, name in q_items:
        out.setdefault(name, 1.0)
    return out


def materialize_device_bucket(bucket):
    out = {}
    for key, value in bucket.items():
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.ndim == 0:
                value = float(value.item())
            else:
                value = [int(x) for x in value.tolist()]
        out[key] = value
    for key in ("count", "scaled_count", "near_zero_count", "saturation_count"):
        if key in out:
            out[key] = int(out[key])
    return out


def merge_materialized_device_stats(raw_stats):
    device_stats = (raw_stats or {}).pop("_device", None)
    if not device_stats:
        return raw_stats
    overall = raw_stats.setdefault("overall", {})
    for mode, bucket in device_stats.get("overall", {}).items():
        merge_raw_intervention_bucket(
            overall.setdefault(mode, {}),
            materialize_device_bucket(bucket),
        )
    by_position = raw_stats.setdefault("by_position", {})
    for mode, by_step in device_stats.get("by_position", {}).items():
        mode_out = by_position.setdefault(mode, {})
        for step, bucket in by_step.items():
            merge_raw_intervention_bucket(
                mode_out.setdefault(str(step), {}),
                materialize_device_bucket(bucket),
            )
    return raw_stats


def finalize_intervention_stats(raw_stats):
    raw_stats = merge_materialized_device_stats(raw_stats or {})

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
        out["saturation_rate"] = out.get("saturation_count", 0) / count
        hist = out.get("suppression_hist_counts")
        bins = int(out.get("suppression_hist_bins", 0) or 0)
        if hist and bins > 0:
            out["suppression_quantiles"] = quantiles_from_hist(hist, bins)
        return out

    if not raw_stats:
        return {"overall": {}, "by_head": {}, "by_position": {}, "by_token_bucket": {}}

    return {
        "overall": {
            mode: finalize_bucket(bucket)
            for mode, bucket in raw_stats.get("overall", {}).items()
        },
        "by_head": {
            key: finalize_bucket(bucket)
            for key, bucket in raw_stats.get("by_head", {}).items()
        },
        "by_position": {
            mode: {
                step: finalize_bucket(bucket)
                for step, bucket in by_step.items()
            }
            for mode, by_step in raw_stats.get("by_position", {}).items()
        },
        "by_token_bucket": {
            mode: {
                label: finalize_bucket(bucket)
                for label, bucket in by_label.items()
            }
            for mode, by_label in raw_stats.get("by_token_bucket", {}).items()
        },
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
        elif key == "suppression_hist_counts":
            old_hist = old_value or []
            new_hist = new_value or []
            size = max(len(old_hist), len(new_hist))
            merged[key] = [
                int(old_hist[i]) if i < len(old_hist) else 0
                for i in range(size)
            ]
            for i, value in enumerate(new_hist):
                merged[key][i] += int(value)
        elif key == "suppression_hist_bins":
            merged[key] = int(new_value or old_value or 0)
        elif key in ("scaled_count", "near_zero_count", "saturation_count"):
            merged[key] = int(old_value or 0) + int(new_value or 0)
        elif key in ("scaled_rate", "near_zero_rate", "saturation_rate", "suppression_quantiles"):
            continue
        elif new_value is not None:
            merged[key] = new_value
        elif old_value is not None:
            merged[key] = old_value

    merged["count"] = total_count
    merged["scaled_rate"] = merged.get("scaled_count", 0) / total_count
    merged["near_zero_rate"] = merged.get("near_zero_count", 0) / total_count
    merged["saturation_rate"] = merged.get("saturation_count", 0) / total_count
    hist = merged.get("suppression_hist_counts")
    bins = int(merged.get("suppression_hist_bins", 0) or 0)
    if hist and bins > 0:
        merged["suppression_quantiles"] = quantiles_from_hist(hist, bins)
    return merged


def merge_intervention_stats(old_stats, new_stats):
    if not old_stats:
        return new_stats
    if not new_stats or total_intervention_count(new_stats) == 0:
        return old_stats

    merged = {"overall": {}, "by_head": {}, "by_position": {}, "by_token_bucket": {}}
    for section in ("overall", "by_head"):
        old_section = old_stats.get(section, {})
        new_section = new_stats.get(section, {})
        for key in sorted(set(old_section) | set(new_section)):
            merged[section][key] = merge_intervention_bucket(
                old_section.get(key),
                new_section.get(key),
            )
    for section in ("by_position", "by_token_bucket"):
        old_section = old_stats.get(section, {})
        new_section = new_stats.get(section, {})
        for mode in sorted(set(old_section) | set(new_section)):
            merged[section][mode] = {}
            old_nested = old_section.get(mode, {})
            new_nested = new_section.get(mode, {})
            for key in sorted(set(old_nested) | set(new_nested)):
                merged[section][mode][key] = merge_intervention_bucket(
                    old_nested.get(key),
                    new_nested.get(key),
                )
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
    merged_count = total_intervention_count(merged)
    print(
        f"[resume] merged intervention stats at {out_file}: "
        f"existing_count={existing_count}, new_count={new_count}, "
        f"merged_count={merged_count}"
    )
    return merged


def save_intervention_stats(model, args):
    if not args.log_intervention_stats:
        return

    out_file = args.intervention_stats_file
    if not out_file:
        out_file = os.path.join(os.path.dirname(os.path.expanduser(args.answers_file)), "intervention_stats.json")
    out_file = os.path.expanduser(out_file)
    if getattr(args, "resume", False) and getattr(args, "_resume_remaining_questions", 1) == 0 and os.path.exists(out_file):
        print(f"[resume] no remaining questions; keeping existing intervention stats at {out_file}")
        return

    stats = finalize_intervention_stats(getattr(model.config, "_intervention_stats", None))
    stats = maybe_merge_resume_intervention_stats(stats, out_file, args)
    stats["config"] = {
        "intervention": args.intervention,
        "topk": args.topk,
        "text_threshold": args.text_threshold,
        "dynamic_strength": args.dynamic_strength,
        "dynamic_ratio_power": args.dynamic_ratio_power,
        "dynamic_score_power": args.dynamic_score_power,
        "dynamic_tau": args.dynamic_tau,
        "dynamic_exp_sharpness": args.dynamic_exp_sharpness,
        "dynamic_late_boost_start": args.dynamic_late_boost_start,
        "dynamic_late_boost_end": args.dynamic_late_boost_end if args.dynamic_late_boost_end > 0 else args.max_new_tokens,
        "dynamic_late_boost_mode": args.dynamic_late_boost_mode,
        "dynamic_late_tau": args.dynamic_late_tau,
        "dynamic_context_mode": args.dynamic_context_mode,
        "dynamic_redistribute": args.dynamic_redistribute,
        "dynamic_renorm": args.dynamic_renorm,
        "use_head_scores": args.use_head_scores,
        "head_file": args.head_file,
        "head_score_key": args.head_score_key,
        "head_score_normalize": args.head_score_normalize,
        "min_head_back_raw": args.min_head_back_raw,
        "resume": args.resume,
        "log_dynamic_trace": args.log_dynamic_trace,
        "dynamic_trace_topn": args.dynamic_trace_topn,
        "dynamic_trace_every": args.dynamic_trace_every,
        "intervention_stats_bins": args.intervention_stats_bins,
        "log_intervention_position_stats": args.log_intervention_position_stats,
        "log_intervention_token_bucket_stats": args.log_intervention_token_bucket_stats,
        "intervention_stats_token_window": args.intervention_stats_token_window,
        "intervention_stats_device_accum": args.intervention_stats_device_accum,
        "baseline_start_layer": args.baseline_start_layer,
        "baseline_end_layer": args.baseline_end_layer,
        "pai_alpha": args.pai_alpha,
        "pai_gamma": args.pai_gamma,
        "pai_use_cfg": args.pai_use_cfg,
        "vaf_enh_para": args.vaf_enh_para,
        "vaf_sup_para": args.vaf_sup_para,
        "tarac_alpha": args.tarac_alpha,
        "tarac_beta": args.tarac_beta,
        "tarac_start_layer": args.tarac_start_layer,
        "tarac_end_layer": args.tarac_end_layer,
    }

    with open(out_file, "w") as f:
        json.dump(stats, f, indent=2)


def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    sampled_ids = load_or_sample_ids(args)
    questions = build_questions(args, sampled_ids)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    total_chunk_questions = len(questions)
    completed_question_ids = load_completed_question_ids(answers_file) if args.resume else set()
    if completed_question_ids:
        questions = [q for q in questions if int(q["question_id"]) not in completed_question_ids]
        print(
            f"[resume] {len(completed_question_ids)} completed answers found in {answers_file}; "
            f"running {len(questions)}/{total_chunk_questions} remaining questions for chunk {args.chunk_idx}."
        )
    args._resume_remaining_questions = len(questions)
    ans_mode = "a" if args.resume else "w"
    ans_file = open(answers_file, ans_mode, encoding="utf-8")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode,
        batch_size=1, num_workers=args.num_workers
    )

    head_cfg = attach_intervention_config(model, args)
    save_run_config(args, head_cfg)
    model.config.log_intervention_stats = args.log_intervention_stats
    model.config._intervention_stats = {
        "overall": {},
        "by_head": {},
        "by_position": {},
        "by_token_bucket": {},
    }

    chair_for_token_stats = None
    if args.log_intervention_stats and args.log_intervention_token_bucket_stats:
        if args.dataset != "coco":
            print("[intervention-stats] token bucket stats are currently implemented for COCO/CHAIR only.")
        else:
            from eval_scripts.eval_utils.chair import CHAIR
            chair_for_token_stats = CHAIR(
                [int(q["question_id"]) for q in questions],
                os.path.dirname(os.path.abspath(args.caption_file_path)),
            )

    for (input_ids, image_tensor, image_sizes, model_input_prompts), line in tqdm(
        zip(data_loader, questions),
        total=len(questions)
    ):
        question_id = line["question_id"]
        cur_prompt = line["text"]
        image_file = line["image"]
        model_input_prompt = model_input_prompts[0]

        model.config.dynamic_trace_sample_id = question_id
        model.config._dynamic_trace_step = 0
        model.config._dynamic_trace_buffer = []
        model.config._tarac_attn_memory = {}

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

        with torch.inference_mode():
            logits_processor = None
            if args.intervention == "pai" and args.pai_use_cfg:
                logits_processor = LogitsProcessorList([
                    PAITextCFGLogitsProcessor(
                        args.pai_gamma,
                        build_text_only_input_ids(input_ids),
                        model,
                    )
                ])

            output_dict = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_attentions=args.output_attentions,
                return_dict_in_generate=True,
                logits_processor=logits_processor,
            )

        output_ids = output_dict["sequences"]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        raw_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
        output_token_ids = output_ids[0].detach().cpu().tolist()
        if args.log_intervention_stats and args.log_intervention_token_bucket_stats:
            step_labels = build_intervention_step_labels(
                tokenizer=tokenizer,
                output_token_ids=output_token_ids,
                caption=outputs,
                question_id=question_id,
                image_file=image_file,
                chair=chair_for_token_stats,
                window=args.intervention_stats_token_window,
            )
            update_intervention_token_bucket_stats(
                getattr(model.config, "_intervention_stats", None),
                step_labels,
            )
        if not args.quiet:
            print(question_id, outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": question_id,
            "image": image_file,
            "prompt": cur_prompt,
            "text": outputs,
            "text_raw": raw_outputs,
            "output_token_ids": output_token_ids,
            "output_token_count": len(output_token_ids),
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {
                "intervention": args.intervention,
                "topk": args.topk,
                "baseline_start_layer": args.baseline_start_layer,
                "baseline_end_layer": args.baseline_end_layer,
                "pai_alpha": args.pai_alpha,
                "pai_gamma": args.pai_gamma,
                "pai_use_cfg": args.pai_use_cfg,
                "vaf_enh_para": args.vaf_enh_para,
                "vaf_sup_para": args.vaf_sup_para,
                "model_input_prompt": model_input_prompt,
            }
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    save_intervention_stats(model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--caption_file_path", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--copy-image-folder", type=str, default=None,
                        help="Optional shared folder for copied/evaluation images. If omitted, images are copied near --answers-file.")
    parser.add_argument("--resume", action="store_true", help="Append to an existing answers file and skip already completed question_ids.")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-attentions", action="store_true",
                        help="Return full attention tensors from generate. Not needed for DEACT intervention or intervention stats.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print generated captions to stdout/decode.log.")

    parser.add_argument("--intervention", type=str, default="dynamic",
                        choices=["none", "dynamic", "late_boost", "linear", "exp", "threshold_exp", "center_exp", "linear_tail_exp", "pai", "vaf", "tarac"])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--baseline-start-layer", type=int, default=0,
                        help="Inclusive first decoder layer for PAI/VAF pre-softmax visual intervention.")
    parser.add_argument("--baseline-end-layer", type=int, default=32,
                        help="Exclusive final decoder layer for PAI/VAF pre-softmax visual intervention.")
    parser.add_argument("--pai-alpha", type=float, default=0.2,
                        help="PAI image-token attention-logit amplification coefficient.")
    parser.add_argument("--pai-gamma", type=float, default=1.1,
                        help="PAI text-only CFG/logits-refine guidance scale.")
    parser.add_argument("--pai-use-cfg", action="store_true",
                        help="Enable PAI text-only logits refine, matching the official CHAIR command.")
    parser.add_argument("--vaf-enh-para", type=float, default=1.15,
                        help="VAF image-token attention-logit multiplier.")
    parser.add_argument("--vaf-sup-para", type=float, default=0.95,
                        help="VAF system-token attention-logit multiplier.")
    parser.add_argument("--tarac-alpha", type=float, default=0.5,
                        help="TARAC memory update factor alpha.")
    parser.add_argument("--tarac-beta", type=float, default=0.5,
                        help="TARAC accumulated image-attention injection coefficient beta.")
    parser.add_argument("--tarac-start-layer", type=int, default=9,
                        help="Inclusive first decoder layer for TARAC.")
    parser.add_argument("--tarac-end-layer", type=int, default=16,
                        help="Exclusive final decoder layer for TARAC.")

    parser.add_argument("--dynamic-strength", type=float, default=1.0,
                        help="Maximum continuous suppression strength before clipping.")
    parser.add_argument("--dynamic-ratio-power", type=float, default=1.0,
                        help="Power applied to text/(text+image) reliance; 0 disables context modulation.")
    parser.add_argument("--dynamic-score-power", type=float, default=1.0,
                        help="Power applied to normalized offline head scores.")
    parser.add_argument("--dynamic-context-mode", type=str, default="ratio_exp",
                        choices=["text_exp", "ratio_exp", "ratio_power", "text_power"])
    parser.add_argument("--dynamic-tau", type=float, default=0.845,
                        help="Center point for exponential dynamic context modes.")
    parser.add_argument("--dynamic-exp-sharpness", type=float, default=8.0,
                        help="Sharpness k in exp(k * (context - tau)).")
    parser.add_argument("--dynamic-late-boost-start", type=int, default=0,
                        help="If >= 0, begin late-step tau scheduling from this generated-token step.")
    parser.add_argument("--dynamic-late-boost-end", type=int, default=128,
                        help="Generated-token step where linear late tau reaches --dynamic-late-tau; <=0 uses --max_new_tokens.")
    parser.add_argument("--dynamic-late-boost-mode", type=str, default="linear", choices=["linear", "step"],
                        help="Late tau schedule: linear decays from base tau to late tau, step switches immediately.")
    parser.add_argument("--dynamic-late-tau", type=float, default=0.80,
                        help="Final late-step tau for late_boost; use a negative value to disable scheduling.")
    parser.add_argument("--dynamic-redistribute", type=str, default="renorm",
                        choices=["none", "renorm", "system", "system_only", "vision", "vision_only", "sysvis"],
                        help="Where to explicitly move the text attention mass removed by dynamic. renorm keeps the original row-renormalization behavior.")
    parser.add_argument("--no-dynamic-renorm", dest="dynamic_renorm", action="store_false",
                        help="Skip the final attention-row renormalization after dynamic text suppression.")
    parser.set_defaults(dynamic_renorm=True)
    parser.add_argument("--use-head-scores", action="store_true")
    parser.add_argument("--log-dynamic-trace", action="store_true",
                        help="Print per-decoding-step dynamic suppression summaries to decode.log.")
    parser.add_argument("--dynamic-trace-topn", type=int, default=10,
                        help="Number of most suppressed heads to include in each dynamic trace line.")
    parser.add_argument("--dynamic-trace-every", type=int, default=1,
                        help="Log one dynamic trace line every N generated steps.")

    # Kept hidden so old run_config/stat helpers and imported utilities remain well-defined.
    parser.add_argument("--text-threshold", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--text-scale", type=float, default=0.5, help=argparse.SUPPRESS)
    parser.add_argument("--gate-scale", type=float, default=1.0, help=argparse.SUPPRESS)

    parser.add_argument("--sample-id-file", type=str, default="")
    parser.add_argument("--save-sample-id-file", type=str, default="")

    parser.add_argument("--head-source", type=str, default="default",
                    choices=["default", "file"])
    parser.add_argument("--head-file", type=str, default="")
    parser.add_argument("--head-score-key", type=str, default="score")
    parser.add_argument("--head-score-normalize", type=str, default="minmax",
                        choices=["minmax", "raw", "logminmax", "rank_percentile"])
    parser.add_argument("--min-head-back-raw", type=float, default=0.0,
                        help="For ranked head files with a back_raw field, drop heads inside the requested top-k whose back_raw is below this value.")
    parser.add_argument("--log-intervention-stats", action="store_true")
    parser.add_argument("--intervention-stats-file", type=str, default="")
    parser.add_argument("--intervention-stats-bins", type=int, default=100,
                        help="Histogram bins for suppression-delta quantile estimates in intervention_stats.json.")
    parser.add_argument("--log-intervention-position-stats", action="store_true",
                        help="Aggregate intervention stats by generated-token position.")
    parser.add_argument("--log-intervention-token-bucket-stats", action="store_true",
                        help="Aggregate intervention stats around CHAIR-counted grounded/hallucinated object tokens.")
    parser.add_argument("--intervention-stats-token-window", type=int, default=2,
                        help="Token window on either side of CHAIR-counted object tokens for token-bucket intervention stats.")
    parser.add_argument("--intervention-stats-device-accum", action="store_true",
                        help="Accumulate overall/position intervention stats on device and materialize at save time. This is much faster but does not collect by-head or token-bucket stats.")

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
