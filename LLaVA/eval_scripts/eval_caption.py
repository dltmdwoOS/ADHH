import argparse
import json
import math
import os
import random
import shutil

import shortuuid
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import set_seed

from eval_scripts.eval_utils.chair import CHAIR
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


ANALYSIS_BUCKETS = (
    "all",
    "object",
    "hallucinated",
    "non_hallucinated",
    "pre_object",
    "pre_hallucinated",
    "pre_non_hallucinated",
)
ANALYSIS_METRICS = (
    "sys_attn",
    "txt_attn",
    "image_attn",
    "txt_img_ratio",
    "sys_img_ratio",
    "txt_sys_ratio",
    "entropy",
)

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



def load_txtattn_heads(head_file, topk):
    with open(os.path.expanduser(head_file), "r") as f:
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

    selected_records = records if topk is None or int(topk) <= 0 else records[:topk]

    heads = []
    for item in selected_records:
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

def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def safe_token_label(tokenizer, token_id):
    if token_id == IMAGE_TOKEN_INDEX:
        return "<image>"
    if token_id < 0:
        return f"<special:{token_id}>"
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return f"<id:{int(token_id)}>"


def extract_generated_ids(output_ids, input_ids):
    prompt_len = int(input_ids.shape[1])
    output_len = int(output_ids.shape[1])
    if output_len >= prompt_len and torch.equal(output_ids[:, :prompt_len], input_ids):
        return output_ids[:, prompt_len:]
    return output_ids


def get_special_token_ids(tokenizer):
    special_ids = set()
    for attr in ("all_special_ids",):
        for token_id in getattr(tokenizer, attr, []) or []:
            if token_id is not None:
                special_ids.add(int(token_id))
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            special_ids.add(int(token_id))
    return special_ids


def infer_step_masks(input_ids, generated_prefix_ids, att_seq_len, special_token_ids):
    prompt_ids = input_ids[0].detach().cpu().tolist()
    generated_prefix_ids = [int(token_id) for token_id in generated_prefix_ids]
    generated_prefix_len = len(generated_prefix_ids)
    image_positions = [idx for idx, token_id in enumerate(prompt_ids) if token_id == IMAGE_TOKEN_INDEX]
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
        raise ValueError(
            f"Invalid image span: image_start={image_start}, image_end={image_end}, att_seq_len={att_seq_len}"
        )

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
        "generated_special_len": int(
            sum(1 for token_id in generated_prefix_ids[:available_generated_len] if token_id in special_token_ids)
        ),
    }
    return sys_mask, txt_mask, image_mask, layout


class RunningHeadStats:
    def __init__(self):
        self.count = None
        self.sum = {}
        self.sumsq = {}

    def _ensure(self, values):
        if self.count is not None:
            return
        shape = values.shape
        self.count = torch.zeros(shape, dtype=torch.float64)
        for metric in ANALYSIS_METRICS:
            self.sum[metric] = torch.zeros(shape, dtype=torch.float64)
            self.sumsq[metric] = torch.zeros(shape, dtype=torch.float64)

    def update(self, values_by_metric):
        base = values_by_metric["sys_attn"].detach().cpu().double()
        self._ensure(base)
        self.count += 1
        for metric in ANALYSIS_METRICS:
            values = values_by_metric[metric].detach().cpu().double()
            self.sum[metric] += values
            self.sumsq[metric] += values * values

    def to_dict(self):
        if self.count is None:
            return {
                "count": 0,
                "num_layers": 0,
                "num_heads": 0,
                "metrics": {},
            }

        count = self.count.clamp_min(1.0)
        metrics = {}
        for metric in ANALYSIS_METRICS:
            mean = self.sum[metric] / count
            var = (self.sumsq[metric] / count - mean * mean).clamp_min(0.0)
            metrics[metric] = {
                "mean": mean.tolist(),
                "var": var.tolist(),
                "mean_over_heads": float(mean.mean().item()),
                "var_over_heads": float(mean.var(unbiased=False).item()),
            }

        return {
            "count": int(self.count.max().item()),
            "num_layers": int(self.count.shape[0]),
            "num_heads": int(self.count.shape[1]),
            "metrics": metrics,
        }


class AttentionAnalysisAccumulator:
    def __init__(self):
        self.buckets = {bucket: RunningHeadStats() for bucket in ANALYSIS_BUCKETS}
        self.samples = []

    def update(self, bucket, values_by_metric):
        self.buckets[bucket].update(values_by_metric)

    def add_sample(self, sample):
        self.samples.append(sample)

    def to_dict(self, include_samples=True):
        result = {
            "buckets": {bucket: stats.to_dict() for bucket, stats in self.buckets.items()},
        }
        if include_samples:
            result["samples"] = self.samples
        return result


def classify_generation_steps(tokenizer, generated_ids_only, chair_evaluator, gt_objects):
    labels = []
    previous_object_count = 0
    token_ids = generated_ids_only[0].detach().cpu().tolist()

    for step_idx in range(len(token_ids)):
        prefix_ids = token_ids[: step_idx + 1]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=True)
        token_text = safe_token_label(tokenizer, token_ids[step_idx])
        objects = []
        hallucinated = []
        non_hallucinated = []

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
                "token_id": int(token_ids[step_idx]),
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


def compute_step_head_values(step_attentions, input_ids, generated_prefix_ids, special_token_ids):
    eps = 1e-12
    att_seq_len = int(step_attentions[0].shape[-1])
    sys_mask, txt_mask, image_mask, layout = infer_step_masks(
        input_ids,
        generated_prefix_ids,
        att_seq_len,
        special_token_ids,
    )
    layer_sys, layer_txt, layer_image = [], [], []
    layer_txt_img_ratio, layer_sys_img_ratio, layer_txt_sys_ratio, layer_entropy = [], [], [], []

    for att in step_attentions:
        att_cpu = att.detach().cpu()
        if att_cpu.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {att_cpu.shape[0]}")
        q_idx = int(att_cpu.shape[-2] - 1)
        rows = att_cpu[0, :, q_idx, :].float()
        row_sums = rows.sum(dim=-1, keepdim=True)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
            rows = torch.softmax(rows, dim=-1)

        sys_attn = rows[:, sys_mask].sum(dim=-1)
        txt_attn = rows[:, txt_mask].sum(dim=-1)
        image_attn = rows[:, image_mask].sum(dim=-1)
        txt_img_ratio = txt_attn / (image_attn + eps)
        sys_img_ratio = sys_attn / (image_attn + eps)
        txt_sys_ratio = txt_attn / (sys_attn + eps)
        probs = rows.clamp_min(eps)
        entropy = -(probs * probs.log()).sum(dim=-1)

        layer_sys.append(sys_attn)
        layer_txt.append(txt_attn)
        layer_image.append(image_attn)
        layer_txt_img_ratio.append(txt_img_ratio)
        layer_sys_img_ratio.append(sys_img_ratio)
        layer_txt_sys_ratio.append(txt_sys_ratio)
        layer_entropy.append(entropy)

    return {
        "sys_attn": torch.stack(layer_sys, dim=0),
        "txt_attn": torch.stack(layer_txt, dim=0),
        "image_attn": torch.stack(layer_image, dim=0),
        "txt_img_ratio": torch.stack(layer_txt_img_ratio, dim=0),
        "sys_img_ratio": torch.stack(layer_sys_img_ratio, dim=0),
        "txt_sys_ratio": torch.stack(layer_txt_sys_ratio, dim=0),
        "entropy": torch.stack(layer_entropy, dim=0),
    }, layout


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
        )
        image_end = int(layout["image_end"])
        label = step_labels[step_idx]
        buckets = ["all"]
        if label["is_object"]:
            buckets.append("object")
        if label["is_hallucinated"]:
            buckets.append("hallucinated")
        if label["is_non_hallucinated"]:
            buckets.append("non_hallucinated")

        head_values = []
        for layer_idx, head_idx in heads:
            att = step_attentions[int(layer_idx)].detach().cpu()
            if att.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {att.shape[0]}")
            q_idx = int(att.shape[-2] - 1)
            row = att[0, int(head_idx), q_idx, :].float()
            row_sum = row.sum()
            if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-3):
                row = torch.softmax(row, dim=-1)

            i_text = float(row[image_end:].sum().item())
            generated_txt_attn = float(row[txt_mask].sum().item())
            image_attn = float(row[image_mask].sum().item())
            head_values.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head_idx),
                    "I_text": i_text,
                    "generated_txt_attn": generated_txt_attn,
                    "image_attn": image_attn,
                    "txt_img_ratio": generated_txt_attn / (image_attn + eps),
                }
            )

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

    return {
        "num_steps": int(num_steps),
        "num_records": int(num_records),
        "trace_note": "I_text matches the intervention slice: sum attention over positions image_end: for each selected head.",
    }


def build_token_mask_debug(tokenizer, input_ids, generated_ids_only, image_len, special_token_ids):
    prompt_ids = input_ids[0].detach().cpu().tolist()
    generated_ids = generated_ids_only[0].detach().cpu().tolist()
    image_positions = [idx for idx, token_id in enumerate(prompt_ids) if token_id == IMAGE_TOKEN_INDEX]
    if len(image_positions) != 1:
        raise ValueError(f"Expected exactly one image token, found {len(image_positions)}")

    image_start = image_positions[0]
    image_end = image_start + int(image_len)
    prompt_after_image_len = len(prompt_ids) - image_start - 1
    generated_start = image_end + prompt_after_image_len
    final_att_seq_len = generated_start + len(generated_ids)

    records = []
    for att_pos in range(final_att_seq_len):
        if att_pos < image_start:
            prompt_idx = att_pos
            token_id = int(prompt_ids[prompt_idx])
            records.append(
                {
                    "att_pos": int(att_pos),
                    "mask": "sys",
                    "source": "prompt",
                    "prompt_idx": int(prompt_idx),
                    "token_id": token_id,
                    "token_text": safe_token_label(tokenizer, token_id),
                }
            )
        elif att_pos < image_end:
            patch_idx = att_pos - image_start
            records.append(
                {
                    "att_pos": int(att_pos),
                    "mask": "img",
                    "source": "image",
                    "image_patch_idx": int(patch_idx),
                    "token_id": IMAGE_TOKEN_INDEX,
                    "token_text": f"<image_patch:{patch_idx}>",
                }
            )
        elif att_pos < generated_start:
            prompt_idx = image_start + 1 + (att_pos - image_end)
            token_id = int(prompt_ids[prompt_idx])
            records.append(
                {
                    "att_pos": int(att_pos),
                    "mask": "sys",
                    "source": "prompt",
                    "prompt_idx": int(prompt_idx),
                    "token_id": token_id,
                    "token_text": safe_token_label(tokenizer, token_id),
                }
            )
        else:
            generated_idx = att_pos - generated_start
            token_id = int(generated_ids[generated_idx])
            is_special = token_id in special_token_ids
            records.append(
                {
                    "att_pos": int(att_pos),
                    "mask": "sys" if is_special else "txt",
                    "source": "generated",
                    "generated_idx": int(generated_idx),
                    "is_special": bool(is_special),
                    "token_id": token_id,
                    "token_text": safe_token_label(tokenizer, token_id),
                }
            )

    generated_special_len = sum(1 for token_id in generated_ids if int(token_id) in special_token_ids)
    return {
        "layout": {
            "prompt_visible_len": int(len(prompt_ids)),
            "generated_len": int(len(generated_ids)),
            "final_att_seq_len": int(final_att_seq_len),
            "image_start": int(image_start),
            "image_end": int(image_end),
            "image_len": int(image_len),
            "generated_start": int(generated_start),
            "sys_len": int(len(prompt_ids) - 1 + generated_special_len),
            "txt_len": int(len(generated_ids) - generated_special_len),
            "generated_special_len": int(generated_special_len),
        },
        "prompt_text": tokenizer.decode([x for x in prompt_ids if x >= 0], skip_special_tokens=False),
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "records": records,
    }


def summarize_attention_for_sample(
    question_id,
    input_ids,
    generated_ids_only,
    attentions,
    step_labels,
    global_acc,
    special_token_ids,
    enable_pre_token_analysis=False,
):
    sample_acc = AttentionAnalysisAccumulator()
    token_records = []
    layouts = []

    if attentions is None:
        return sample_acc.to_dict(include_samples=False)

    num_steps = min(len(attentions), int(generated_ids_only.shape[1]), len(step_labels))
    generated_ids_list = generated_ids_only[0].detach().cpu().tolist()
    for step_idx in range(num_steps):
        values_by_metric, layout = compute_step_head_values(
            attentions[step_idx],
            input_ids,
            generated_ids_list[:step_idx],
            special_token_ids,
        )
        label = step_labels[step_idx]
        buckets = ["all"]
        if label["is_object"]:
            buckets.append("object")
        if label["is_hallucinated"]:
            buckets.append("hallucinated")
        if label["is_non_hallucinated"]:
            buckets.append("non_hallucinated")

        pre_token_buckets = []
        if enable_pre_token_analysis:
            # In HF generation, attentions[step_idx] are computed from the prefix
            # before generated_ids[step_idx] is appended, so these are genuinely
            # pre-token attentions for the current generated token label.
            if label["is_object"]:
                pre_token_buckets.append("pre_object")
            if label["is_hallucinated"]:
                pre_token_buckets.append("pre_hallucinated")
            if label["is_non_hallucinated"]:
                pre_token_buckets.append("pre_non_hallucinated")

        for bucket in buckets + pre_token_buckets:
            sample_acc.update(bucket, values_by_metric)
            global_acc.update(bucket, values_by_metric)

        token_records.append(
            {
                "step_idx": int(step_idx),
                "token_id": label["token_id"],
                "token_text": label["token_text"],
                "buckets": buckets,
                "pre_token_buckets": pre_token_buckets,
                "objects": label["objects"],
                "hallucinated_objects": label["hallucinated_objects"],
                "non_hallucinated_objects": label["non_hallucinated_objects"],
            }
        )
        layouts.append(layout)

    summary = sample_acc.to_dict(include_samples=False)
    summary.update(
        {
            "question_id": int(question_id),
            "num_generation_steps": int(num_steps),
            "num_object_steps": int(sum(1 for rec in token_records if "object" in rec["buckets"])),
            "num_hallucinated_steps": int(sum(1 for rec in token_records if "hallucinated" in rec["buckets"])),
            "num_non_hallucinated_steps": int(sum(1 for rec in token_records if "non_hallucinated" in rec["buckets"])),
            "num_pre_object_steps": int(sum(1 for rec in token_records if "pre_object" in rec["pre_token_buckets"])),
            "num_pre_hallucinated_steps": int(sum(1 for rec in token_records if "pre_hallucinated" in rec["pre_token_buckets"])),
            "num_pre_non_hallucinated_steps": int(sum(1 for rec in token_records if "pre_non_hallucinated" in rec["pre_token_buckets"])),
            "pre_token_attention_note": "attentions[step_idx] are prefix attentions used to predict generated_ids[step_idx] before that token is appended",
            "tokens": token_records,
            "layout_first_step": layouts[0] if layouts else None,
            "layout_last_step": layouts[-1] if layouts else None,
        }
    )
    return summary


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
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(
    questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )


def build_questions_from_existing_sample_file(args):
    sample_path = os.path.expanduser(args.existing_sample_file)
    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        entries = data.get("sentences")
        if entries is None:
            entries = data.get("samples")
    elif isinstance(data, list):
        entries = data
    else:
        entries = None

    if not entries:
        raise ValueError(f"No sample entries found in existing sample file: {sample_path}")

    questions = []
    sampled_meta = []
    for idx, entry in enumerate(entries):
        question_id = entry.get("question_id", entry.get("image_id"))
        image_file = entry.get("image")
        if question_id is None or image_file is None:
            raise ValueError(
                f"Entry {idx} in {sample_path} must contain image_id/question_id and image fields: {entry}"
            )
        questions.append(
            {
                "question_id": int(question_id),
                "image": image_file,
                "text": args.prompt_text,
            }
        )
        sampled_meta.append(
            {
                "question_id": int(question_id),
                "image": image_file,
                "prompt": args.prompt_text,
                "source_index": int(idx),
                "source_file": sample_path,
            }
        )

    if args.save_sample_ids:
        os.makedirs(os.path.dirname(args.save_sample_ids), exist_ok=True)
        with open(args.save_sample_ids, "w") as f:
            json.dump(sampled_meta, f, indent=2)

    print(f"Loaded {len(questions)} existing samples from {sample_path}")
    return questions


def build_questions(args):
    if args.dataset != "coco":
        raise ValueError("attention analysis path currently supports --dataset coco")

    if args.use_existing_sample_file:
        if not args.existing_sample_file:
            raise ValueError("--existing-sample-file is required when --use-existing-sample-file is set")
        return build_questions_from_existing_sample_file(args)

    coco = COCO(args.caption_file_path)
    img_ids = coco.getImgIds()
    sampled_img_ids = random.sample(img_ids, args.num_samples)
    questions = []
    sampled_meta = []
    dest_image_folder = os.path.join(
        os.path.split(os.path.split(os.path.dirname(args.answers_file))[0])[0],
        "images",
        f"seed{args.seed}_{args.num_samples}",
    )
    os.makedirs(dest_image_folder, exist_ok=True)

    for sampled_img_id in sampled_img_ids:
        image_file = coco.loadImgs(sampled_img_id)[0]["file_name"]
        questions.append(
            {
                "question_id": sampled_img_id,
                "image": image_file,
                "text": args.prompt_text,
            }
        )
        src = os.path.join(args.image_folder, image_file)
        dst = os.path.join(dest_image_folder, image_file)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
        sampled_meta.append(
            {
                "question_id": sampled_img_id,
                "image": image_file,
                "prompt": args.prompt_text,
            }
        )

    if args.save_sample_ids:
        os.makedirs(os.path.dirname(args.save_sample_ids), exist_ok=True)
        with open(args.save_sample_ids, "w") as f:
            json.dump(sampled_meta, f, indent=2)

    return questions


def build_chair_evaluator(args, questions):
    if not (args.enable_attention_analysis or args.enable_txtattn_trace):
        return None
    annotation_dir = args.annotation_dir or os.path.dirname(args.caption_file_path)
    evaluator = CHAIR([q["question_id"] for q in questions], annotation_dir)
    evaluator.get_annotations()
    return evaluator


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    questions = build_questions(args)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"Auto switching conversation mode to {args.conv_mode}.")

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        args.conv_mode,
        num_workers=args.num_workers,
    )

    chair_evaluator = build_chair_evaluator(args, questions)
    global_acc = AttentionAnalysisAccumulator()
    special_token_ids = get_special_token_ids(tokenizer)

    txtattn_heads = []
    txtattn_stats = None
    txtattn_writer = None
    if args.enable_txtattn_trace:
        if not args.txtattn_head_file:
            raise ValueError("--enable-txtattn-trace requires --txtattn-head-file")
        txtattn_heads = load_txtattn_heads(args.txtattn_head_file, args.txtattn_topk)
        txtattn_output_file = args.txtattn_output_file or os.path.join(
            os.path.dirname(answers_file), "txtattn_trace.jsonl"
        )
        os.makedirs(os.path.dirname(os.path.expanduser(txtattn_output_file)), exist_ok=True)
        txtattn_writer = open(os.path.expanduser(txtattn_output_file), "w", encoding="utf-8")
        txtattn_stats = TxtAttnTraceStats(txtattn_heads)
        print(f"[txtattn] tracing {len(txtattn_heads)} heads -> {txtattn_output_file}")

    sample_dir = None
    mask_debug_dir = None
    if args.enable_attention_analysis and args.output_path:
        sample_dir = os.path.join(args.output_path, "samples")
        os.makedirs(sample_dir, exist_ok=True)
        if args.debug_token_masks:
            mask_debug_dir = os.path.join(args.output_path, "token_mask_debug")
            os.makedirs(mask_debug_dir, exist_ok=True)

    for sample_idx, ((input_ids, image_tensor, image_sizes), line) in tqdm(
        enumerate(zip(data_loader, questions), start=1),
        total=len(questions),
    ):
        question_id = line["question_id"]
        cur_prompt = line["text"]
        image_file = line["image"]

        input_ids = input_ids.to(device="cuda", non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

        with torch.inference_mode():
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
                output_attentions=(args.enable_attention_analysis or args.enable_txtattn_trace),
                output_scores=False,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )

        output_ids = output_dict["sequences"]
        generated_ids_only = extract_generated_ids(output_ids, input_ids)
        outputs = tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)[0].strip()
        print(f"[{sample_idx}/{len(questions)}] question_id={question_id}")
        print(outputs)

        metadata = {}
        step_labels = None
        gt_objects = set()
        if args.enable_attention_analysis or args.enable_txtattn_trace:
            gt_objects = set(chair_evaluator.imid_to_objects.get(question_id, set())) if chair_evaluator else set()
            step_labels = classify_generation_steps(tokenizer, generated_ids_only, chair_evaluator, gt_objects)

        if args.enable_attention_analysis:
            sample_summary = summarize_attention_for_sample(
                question_id=question_id,
                input_ids=input_ids,
                generated_ids_only=generated_ids_only,
                attentions=output_dict.attentions,
                step_labels=step_labels,
                global_acc=global_acc,
                special_token_ids=special_token_ids,
                enable_pre_token_analysis=args.enable_pre_token_analysis,
            )
            sample_summary["image"] = image_file
            sample_summary["caption"] = outputs
            sample_summary["gt_objects"] = sorted(gt_objects)

            if mask_debug_dir is not None and sample_summary.get("layout_first_step"):
                mask_debug = build_token_mask_debug(
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    generated_ids_only=generated_ids_only,
                    image_len=sample_summary["layout_first_step"]["image_len"],
                    special_token_ids=special_token_ids,
                )
                mask_debug["question_id"] = int(question_id)
                mask_debug["image"] = image_file
                mask_debug["caption"] = outputs
                mask_debug_path = os.path.join(mask_debug_dir, f"{question_id}.json")
                with open(mask_debug_path, "w") as f:
                    json.dump(mask_debug, f, indent=2, ensure_ascii=False)
                print(f"[debug] token-mask map saved: {mask_debug_path}")

            if sample_dir is not None:
                with open(os.path.join(sample_dir, f"{question_id}.json"), "w") as f:
                    json.dump(sample_summary, f, indent=2)

            compact_sample = {
                "question_id": int(question_id),
                "image": image_file,
                "caption": outputs,
                "num_generation_steps": sample_summary["num_generation_steps"],
                "num_object_steps": sample_summary["num_object_steps"],
                "num_hallucinated_steps": sample_summary["num_hallucinated_steps"],
                "num_non_hallucinated_steps": sample_summary["num_non_hallucinated_steps"],
                "num_pre_hallucinated_steps": sample_summary.get("num_pre_hallucinated_steps", 0),
                "num_pre_non_hallucinated_steps": sample_summary.get("num_pre_non_hallucinated_steps", 0),
            }
            global_acc.add_sample(compact_sample)
            metadata["attention_analysis"] = compact_sample

        if args.enable_txtattn_trace:
            trace_summary = extract_txtattn_trace_for_sample(
                question_id=question_id,
                image_file=image_file,
                caption=outputs,
                input_ids=input_ids,
                generated_ids_only=generated_ids_only,
                attentions=output_dict.attentions,
                step_labels=step_labels,
                special_token_ids=special_token_ids,
                heads=txtattn_heads,
                writer=txtattn_writer,
                stats=txtattn_stats,
            )
            metadata["txtattn_trace"] = trace_summary

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": question_id,
                    "image": image_file,
                    "prompt": cur_prompt,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": metadata,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()
    if txtattn_writer is not None:
        txtattn_writer.close()

    if args.enable_txtattn_trace:
        txtattn_summary_file = args.txtattn_summary_file or os.path.join(
            os.path.dirname(answers_file), "txtattn_summary.json"
        )
        os.makedirs(os.path.dirname(os.path.expanduser(txtattn_summary_file)), exist_ok=True)
        summary = txtattn_stats.to_dict()
        summary["config"] = {
            "txtattn_head_file": args.txtattn_head_file,
            "txtattn_topk": args.txtattn_topk,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "note": "I_text is sum of attention over image_end:, matching the text slice used by intervention code.",
        }
        with open(os.path.expanduser(txtattn_summary_file), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.enable_attention_analysis and args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, "attention_summary.json"), "w") as f:
            json.dump(global_acc.to_dict(), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--caption_file_path", type=str, default="")
    parser.add_argument("--annotation-dir", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--dataset", type=str, default="coco")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prompt-text", type=str, default="Please describe this image in detail.")
    parser.add_argument("--save-sample-ids", type=str, default="")
    parser.add_argument(
        "--use-existing-sample-file",
        action="store_true",
        help="Use image/question ids from --existing-sample-file instead of randomly sampling COCO ids.",
    )
    parser.add_argument(
        "--existing-sample-file",
        type=str,
        default="",
        help="JSON file containing existing samples, e.g. captions_eval_results.json with a sentences list.",
    )
    parser.add_argument("--enable-attention-analysis", action="store_true")
    parser.add_argument(
        "--enable-pre-token-analysis",
        action="store_true",
        help="Add pre_object/pre_hallucinated/pre_non_hallucinated buckets using prefix attentions for each generated token.",
    )
    parser.add_argument(
        "--debug-token-masks",
        action="store_true",
        help="Save per-sample token-to-mask maps under output_path/token_mask_debug.",
    )

    parser.add_argument("--enable-txtattn-trace", action="store_true")
    parser.add_argument("--txtattn-head-file", type=str, default="")
    parser.add_argument("--txtattn-topk", type=int, default=20)
    parser.add_argument("--txtattn-output-file", type=str, default="")
    parser.add_argument("--txtattn-summary-file", type=str, default="")

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
