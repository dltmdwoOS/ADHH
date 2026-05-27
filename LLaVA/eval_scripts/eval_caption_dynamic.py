import os
import json
import math
import shutil
import random
import argparse

import shortuuid
import torch

from tqdm import tqdm
from PIL import Image
from transformers import set_seed
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


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def default_head_setup(model_path):
    if model_path == "liuhaotian/llava-v1.5-7b":
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
    elif model_path == "liuhaotian/llava-v1.5-13b":
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
    elif model_path == "liuhaotian/llava-v1.6-34b":
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


def load_selected_heads(head_file, topk, score_key="score", score_normalize="minmax"):
    with open(head_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "heads" in data and isinstance(data["heads"], list):
            records = data["heads"]
            top = records[:topk]
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
        top = records[:topk]
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

        dest_image_folder = os.path.join(
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

        dest_image_folder = os.path.join(
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
        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


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
    model.config.gate_floor = args.gate_floor
    model.config.gate_midpoint = args.gate_midpoint
    model.config.gate_sharpness = args.gate_sharpness
    model.config.gate_score_power = args.gate_score_power
    model.config.gate_hard_threshold = args.gate_hard_threshold
    model.config.dynamic_strength = args.dynamic_strength
    model.config.dynamic_ratio_power = args.dynamic_ratio_power
    model.config.dynamic_score_power = args.dynamic_score_power
    model.config.dynamic_tau = args.dynamic_tau
    model.config.dynamic_exp_sharpness = args.dynamic_exp_sharpness
    model.config.dynamic_context_mode = args.dynamic_context_mode
    model.config.use_head_scores = args.use_head_scores
    model.config.log_dynamic_trace = args.log_dynamic_trace
    model.config.dynamic_trace_topn = args.dynamic_trace_topn
    model.config.dynamic_trace_every = args.dynamic_trace_every
    model.config._dynamic_trace_step = 0
    model.config._dynamic_trace_buffer = []

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
        "intervention": args.intervention,
        "head_source": args.head_source,
        "topk": args.topk,
        "text_threshold": args.text_threshold,
        "text_scale": args.text_scale,
        "gate_floor": args.gate_floor,
        "gate_midpoint": args.gate_midpoint,
        "gate_sharpness": args.gate_sharpness,
        "gate_score_power": args.gate_score_power,
        "gate_hard_threshold": args.gate_hard_threshold,
        "dynamic_strength": args.dynamic_strength,
        "dynamic_ratio_power": args.dynamic_ratio_power,
        "dynamic_score_power": args.dynamic_score_power,
        "dynamic_tau": args.dynamic_tau,
        "dynamic_exp_sharpness": args.dynamic_exp_sharpness,
        "dynamic_context_mode": args.dynamic_context_mode,
        "use_head_scores": args.use_head_scores,
        "head_file": args.head_file,
        "head_score_key": args.head_score_key,
        "head_score_normalize": args.head_score_normalize,
        "log_intervention_stats": args.log_intervention_stats,
        "intervention_stats_file": args.intervention_stats_file,
        "log_dynamic_trace": args.log_dynamic_trace,
        "dynamic_trace_topn": args.dynamic_trace_topn,
        "dynamic_trace_every": args.dynamic_trace_every,
        "selected_heads": head_cfg["heads"],
    }
    out_dir = os.path.dirname(os.path.expanduser(args.answers_file))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)


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
        "overall": {
            mode: finalize_bucket(bucket)
            for mode, bucket in raw_stats.get("overall", {}).items()
        },
        "by_head": {
            key: finalize_bucket(bucket)
            for key, bucket in raw_stats.get("by_head", {}).items()
        },
    }


def save_intervention_stats(model, args):
    if not args.log_intervention_stats:
        return

    stats = finalize_intervention_stats(getattr(model.config, "_intervention_stats", None))
    stats["config"] = {
        "intervention": args.intervention,
        "topk": args.topk,
        "text_threshold": args.text_threshold,
        "gate_floor": args.gate_floor,
        "gate_midpoint": args.gate_midpoint,
        "gate_sharpness": args.gate_sharpness,
        "gate_score_power": args.gate_score_power,
        "gate_hard_threshold": args.gate_hard_threshold,
        "dynamic_strength": args.dynamic_strength,
        "dynamic_ratio_power": args.dynamic_ratio_power,
        "dynamic_score_power": args.dynamic_score_power,
        "dynamic_tau": args.dynamic_tau,
        "dynamic_exp_sharpness": args.dynamic_exp_sharpness,
        "dynamic_context_mode": args.dynamic_context_mode,
        "use_head_scores": args.use_head_scores,
        "head_file": args.head_file,
        "head_score_key": args.head_score_key,
        "head_score_normalize": args.head_score_normalize,
        "log_dynamic_trace": args.log_dynamic_trace,
        "dynamic_trace_topn": args.dynamic_trace_topn,
        "dynamic_trace_every": args.dynamic_trace_every,
    }

    out_file = args.intervention_stats_file
    if not out_file:
        out_file = os.path.join(os.path.dirname(os.path.expanduser(args.answers_file)), "intervention_stats.json")

    with open(os.path.expanduser(out_file), "w") as f:
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
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode,
        batch_size=1, num_workers=args.num_workers
    )

    head_cfg = attach_intervention_config(model, args)
    save_run_config(args, head_cfg)
    model.config.log_intervention_stats = args.log_intervention_stats
    model.config._intervention_stats = {"overall": {}, "by_head": {}}

    for (input_ids, image_tensor, image_sizes), line in tqdm(
        zip(data_loader, questions),
        total=len(questions)
    ):
        question_id = line["question_id"]
        cur_prompt = line["text"]
        image_file = line["image"]

        model.config.dynamic_trace_sample_id = question_id
        model.config._dynamic_trace_step = 0
        model.config._dynamic_trace_buffer = []

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

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
                output_attentions=True,
                return_dict_in_generate=True,
            )

        output_ids = output_dict["sequences"]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print(question_id, outputs)

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": question_id,
            "image": image_file,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {
                "intervention": args.intervention,
                "topk": args.topk,
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

    parser.add_argument("--intervention", type=str, default="dynamic",
                        choices=["none", "dynamic"])
    parser.add_argument("--topk", type=int, default=20)

    parser.add_argument("--dynamic-strength", type=float, default=1.0,
                        help="Maximum continuous suppression strength before clipping.")
    parser.add_argument("--dynamic-ratio-power", type=float, default=1.0,
                        help="Power applied to text/(text+image) reliance; 0 disables context modulation.")
    parser.add_argument("--dynamic-score-power", type=float, default=1.0,
                        help="Power applied to normalized offline head scores.")
    parser.add_argument("--dynamic-context-mode", type=str, default="text_exp",
                        choices=["text_exp", "ratio_exp", "ratio_power", "text_power"])
    parser.add_argument("--dynamic-tau", type=float, default=0.5,
                        help="Center point for exponential dynamic context modes.")
    parser.add_argument("--dynamic-exp-sharpness", type=float, default=6.0,
                        help="Sharpness k in exp(k * (context - tau)).")
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
    parser.add_argument("--gate-floor", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--gate-midpoint", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--gate-sharpness", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--gate-score-power", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--gate-hard-threshold", type=float, default=0.0, help=argparse.SUPPRESS)

    parser.add_argument("--sample-id-file", type=str, default="")
    parser.add_argument("--save-sample-id-file", type=str, default="")

    parser.add_argument("--head-source", type=str, default="default",
                    choices=["default", "file"])
    parser.add_argument("--head-file", type=str, default="")
    parser.add_argument("--head-score-key", type=str, default="score")
    parser.add_argument("--head-score-normalize", type=str, default="minmax",
                        choices=["minmax", "raw", "logminmax", "rank_percentile"])
    parser.add_argument("--log-intervention-stats", action="store_true")
    parser.add_argument("--intervention-stats-file", type=str, default="")

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
