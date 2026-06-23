import argparse
import json
import math
import os
import random
import time
from types import SimpleNamespace

import shortuuid
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import set_seed

from eval_scripts.eval_caption_dynamic import resolve_head_config
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


def load_sample_ids(args):
    if args.sample_id_file and os.path.exists(args.sample_id_file):
        with open(args.sample_id_file, "r", encoding="utf-8") as f:
            return [int(x) for x in json.load(f)][: args.num_samples]

    coco = COCO(args.caption_file_path)
    img_ids = coco.getImgIds()
    random.seed(args.seed)
    sampled = random.sample(img_ids, args.num_samples)
    if args.sample_id_file:
        os.makedirs(os.path.dirname(os.path.expanduser(args.sample_id_file)), exist_ok=True)
        with open(os.path.expanduser(args.sample_id_file), "w", encoding="utf-8") as f:
            json.dump(sampled, f, indent=2)
    return sampled


def build_questions(args, sample_ids):
    coco = COCO(args.caption_file_path)
    id_to_img = {int(img["id"]): img for img in coco.dataset["images"]}
    questions = []
    for image_id in sample_ids:
        image_file = id_to_img[int(image_id)]["file_name"]
        questions.append(
            {
                "question_id": int(image_id),
                "image": image_file,
                "text": args.prompt_text,
            }
        )
    return questions


class CaptionDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, line["image"])).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return input_ids, image_tensor, image.size, prompt

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, prompts = zip(*batch)
    return torch.stack(input_ids, dim=0), torch.stack(image_tensors, dim=0), image_sizes, prompts


def create_loader(args, questions, tokenizer, image_processor, model_config):
    dataset = CaptionDataset(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model_config,
        args.conv_mode,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def extract_generated_ids(output_ids, input_ids):
    prompt_len = int(input_ids.shape[1])
    if int(output_ids.shape[1]) >= prompt_len and torch.equal(output_ids[:, :prompt_len], input_ids):
        return output_ids[:, prompt_len:]
    return output_ids


def read_auto_tau(tau_file, fallback_hi, fallback_lo):
    if not tau_file or not os.path.exists(tau_file):
        return fallback_hi, fallback_lo
    with open(tau_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    hi = float(data.get("recommended_tau_hi_str", data.get("recommended_tau_str", fallback_hi)))
    lo = float(data.get("recommended_tau_lo_str", data.get("recommended_tau_str", fallback_lo)))
    return hi, lo


def reset_method_config(model):
    cfg = model.config
    cfg.intervention = "none"
    cfg.enable_txtattn_last_row_trace = False
    cfg.log_dynamic_trace = False
    cfg.log_intervention_stats = False
    cfg._dynamic_trace_step = 0
    cfg._dynamic_trace_buffer = []
    cfg._intervention_stats = {"overall": {}, "by_head": {}}
    cfg._tarac_attn_memory = {}


def configure_method(model, args, method):
    reset_method_config(model)
    if method == "greedy":
        return {"method": "greedy", "intervention": "none"}

    if method == "adhh":
        method_args = SimpleNamespace(
            model_path=args.model_path,
            head_source=args.adhh_head_source,
            head_file=args.adhh_head_file,
            topk=args.adhh_topk,
            head_score_key=args.adhh_head_score_key,
            head_score_normalize=args.adhh_head_score_normalize,
            min_head_back_raw=0.0,
        )
        head_cfg = resolve_head_config(method_args)
        model.config.intervention = "adhh"
        model.config.intervention_heads = head_cfg["heads"]
        model.config.intervention_scores = head_cfg["scores"]
        model.config.hal_attention_heads = head_cfg["heads"]
        model.config.img_start_pos = head_cfg["img_start_pos"]
        model.config.img_length = head_cfg["img_length"]
        model.config.text_threshold = args.adhh_threshold
        model.config.adhh_threshold = args.adhh_threshold
        model.config.text_scale = args.adhh_text_scale
        model.config.use_head_scores = args.adhh_use_head_scores
        model.config.adaptive_deactivate = True
        return {
            "method": "adhh",
            "intervention": "adhh",
            "topk": args.adhh_topk,
            "text_threshold": args.adhh_threshold,
            "head_source": args.adhh_head_source,
            "head_file": args.adhh_head_file,
            "selected_head_count": len(head_cfg["heads"]),
        }

    if method == "deact":
        tau_hi, tau_lo = args.deact_tau, args.deact_late_tau
        if args.deact_auto_tau:
            tau_hi, tau_lo = read_auto_tau(args.deact_tau_file, tau_hi, tau_lo)
        method_args = SimpleNamespace(
            model_path=args.model_path,
            head_source="file",
            head_file=args.deact_head_file,
            topk=args.deact_topk,
            head_score_key=args.deact_head_score_key,
            head_score_normalize=args.deact_head_score_normalize,
            min_head_back_raw=args.deact_min_head_back_raw,
        )
        head_cfg = resolve_head_config(method_args)
        cfg = model.config
        cfg.intervention = "late_boost"
        cfg.intervention_heads = head_cfg["heads"]
        cfg.intervention_scores = head_cfg["scores"]
        cfg.hal_attention_heads = head_cfg["heads"]
        cfg.img_start_pos = head_cfg["img_start_pos"]
        cfg.img_length = head_cfg["img_length"]
        cfg.text_threshold = 0.0
        cfg.text_scale = 0.5
        cfg.dynamic_strength = args.deact_strength
        cfg.dynamic_ratio_power = args.deact_ratio_power
        cfg.dynamic_score_power = args.deact_score_power
        cfg.dynamic_tau = tau_hi
        cfg.dynamic_exp_sharpness = args.deact_exp_sharpness
        cfg.dynamic_late_boost_start = args.deact_late_boost_start
        cfg.dynamic_late_boost_end = args.deact_late_boost_end if args.deact_late_boost_end > 0 else args.max_new_tokens
        cfg.dynamic_late_boost_mode = args.deact_late_boost_mode
        cfg.dynamic_late_tau = tau_lo
        cfg.dynamic_context_mode = args.deact_context_mode
        cfg.dynamic_redistribute = args.deact_redistribute
        cfg.dynamic_renorm = args.deact_renorm
        cfg.use_head_scores = args.deact_use_head_scores
        cfg.adaptive_deactivate = False
        return {
            "method": "deact",
            "intervention": "late_boost",
            "topk": args.deact_topk,
            "dynamic_tau": tau_hi,
            "dynamic_late_tau": tau_lo,
            "dynamic_exp_sharpness": args.deact_exp_sharpness,
            "dynamic_redistribute": args.deact_redistribute,
            "dynamic_renorm": args.deact_renorm,
            "head_file": args.deact_head_file,
            "selected_head_count": len(head_cfg["heads"]),
        }

    raise ValueError(f"Unknown method: {method}")


def run_generate(model, tokenizer, args, input_ids, image_tensor, image_sizes):
    output_dict = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        output_attentions=args.force_output_attentions,
        output_scores=False,
        output_hidden_states=False,
        return_dict_in_generate=True,
    )
    output_ids = output_dict["sequences"]
    generated_ids = extract_generated_ids(output_ids, input_ids)
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return output_ids, generated_ids, text


def cuda_memory_snapshot(prefix):
    if not torch.cuda.is_available():
        return {}
    return {
        f"{prefix}_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        f"{prefix}_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        f"{prefix}_max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        f"{prefix}_max_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
    }


def run_method(model, tokenizer, loader, questions, args, method, model_name):
    method_dir = os.path.join(args.output_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    captions_file = os.path.join(method_dir, "captions.jsonl")
    metrics_file = os.path.join(method_dir, "latency_memory.json")

    method_config = configure_method(model, args, method)
    warmup_done = 0
    if args.warmup_samples > 0:
        for (input_ids, image_tensor, image_sizes, _), _line in zip(loader, questions):
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
            with torch.inference_mode():
                run_generate(model, tokenizer, args, input_ids, image_tensor, image_sizes)
            warmup_done += 1
            if warmup_done >= args.warmup_samples:
                break
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    records = []
    total_seconds = 0.0
    total_generated_tokens = 0
    with open(captions_file, "w", encoding="utf-8") as ans_file:
        for idx, ((input_ids, image_tensor, image_sizes, prompts), line) in enumerate(
            tqdm(zip(loader, questions), total=len(questions), desc=method),
            start=1,
        ):
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output_ids, generated_ids, text = run_generate(
                    model, tokenizer, args, input_ids, image_tensor, image_sizes
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            generated_token_count = int(generated_ids.shape[1])
            total_seconds += elapsed
            total_generated_tokens += generated_token_count
            records.append(
                {
                    "question_id": int(line["question_id"]),
                    "image": line["image"],
                    "seconds": elapsed,
                    "generated_tokens": generated_token_count,
                    "tokens_per_second": generated_token_count / elapsed if elapsed > 0 else None,
                }
            )
            ans_file.write(
                json.dumps(
                    {
                        "question_id": int(line["question_id"]),
                        "image": line["image"],
                        "prompt": line["text"],
                        "text": text,
                        "answer_id": shortuuid.uuid(),
                        "model_id": model_name,
                        "metadata": {
                            "benchmark_method": method,
                            "model_input_prompt": prompts[0],
                            "output_token_ids": output_ids[0].detach().cpu().tolist(),
                            "generated_token_count": generated_token_count,
                            "seconds": elapsed,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ans_file.flush()
            if args.limit and idx >= args.limit:
                break

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    summary = {
        "method": method,
        "model_name": model_name,
        "num_images": len(records),
        "warmup_samples": warmup_done,
        "total_seconds": total_seconds,
        "seconds_per_image": total_seconds / max(len(records), 1),
        "total_generated_tokens": total_generated_tokens,
        "tokens_per_second": total_generated_tokens / total_seconds if total_seconds > 0 else None,
        "mean_generated_tokens": total_generated_tokens / max(len(records), 1),
        "force_output_attentions": args.force_output_attentions,
        "baseline_allocated_gb": baseline_allocated / (1024**3),
        "baseline_reserved_gb": baseline_reserved / (1024**3),
        "peak_allocated_gb": peak_allocated / (1024**3),
        "peak_reserved_gb": peak_reserved / (1024**3),
        "peak_generation_delta_allocated_gb": (peak_allocated - baseline_allocated) / (1024**3),
        "peak_generation_delta_reserved_gb": (peak_reserved - baseline_reserved) / (1024**3),
        "method_config": method_config,
        "per_sample": records,
    }
    summary.update(cuda_memory_snapshot("final"))
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_combined_summary(output_dir, summaries):
    greedy_seconds = next((x["seconds_per_image"] for x in summaries if x["method"] == "greedy"), None)
    greedy_memory = next(
        (x["peak_generation_delta_allocated_gb"] for x in summaries if x["method"] == "greedy"),
        None,
    )
    rows = []
    for item in summaries:
        row = dict(item)
        row.pop("per_sample", None)
        row["runtime_multiplier_vs_greedy"] = (
            row["seconds_per_image"] / greedy_seconds if greedy_seconds and greedy_seconds > 0 else None
        )
        row["memory_delta_multiplier_vs_greedy"] = (
            row["peak_generation_delta_allocated_gb"] / greedy_memory
            if greedy_memory and greedy_memory > 0
            else None
        )
        rows.append(row)
    with open(os.path.join(output_dir, "latency_memory_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"methods": rows}, f, indent=2)


def eval_model(args):
    disable_torch_init()
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"

    sample_ids = load_sample_ids(args)
    questions = build_questions(args, sample_ids)
    if args.limit:
        questions = questions[: args.limit]
    loader = create_loader(args, questions, tokenizer, image_processor, model.config)

    run_config = vars(args).copy()
    run_config["methods"] = args.methods
    with open(os.path.join(args.output_dir, "benchmark_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    summaries = []
    for method in args.methods.split(","):
        method = method.strip()
        if not method:
            continue
        summaries.append(run_method(model, tokenizer, loader, questions, args, method, model_name))
    write_combined_summary(args.output_dir, summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--caption_file_path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample-id-file", type=str, default="")
    parser.add_argument("--methods", type=str, default="greedy,adhh,deact")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup-samples", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--prompt-text", type=str, default="Please describe this image in detail.")
    parser.add_argument("--force-output-attentions", action="store_true")

    parser.add_argument("--adhh-topk", type=int, default=20)
    parser.add_argument("--adhh-threshold", type=float, default=0.4)
    parser.add_argument("--adhh-text-scale", type=float, default=0.5)
    parser.add_argument("--adhh-head-source", choices=["default", "file"], default="file")
    parser.add_argument("--adhh-head-file", type=str, default="")
    parser.add_argument("--adhh-head-score-key", type=str, default="score")
    parser.add_argument("--adhh-head-score-normalize", choices=["minmax", "raw", "logminmax", "rank_percentile"], default="minmax")
    parser.add_argument("--adhh-use-head-scores", action="store_true")

    parser.add_argument("--deact-head-file", type=str, required=True)
    parser.add_argument("--deact-tau-file", type=str, default="")
    parser.add_argument("--deact-auto-tau", action="store_true")
    parser.add_argument("--deact-topk", type=int, default=100)
    parser.add_argument("--deact-strength", type=float, default=1.0)
    parser.add_argument("--deact-ratio-power", type=float, default=1.0)
    parser.add_argument("--deact-score-power", type=float, default=1.0)
    parser.add_argument("--deact-tau", type=float, default=0.90)
    parser.add_argument("--deact-late-tau", type=float, default=0.80)
    parser.add_argument("--deact-exp-sharpness", type=float, default=10.0)
    parser.add_argument("--deact-late-boost-start", type=int, default=0)
    parser.add_argument("--deact-late-boost-end", type=int, default=128)
    parser.add_argument("--deact-late-boost-mode", choices=["linear", "step"], default="linear")
    parser.add_argument("--deact-context-mode", choices=["text_exp", "ratio_exp", "ratio_power", "text_power"], default="ratio_exp")
    parser.add_argument("--deact-redistribute", choices=["none", "renorm", "system", "system_only", "vision", "vision_only", "sysvis"], default="sysvis")
    parser.add_argument("--no-deact-renorm", dest="deact_renorm", action="store_false")
    parser.set_defaults(deact_renorm=True)
    parser.add_argument("--deact-use-head-scores", action="store_true")
    parser.add_argument("--deact-head-score-key", type=str, default="global__itext_all__C_toi_HminusG_signed")
    parser.add_argument("--deact-head-score-normalize", choices=["minmax", "raw", "logminmax", "rank_percentile"], default="rank_percentile")
    parser.add_argument("--deact-min-head-back-raw", type=float, default=0.0)

    parsed = parser.parse_args()
    set_seed(parsed.seed)
    eval_model(parsed)
