import argparse
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import set_seed

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

from eval_scripts.eval_caption_dynamic import (
    attach_intervention_config,
    finalize_intervention_stats,
    maybe_merge_resume_intervention_stats,
)


def load_amber_generative_queries(args):
    query_file = os.path.expanduser(args.query_file)
    with open(query_file, "r", encoding="utf-8") as f:
        rows = json.load(f)

    rows = [row for row in rows if int(row["id"]) <= 1004]
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    questions = []
    for row in rows:
        query = row.get("query", "Describe this image.")
        prompt = args.prompt_template.format(query=query)
        questions.append(
            {
                "id": int(row["id"]),
                "question_id": int(row["id"]),
                "image": row["image"],
                "query": query,
                "text": prompt,
            }
        )
    return questions


def load_existing_answer_records(path):
    records_by_id = OrderedDict()
    total_valid = 0
    malformed = 0
    missing_id = 0

    if not os.path.exists(path):
        return [], {
            "found": False,
            "total_valid": 0,
            "unique_valid": 0,
            "malformed": 0,
            "missing_id": 0,
            "duplicates": 0,
        }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            question_id = record.get("id", record.get("question_id"))
            if question_id is None:
                missing_id += 1
                continue

            total_valid += 1
            records_by_id[int(question_id)] = record

    records = list(records_by_id.values())
    return records, {
        "found": True,
        "total_valid": total_valid,
        "unique_valid": len(records),
        "malformed": malformed,
        "missing_id": missing_id,
        "duplicates": total_valid - len(records),
    }


def rewrite_answer_records(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in sorted(records, key=lambda x: int(x["id"])):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class AmberGenerativeDataset(Dataset):
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

        image_path = os.path.join(self.image_folder, line["image"])
        image = Image.open(image_path).convert("RGB")
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


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, num_workers=4):
    dataset = AmberGenerativeDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def extract_generated_ids(output_ids, input_ids):
    prompt_len = int(input_ids.shape[1])
    output_len = int(output_ids.shape[1])
    if output_len >= prompt_len and torch.equal(output_ids[:, :prompt_len], input_ids):
        return output_ids[:, prompt_len:]
    return output_ids


def update_image_start_from_input_ids(model, input_ids):
    positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return None
    pos = int(positions[0].item())
    model.config.img_start_pos = pos
    return pos


def write_official_response_file(records, path):
    payload = [
        {
            "id": int(record["id"]),
            "response": record["response"],
        }
        for record in sorted(records, key=lambda x: int(x["id"]))
    ]
    with open(os.path.expanduser(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_official_generative_stdout(stdout):
    metrics = {}
    patterns = {
        "CHAIR": r"CHAIR:\s*([0-9.]+)",
        "Cover": r"Cover:\s*([0-9.]+)",
        "Hal": r"Hal:\s*([0-9.]+)",
        "Cog": r"Cog:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def run_official_amber_eval(args, response_file):
    out_dir = os.path.dirname(os.path.expanduser(args.answers_file))
    stdout_path = args.official_stdout_file or os.path.join(out_dir, "amber_official_eval_stdout.txt")
    stderr_path = args.official_stderr_file or os.path.join(out_dir, "amber_official_eval_stderr.txt")
    metrics_path = args.metrics_file or os.path.join(out_dir, "amber_metrics.json")

    amber_root = os.path.abspath(os.path.expanduser(args.amber_root))
    cmd = [
        sys.executable,
        "inference.py",
        "--inference_data",
        os.path.abspath(os.path.expanduser(response_file)),
        "--evaluation_type",
        "g",
    ]

    proc = subprocess.run(
        cmd,
        cwd=amber_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(proc.stderr)

    result = {
        "official_command": cmd,
        "official_cwd": amber_root,
        "returncode": proc.returncode,
        "stdout_file": stdout_path,
        "stderr_file": stderr_path,
        "metrics": parse_official_generative_stdout(proc.stdout),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if proc.returncode != 0 and args.strict_official_eval:
        raise RuntimeError(
            f"AMBER official evaluator failed with return code {proc.returncode}. "
            f"See {stderr_path}"
        )
    return result


def save_run_config(args, head_cfg, questions, pending_questions=None, resumed_records=0, resume_info=None):
    out_dir = os.path.dirname(os.path.expanduser(args.answers_file))
    os.makedirs(out_dir, exist_ok=True)
    cfg = vars(args).copy()
    cfg["selected_heads"] = head_cfg["heads"]
    cfg["num_questions"] = len(questions)
    cfg["num_pending_questions"] = len(pending_questions) if pending_questions is not None else len(questions)
    cfg["resumed_records"] = int(resumed_records)
    cfg["resume_info"] = resume_info or {}
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def save_intervention_stats(model, args):
    if not args.log_intervention_stats:
        return
    stats = finalize_intervention_stats(getattr(model.config, "_intervention_stats", None))
    stats["config"] = {
        "intervention": args.intervention,
        "topk": args.topk,
        "text_threshold": args.text_threshold,
        "text_scale": args.text_scale,
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
    }
    out_file = args.intervention_stats_file or os.path.join(
        os.path.dirname(os.path.expanduser(args.answers_file)), "intervention_stats.json"
    )
    out_file = os.path.expanduser(out_file)
    stats = maybe_merge_resume_intervention_stats(stats, out_file, args)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def eval_model(args):
    questions = load_amber_generative_queries(args)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    response_file = args.response_file or os.path.join(os.path.dirname(answers_file), "amber_responses.json")

    existing_records = []
    resume_info = {}
    if args.resume:
        existing_records, resume_info = load_existing_answer_records(answers_file)
        rewrite_answer_records(answers_file, existing_records)

    processed_ids = {int(record["id"]) for record in existing_records}
    pending_questions = [q for q in questions if int(q["id"]) not in processed_ids]

    if args.skip_generation:
        records = existing_records
        if not records:
            records, resume_info = load_existing_answer_records(answers_file)
        write_official_response_file(records, response_file)
        if args.run_official_eval:
            run_official_amber_eval(args, response_file)
        return

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"Auto switching conversation mode to {args.conv_mode}.")

    data_loader = create_data_loader(
        pending_questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        args.conv_mode,
        num_workers=args.num_workers,
    )

    head_cfg = attach_intervention_config(model, args)
    save_run_config(
        args,
        head_cfg,
        questions,
        pending_questions=pending_questions,
        resumed_records=len(existing_records),
        resume_info=resume_info,
    )
    model.config.log_intervention_stats = args.log_intervention_stats
    model.config._intervention_stats = {"overall": {}, "by_head": {}}

    records = list(existing_records)
    answer_mode = "a" if args.resume else "w"
    with open(answers_file, answer_mode, encoding="utf-8") as ans_file:
        for (input_ids, image_tensor, image_sizes), line in tqdm(
            zip(data_loader, pending_questions), total=len(pending_questions)
        ):
            input_ids = input_ids.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

            inferred_img_start = None
            if args.infer_image_start:
                inferred_img_start = update_image_start_from_input_ids(model, input_ids)

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
            generated_ids = extract_generated_ids(output_ids, input_ids)
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            record = {
                "id": int(line["id"]),
                "question_id": int(line["question_id"]),
                "image": line["image"],
                "query": line["query"],
                "prompt": line["text"],
                "response": outputs,
                "model_id": model_name,
                "metadata": {
                    "intervention": args.intervention,
                    "topk": args.topk,
                    "inferred_img_start_pos": inferred_img_start,
                },
            }
            ans_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            ans_file.flush()
            records.append(record)

    write_official_response_file(records, response_file)
    save_intervention_stats(model, args)

    if args.run_official_eval:
        result = run_official_amber_eval(args, response_file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Saved AMBER official response file -> {response_file}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="../dataset/AMBER/images")
    parser.add_argument("--amber-root", type=str, default="../third_party/AMBER")
    parser.add_argument("--query-file", type=str, default="../third_party/AMBER/data/query/query_generative.json")
    parser.add_argument("--answers-file", type=str, default="results_amber/generative/answers.jsonl")
    parser.add_argument("--response-file", type=str, default="")
    parser.add_argument("--metrics-file", type=str, default="")
    parser.add_argument("--official-stdout-file", type=str, default="")
    parser.add_argument("--official-stderr-file", type=str, default="")
    parser.add_argument("--run-official-eval", action="store_true")
    parser.add_argument("--strict-official-eval", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")

    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--prompt-template", type=str, default="{query}")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--infer-image-start", action="store_true", default=True)
    parser.add_argument("--no-infer-image-start", dest="infer_image_start", action="store_false")

    parser.add_argument("--intervention", type=str, default="none", choices=["none", "adhh", "soft", "dynamic"])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--head-source", type=str, default="default", choices=["default", "file"])
    parser.add_argument("--head-file", type=str, default="")
    parser.add_argument("--head-score-key", type=str, default="score")
    parser.add_argument("--head-score-normalize", type=str, default="minmax", choices=["minmax", "raw", "logminmax"])
    parser.add_argument("--use-head-scores", action="store_true")

    parser.add_argument("--text-threshold", type=float, default=0.4)
    parser.add_argument("--text-scale", type=float, default=0.5)

    parser.add_argument("--dynamic-strength", type=float, default=1.0)
    parser.add_argument("--dynamic-ratio-power", type=float, default=1.0)
    parser.add_argument("--dynamic-score-power", type=float, default=1.0)
    parser.add_argument("--dynamic-context-mode", type=str, default="ratio_exp", choices=["text_exp", "ratio_exp", "ratio_power", "text_power"])
    parser.add_argument("--dynamic-tau", type=float, default=0.9)
    parser.add_argument("--dynamic-exp-sharpness", type=float, default=6.0)
    parser.add_argument("--dynamic-redistribute", type=str, default="renorm", choices=["renorm", "system", "system_only", "vision", "vision_only"])

    parser.add_argument("--log-intervention-stats", action="store_true")
    parser.add_argument("--intervention-stats-file", type=str, default="")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
