import argparse
import csv
import json
import os
import shlex
import subprocess
from collections import Counter, OrderedDict, defaultdict

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
)


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _word_count(text):
    return len(str(text or "").strip().split())


def _answer_operator(answer):
    answer = str(answer or "")
    has_and = "<AND>" in answer
    has_or = "<OR>" in answer
    if has_and and has_or:
        return "and_or"
    if has_and:
        return "and"
    if has_or:
        return "or"
    return "single"


def _combo_name(capability):
    caps = sorted(_as_list(capability))
    return "+".join(caps) if caps else "unknown"


def build_question_prompt(question, args):
    if args.prompt_style == "exact":
        return args.prompt_template.format(question=question)
    if args.prompt_style == "direct":
        return args.prompt_template.format(question=question) + "\nAnswer the question directly."
    if args.prompt_style == "short":
        return args.prompt_template.format(question=question) + "\nAnswer with the final answer only when possible."
    raise ValueError(f"Unknown prompt_style: {args.prompt_style}")


def load_mmvet_questions(args):
    question_file = os.path.expanduser(args.question_file)
    with open(question_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = list(data.items())
    elif isinstance(data, list):
        items = [(str(i), row) for i, row in enumerate(data)]
    else:
        raise ValueError(f"Unsupported MM-Vet question file format: {question_file}")

    if args.max_samples and args.max_samples > 0:
        items = items[: args.max_samples]

    questions = []
    for key, row in items:
        question_id = str(row.get("id", row.get("question_id", key)))
        image = (
            row.get("imagename")
            or row.get("image")
            or row.get("image_path")
            or row.get("image_name")
        )
        question = row.get("question", row.get("query", row.get("text")))
        if image is None or question is None:
            raise ValueError(f"MM-Vet row {question_id} must contain image and question fields: {row}")

        prompt = build_question_prompt(question, args)
        questions.append(
            {
                "question_id": question_id,
                "image": image,
                "question": question,
                "text": prompt,
                "answer": row.get("answer", ""),
                "capability": _as_list(row.get("capability", row.get("capabilities"))),
                "capability_combo": _combo_name(row.get("capability", row.get("capabilities"))),
                "imagesource": row.get("imagesource", row.get("image_source", "")),
                "answer_operator": _answer_operator(row.get("answer", "")),
                "raw": row,
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

            question_id = record.get("question_id")
            if question_id is None:
                missing_id += 1
                continue

            total_valid += 1
            records_by_id[str(question_id)] = record

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
        for record in sorted(records, key=lambda x: str(x["question_id"])):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class MMVetDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode, system_prompt=""):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode
        self.system_prompt = system_prompt

    def __getitem__(self, index):
        line = self.questions[index]
        qs = line["text"]

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        if self.system_prompt:
            conv.system = self.system_prompt
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_path = os.path.join(self.image_folder, line["image"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing MM-Vet image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return input_ids, image_tensor, image.size, prompt

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes, prompts = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes, prompts


def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, system_prompt="", num_workers=4):
    dataset = MMVetDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode, system_prompt=system_prompt)
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


def write_mmvet_response_file(records, path):
    payload = {
        str(record["question_id"]): record["text"]
        for record in sorted(records, key=lambda x: str(x["question_id"]))
    }
    with open(os.path.expanduser(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_mmvet_eval_payload(records, path):
    payload = [
        {
            "question_id": str(record["question_id"]),
            "image": record["image"],
            "question": record["question"],
            "prompt": record.get("prompt", ""),
            "full_prompt": record.get("full_prompt", ""),
            "answer": record.get("answer", ""),
            "prediction": record["text"],
            "capability": record.get("capability", []),
            "capability_combo": record.get("capability_combo", _combo_name(record.get("capability", []))),
            "imagesource": record.get("imagesource", ""),
            "answer_operator": record.get("answer_operator", _answer_operator(record.get("answer", ""))),
            "prediction_word_count": _word_count(record.get("text", "")),
        }
        for record in sorted(records, key=lambda x: str(x["question_id"]))
    ]
    with open(os.path.expanduser(path), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)




def write_mmvet_local_summary(records, path):
    summary = {
        "num_records": len(records),
        "capability_counts": Counter(),
        "capability_combo_counts": Counter(),
        "imagesource_counts": Counter(),
        "answer_operator_counts": Counter(),
        "generation_length": {"avg_words": 0.0, "max_words": 0, "min_words": 0},
        "by_capability": defaultdict(lambda: {"count": 0, "total_words": 0, "max_words": 0}),
        "by_capability_combo": defaultdict(lambda: {"count": 0, "total_words": 0, "max_words": 0}),
    }
    word_counts = []
    for record in records:
        caps = _as_list(record.get("capability", []))
        combo = record.get("capability_combo") or _combo_name(caps)
        imagesource = record.get("imagesource", "") or "unknown"
        answer_operator = record.get("answer_operator") or _answer_operator(record.get("answer", ""))
        words = _word_count(record.get("text", ""))
        word_counts.append(words)

        summary["capability_combo_counts"][combo] += 1
        summary["imagesource_counts"][imagesource] += 1
        summary["answer_operator_counts"][answer_operator] += 1

        combo_bucket = summary["by_capability_combo"][combo]
        combo_bucket["count"] += 1
        combo_bucket["total_words"] += words
        combo_bucket["max_words"] = max(combo_bucket["max_words"], words)

        for cap in caps or ["unknown"]:
            summary["capability_counts"][cap] += 1
            cap_bucket = summary["by_capability"][cap]
            cap_bucket["count"] += 1
            cap_bucket["total_words"] += words
            cap_bucket["max_words"] = max(cap_bucket["max_words"], words)

    if word_counts:
        summary["generation_length"] = {
            "avg_words": sum(word_counts) / len(word_counts),
            "max_words": max(word_counts),
            "min_words": min(word_counts),
        }

    for section_name in ("by_capability", "by_capability_combo"):
        for bucket in summary[section_name].values():
            count = max(bucket["count"], 1)
            bucket["avg_words"] = bucket["total_words"] / count
            del bucket["total_words"]

    serializable = {
        key: (dict(value) if isinstance(value, (Counter, defaultdict)) else value)
        for key, value in summary.items()
    }
    with open(os.path.expanduser(path), "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def collect_official_eval_outputs(args, response_file, result):
    out_dir = os.path.dirname(os.path.expanduser(args.answers_file))
    stem = os.path.splitext(os.path.basename(response_file))[0]
    gpt_model = args.official_gpt_model
    num_run = args.official_num_run
    candidates = {
        "grade_file": os.path.join(out_dir, f"{stem}_{gpt_model}-grade-{num_run}runs.json"),
        "cap_score_file": os.path.join(out_dir, f"{stem}_{gpt_model}-cap-score-{num_run}runs.csv"),
        "cap_int_score_file": os.path.join(out_dir, f"{stem}_{gpt_model}-cap-int-score-{num_run}runs.csv"),
    }
    result["official_outputs"] = candidates

    score_tables = {}
    for key in ("cap_score_file", "cap_int_score_file"):
        path = candidates[key]
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            score_tables[key] = rows
    if score_tables:
        result["score_tables"] = score_tables
    return result


def run_official_mmvet_eval(args, response_file):
    out_dir = os.path.dirname(os.path.expanduser(args.answers_file))
    stdout_path = args.official_stdout_file or os.path.join(out_dir, "mmvet_official_eval_stdout.txt")
    stderr_path = args.official_stderr_file or os.path.join(out_dir, "mmvet_official_eval_stderr.txt")
    metrics_path = args.metrics_file or os.path.join(out_dir, "mmvet_metrics.json")

    official_eval_command = args.official_eval_command or (
        "python mm-vet_evaluator.py "
        "--mmvet_path {mmvet_data_root} "
        "--result_file {response_file} "
        "--result_path {output_dir} "
        "--gpt_model {official_gpt_model} "
        "--num_run {official_num_run}"
    )
    formatted = official_eval_command.format(
        response_file=os.path.abspath(os.path.expanduser(response_file)),
        question_file=os.path.abspath(os.path.expanduser(args.question_file)),
        metrics_file=os.path.abspath(os.path.expanduser(metrics_path)),
        mmvet_root=os.path.abspath(os.path.expanduser(args.mmvet_root)),
        mmvet_data_root=os.path.abspath(os.path.expanduser(args.mmvet_data_root)),
        output_dir=os.path.abspath(out_dir),
        official_gpt_model=args.official_gpt_model,
        official_num_run=args.official_num_run,
    )
    cmd = shlex.split(formatted)
    proc = subprocess.run(
        cmd,
        cwd=os.path.abspath(os.path.expanduser(args.mmvet_root)),
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
        "official_cwd": os.path.abspath(os.path.expanduser(args.mmvet_root)),
        "returncode": proc.returncode,
        "stdout_file": stdout_path,
        "stderr_file": stderr_path,
        "metrics_file": metrics_path,
    }
    result = collect_official_eval_outputs(args, response_file, result)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if proc.returncode != 0 and args.strict_official_eval:
        raise RuntimeError(
            f"MM-Vet official evaluator failed with return code {proc.returncode}. "
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




def filter_records_for_questions(records, questions):
    active_ids = {str(question["question_id"]) for question in questions}
    return [record for record in records if str(record.get("question_id")) in active_ids]


def have_complete_records(records, questions):
    active_ids = {str(question["question_id"]) for question in questions}
    record_ids = {str(record.get("question_id")) for record in records}
    missing_ids = sorted(active_ids - record_ids)
    return len(missing_ids) == 0, missing_ids


def write_mmvet_outputs(records, response_file, eval_payload_file, local_summary_file):
    write_mmvet_response_file(records, response_file)
    write_mmvet_eval_payload(records, eval_payload_file)
    write_mmvet_local_summary(records, local_summary_file)


def save_intervention_stats(model, args):
    if not args.log_intervention_stats:
        return
    stats = finalize_intervention_stats(getattr(model.config, "_intervention_stats", None))
    stats["config"] = {
        "intervention": args.intervention,
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
    }
    out_file = args.intervention_stats_file or os.path.join(
        os.path.dirname(os.path.expanduser(args.answers_file)), "intervention_stats.json"
    )
    with open(os.path.expanduser(out_file), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


def eval_model(args):
    questions = load_mmvet_questions(args)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    response_file = args.response_file or os.path.join(os.path.dirname(answers_file), "mmvet_responses.json")
    eval_payload_file = args.eval_payload_file or os.path.join(os.path.dirname(answers_file), "mmvet_eval_payload.json")
    local_summary_file = args.local_summary_file or os.path.join(os.path.dirname(answers_file), "mmvet_local_summary.json")

    existing_records = []
    resume_info = {}

    if args.skip_generation or args.skip_generation_if_complete or args.resume:
        existing_records, resume_info = load_existing_answer_records(answers_file)
        existing_records = filter_records_for_questions(existing_records, questions)
        resume_info["active_valid"] = len(existing_records)

    complete_existing, missing_ids = have_complete_records(existing_records, questions)

    if args.skip_generation_if_complete and complete_existing:
        print(f"Found complete MM-Vet answers at {answers_file}; skipping generation.")
        write_mmvet_outputs(existing_records, response_file, eval_payload_file, local_summary_file)
        if args.run_official_eval:
            run_official_mmvet_eval(args, response_file)
        return

    if args.resume and resume_info.get("found"):
        rewrite_answer_records(answers_file, existing_records)

    processed_ids = {str(record["question_id"]) for record in existing_records}
    pending_questions = [
        question for question in questions
        if str(question["question_id"]) not in processed_ids
    ]

    if args.skip_generation:
        if not complete_existing:
            preview = ", ".join(missing_ids[:10])
            raise ValueError(
                f"--skip-generation requested, but {len(missing_ids)} MM-Vet answers are missing. "
                f"First missing ids: {preview}"
            )
        write_mmvet_outputs(existing_records, response_file, eval_payload_file, local_summary_file)
        if args.run_official_eval:
            run_official_mmvet_eval(args, response_file)
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
        system_prompt=args.system_prompt,
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
        for (input_ids, image_tensor, image_sizes, prompts), line in tqdm(
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
                "question_id": line["question_id"],
                "image": line["image"],
                "question": line["question"],
                "prompt": line["text"],
                "full_prompt": prompts[0],
                "answer": line.get("answer", ""),
                "capability": line.get("capability", []),
                "capability_combo": line.get("capability_combo", _combo_name(line.get("capability", []))),
                "imagesource": line.get("imagesource", ""),
                "answer_operator": line.get("answer_operator", _answer_operator(line.get("answer", ""))),
                "text": outputs,
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

    write_mmvet_outputs(records, response_file, eval_payload_file, local_summary_file)
    save_intervention_stats(model, args)

    if args.run_official_eval:
        result = run_official_mmvet_eval(args, response_file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Saved MM-Vet response file -> {response_file}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="../dataset/mm-vet/images")
    parser.add_argument("--mmvet-root", type=str, default="../third_party/MM-Vet")
    parser.add_argument("--mmvet-data-root", type=str, default="../dataset/mm-vet")
    parser.add_argument("--question-file", type=str, default="../dataset/mm-vet/mm-vet.json")
    parser.add_argument("--answers-file", type=str, default="results_mmvet/answers.jsonl")
    parser.add_argument("--response-file", type=str, default="")
    parser.add_argument("--eval-payload-file", type=str, default="")
    parser.add_argument("--local-summary-file", type=str, default="")
    parser.add_argument("--metrics-file", type=str, default="")
    parser.add_argument("--official-stdout-file", type=str, default="")
    parser.add_argument("--official-stderr-file", type=str, default="")
    parser.add_argument("--run-official-eval", action="store_true")
    parser.add_argument("--strict-official-eval", action="store_true")
    parser.add_argument("--official-eval-command", type=str, default="")
    parser.add_argument("--official-gpt-model", type=str, default="gpt-4-0613")
    parser.add_argument("--official-num-run", type=int, default=1)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-generation-if-complete", action="store_true")

    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--system-prompt", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--prompt-template", type=str, default="{question}")
    parser.add_argument("--prompt-style", type=str, default="exact", choices=["exact", "direct", "short"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--infer-image-start", action="store_true", default=True)
    parser.add_argument("--no-infer-image-start", dest="infer_image_start", action="store_false")

    parser.add_argument("--intervention", type=str, default="none", choices=["none", "adhh", "soft", "dynamic"])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--head-source", type=str, default="default", choices=["default", "file"])
    parser.add_argument("--head-file", type=str, default="")
    parser.add_argument("--head-score-key", type=str, default="score")
    parser.add_argument("--head-score-normalize", type=str, default="minmax", choices=["minmax", "raw", "logminmax", "rank_percentile"])
    parser.add_argument("--use-head-scores", action="store_true")

    parser.add_argument("--text-threshold", type=float, default=0.4)
    parser.add_argument("--text-scale", type=float, default=0.5)
    parser.add_argument("--gate-floor", type=float, default=0.0)
    parser.add_argument("--gate-midpoint", type=float, default=0.35)
    parser.add_argument("--gate-sharpness", type=float, default=12.0)
    parser.add_argument("--gate-score-power", type=float, default=1.0)
    parser.add_argument("--gate-hard-threshold", type=float, default=0.65)

    parser.add_argument("--dynamic-strength", type=float, default=1.0)
    parser.add_argument("--dynamic-ratio-power", type=float, default=1.0)
    parser.add_argument("--dynamic-score-power", type=float, default=1.0)
    parser.add_argument("--dynamic-context-mode", type=str, default="ratio_exp", choices=["text_exp", "ratio_exp", "ratio_power", "text_power"])
    parser.add_argument("--dynamic-tau", type=float, default=0.9)
    parser.add_argument("--dynamic-exp-sharpness", type=float, default=6.0)

    parser.add_argument("--log-intervention-stats", action="store_true")
    parser.add_argument("--intervention-stats-file", type=str, default="")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
