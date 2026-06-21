import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import set_seed
from transformers.generation.logits_process import LogitsProcessorList

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
    PAITextCFGLogitsProcessor,
    attach_intervention_config,
    build_text_only_input_ids,
    finalize_intervention_stats,
    maybe_merge_resume_intervention_stats,
)
from eval_scripts.amber_object_metrics import compute_amber_generative_object_metrics


class Tee:
    """Write stdout both to terminal and decode.log."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


@contextlib.contextmanager
def tee_stdout_to_decode_log(path, enabled=True, append=False):
    """
    Mirror stdout to decode.log.

    This captures normal print/tqdm output and also captures dynamic trace lines
    if the LLaVA attention hook prints them to stdout.
    """
    if not enabled:
        yield
        return

    path = os.path.expanduser(path)
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    mode = "a" if append else "w"
    old_stdout = sys.stdout
    with open(path, mode, encoding="utf-8") as log_f:
        sys.stdout = Tee(old_stdout, log_f)
        try:
            yield
        finally:
            sys.stdout = old_stdout


def resolve_decode_log_file(args):
    if getattr(args, "decode_log_file", ""):
        return os.path.expanduser(args.decode_log_file)
    return os.path.join(
        os.path.dirname(os.path.expanduser(args.answers_file)) or ".",
        "decode.log",
    )


def ensure_eval_caption_dynamic_namespace(args):
    """
    eval_caption_dynamic.py was originally written for caption evaluation, so it
    expects some argparse fields that this AMBER script may not define.
    Fill them defensively to prevent Namespace AttributeError.
    """
    defaults = {
        # Used by eval_caption_dynamic.attach_intervention_config().
        "log_dynamic_trace": False,
        "dynamic_trace_topn": 10,
        "dynamic_trace_every": 1,

        # Compatibility with older helpers / configs.
        "gate_scale": 1.0,
        "baseline_start_layer": 0,
        "baseline_end_layer": 32,
        "pai_alpha": 0.2,
        "pai_gamma": 1.1,
        "pai_use_cfg": False,
        "vaf_enh_para": 1.15,
        "vaf_sup_para": 0.95,
        "tarac_alpha": 0.5,
        "tarac_beta": 0.5,
        "tarac_start_layer": 9,
        "tarac_end_layer": 16,

        # decode.log controls.
        "write_decode_log": True,
        "decode_log_file": "",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    return args


def log_decode_event(event, **kwargs):
    """Write one structured JSONL-style event into decode.log/stdout."""
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **kwargs,
    }
    print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def flush_dynamic_trace_buffer_if_any(model, sample_id):
    """
    Some dynamic-attention hooks print trace lines directly. Others may store
    trace objects in model.config._dynamic_trace_buffer. This logs the latter.
    """
    buffer = getattr(model.config, "_dynamic_trace_buffer", None)
    if not buffer:
        return

    for item in buffer:
        log_decode_event(
            "dynamic_trace_buffer",
            sample_id=int(sample_id),
            trace=item,
        )

    model.config._dynamic_trace_buffer = []


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
    out_dir = os.path.dirname(os.path.expanduser(args.answers_file)) or "."
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

    log_decode_event(
        "official_eval_start",
        command=cmd,
        cwd=amber_root,
        response_file=response_file,
    )

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

    official_metrics = parse_official_generative_stdout(proc.stdout)
    result = {
        "official_command": cmd,
        "official_cwd": amber_root,
        "returncode": proc.returncode,
        "stdout_file": stdout_path,
        "stderr_file": stderr_path,
        "metrics": official_metrics,
    }
    try:
        object_result = compute_amber_generative_object_metrics(
            response_file=response_file,
            amber_root=amber_root,
            official_metrics=official_metrics,
        )
        result["metrics"].update(object_result["metrics"])
        result["generative_object_metrics"] = {
            key: value for key, value in object_result.items() if key != "per_sample"
        }
    except Exception as exc:
        result["generative_object_metrics_error"] = repr(exc)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    log_decode_event(
        "official_eval_done",
        returncode=proc.returncode,
        stdout_file=stdout_path,
        stderr_file=stderr_path,
        metrics_file=metrics_path,
        metrics=result["metrics"],
    )

    if proc.returncode != 0 and args.strict_official_eval:
        raise RuntimeError(
            f"AMBER official evaluator failed with return code {proc.returncode}. "
            f"See {stderr_path}"
        )
    return result


def save_run_config(args, head_cfg, questions, pending_questions=None, resumed_records=0, resume_info=None):
    out_dir = os.path.dirname(os.path.expanduser(args.answers_file)) or "."
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
        "log_dynamic_trace": args.log_dynamic_trace,
        "dynamic_trace_topn": args.dynamic_trace_topn,
        "dynamic_trace_every": args.dynamic_trace_every,
        "decode_log_file": resolve_decode_log_file(args),
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
    out_file = args.intervention_stats_file or os.path.join(
        os.path.dirname(os.path.expanduser(args.answers_file)) or ".",
        "intervention_stats.json",
    )
    out_file = os.path.expanduser(out_file)
    stats = maybe_merge_resume_intervention_stats(stats, out_file, args)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    log_decode_event("intervention_stats_saved", path=out_file)


def eval_model(args):
    args = ensure_eval_caption_dynamic_namespace(args)
    questions = load_amber_generative_queries(args)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file) or ".", exist_ok=True)
    response_file = args.response_file or os.path.join(os.path.dirname(answers_file) or ".", "amber_responses.json")

    log_decode_event(
        "eval_model_start",
        answers_file=answers_file,
        response_file=response_file,
        num_loaded_questions=len(questions),
        skip_generation=args.skip_generation,
        resume=args.resume,
    )

    existing_records = []
    resume_info = {}
    if args.resume:
        existing_records, resume_info = load_existing_answer_records(answers_file)
        rewrite_answer_records(answers_file, existing_records)
        log_decode_event("resume_loaded", **resume_info)

    processed_ids = {int(record["id"]) for record in existing_records}
    pending_questions = [q for q in questions if int(q["id"]) not in processed_ids]

    if args.skip_generation:
        records = existing_records
        if not records:
            records, resume_info = load_existing_answer_records(answers_file)
            log_decode_event("skip_generation_loaded_records", **resume_info)
        write_official_response_file(records, response_file)
        log_decode_event(
            "official_response_file_saved",
            path=response_file,
            num_records=len(records),
        )
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

    log_decode_event(
        "run_start",
        model_path=args.model_path,
        model_id=model_name,
        answers_file=answers_file,
        num_questions=len(questions),
        num_pending_questions=len(pending_questions),
        resumed_records=len(existing_records),
        intervention=args.intervention,
        topk=args.topk,
        selected_heads=head_cfg["heads"],
        log_dynamic_trace=args.log_dynamic_trace,
        dynamic_trace_topn=args.dynamic_trace_topn,
        dynamic_trace_every=args.dynamic_trace_every,
        decode_log_file=resolve_decode_log_file(args),
    )

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
            sample_id = int(line["id"])

            # These fields match the eval_caption_dynamic.py dynamic trace convention.
            model.config.dynamic_trace_sample_id = sample_id
            model.config._dynamic_trace_step = 0
            model.config._dynamic_trace_buffer = []
            model.config._tarac_attn_memory = {}

            log_decode_event(
                "sample_start",
                id=sample_id,
                image=line["image"],
                query=line["query"],
                prompt=line["text"],
            )

            input_ids = input_ids.to(device="cuda", non_blocking=True)
            image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

            inferred_img_start = None
            if args.infer_image_start:
                inferred_img_start = update_image_start_from_input_ids(model, input_ids)

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
                    output_attentions=True,
                    return_dict_in_generate=True,
                    logits_processor=logits_processor,
                )

            output_ids = output_dict["sequences"]
            generated_ids = extract_generated_ids(output_ids, input_ids)
            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

            flush_dynamic_trace_buffer_if_any(model, sample_id)

            log_decode_event(
                "sample_done",
                id=sample_id,
                image=line["image"],
                response=outputs,
                inferred_img_start_pos=inferred_img_start,
                intervention=args.intervention,
                topk=args.topk,
            )

            record = {
                "id": sample_id,
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
                    "baseline_start_layer": args.baseline_start_layer,
                    "baseline_end_layer": args.baseline_end_layer,
                    "pai_alpha": args.pai_alpha,
                    "pai_gamma": args.pai_gamma,
                    "pai_use_cfg": args.pai_use_cfg,
                    "vaf_enh_para": args.vaf_enh_para,
                    "vaf_sup_para": args.vaf_sup_para,
                },
            }
            ans_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            ans_file.flush()
            records.append(record)

    write_official_response_file(records, response_file)
    log_decode_event(
        "official_response_file_saved",
        path=response_file,
        num_records=len(records),
    )

    save_intervention_stats(model, args)

    if args.run_official_eval:
        result = run_official_amber_eval(args, response_file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Saved AMBER official response file -> {response_file}")

    log_decode_event("eval_model_done", num_records=len(records))


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

    parser.add_argument("--intervention", type=str, default="none", choices=["none", "adhh", "soft", "dynamic", "late_boost", "pai", "vaf", "tarac"])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--head-source", type=str, default="default", choices=["default", "file"])
    parser.add_argument("--head-file", type=str, default="")
    parser.add_argument("--head-score-key", type=str, default="score")
    parser.add_argument("--head-score-normalize", type=str, default="minmax", choices=["minmax", "raw", "logminmax", "rank_percentile"])
    parser.add_argument(
        "--min-head-back-raw",
        type=float,
        default=0.0,
        help="For ranked head files with a back_raw field, drop heads inside the requested top-k whose back_raw is below this value.",
    )
    parser.add_argument("--use-head-scores", action="store_true")

    parser.add_argument("--text-threshold", type=float, default=0.4)
    parser.add_argument("--text-scale", type=float, default=0.5)
    parser.add_argument("--baseline-start-layer", type=int, default=0)
    parser.add_argument("--baseline-end-layer", type=int, default=32)
    parser.add_argument("--pai-alpha", type=float, default=0.2)
    parser.add_argument("--pai-gamma", type=float, default=1.1)
    parser.add_argument("--pai-use-cfg", action="store_true")
    parser.add_argument("--vaf-enh-para", type=float, default=1.15)
    parser.add_argument("--vaf-sup-para", type=float, default=0.95)
    parser.add_argument("--tarac-alpha", type=float, default=0.5)
    parser.add_argument("--tarac-beta", type=float, default=0.5)
    parser.add_argument("--tarac-start-layer", type=int, default=9)
    parser.add_argument("--tarac-end-layer", type=int, default=16)

    parser.add_argument("--dynamic-strength", type=float, default=1.0)
    parser.add_argument("--dynamic-ratio-power", type=float, default=1.0)
    parser.add_argument("--dynamic-score-power", type=float, default=1.0)
    parser.add_argument("--dynamic-context-mode", type=str, default="ratio_exp", choices=["text_exp", "ratio_exp", "ratio_power", "text_power"])
    parser.add_argument("--dynamic-tau", type=float, default=0.9)
    parser.add_argument("--dynamic-exp-sharpness", type=float, default=6.0)
    parser.add_argument(
        "--dynamic-late-boost-start",
        type=int,
        default=0,
        help="If >= 0, begin late-step tau scheduling from this generated-token step.",
    )
    parser.add_argument(
        "--dynamic-late-boost-end",
        type=int,
        default=128,
        help="Generated-token step where linear late tau reaches --dynamic-late-tau; <=0 uses --max_new_tokens.",
    )
    parser.add_argument(
        "--dynamic-late-boost-mode",
        type=str,
        default="linear",
        choices=["linear", "step"],
        help="Late tau schedule: linear decays from base tau to late tau, step switches immediately.",
    )
    parser.add_argument(
        "--dynamic-late-tau",
        type=float,
        default=0.80,
        help="Final late-step tau for late_boost; use a negative value to disable scheduling.",
    )
    parser.add_argument(
        "--dynamic-redistribute",
        type=str,
        default="renorm",
        choices=["none", "renorm", "system", "system_only", "vision", "vision_only", "sysvis"],
    )
    parser.add_argument(
        "--no-dynamic-renorm",
        dest="dynamic_renorm",
        action="store_false",
        help="Skip the final attention-row renormalization after dynamic text suppression.",
    )
    parser.set_defaults(dynamic_renorm=True)

    parser.add_argument("--log-intervention-stats", action="store_true")
    parser.add_argument("--intervention-stats-file", type=str, default="")

    # decode.log / dynamic trace compatibility with eval_caption_dynamic.py.
    parser.add_argument("--decode-log-file", type=str, default="")
    parser.add_argument("--no-decode-log", dest="write_decode_log", action="store_false")
    parser.set_defaults(write_decode_log=True)

    parser.add_argument("--log-dynamic-trace", dest="log_dynamic_trace", action="store_true", default=False)
    parser.add_argument("--no-log-dynamic-trace", dest="log_dynamic_trace", action="store_false")
    parser.add_argument("--dynamic-trace-topn", type=int, default=10)
    parser.add_argument("--dynamic-trace-every", type=int, default=10)

    # Kept hidden so imported helpers remain well-defined across script variants.
    parser.add_argument("--gate-scale", type=float, default=1.0, help=argparse.SUPPRESS)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args = ensure_eval_caption_dynamic_namespace(args)
    set_seed(args.seed)

    decode_log_file = resolve_decode_log_file(args)
    with tee_stdout_to_decode_log(
        decode_log_file,
        enabled=args.write_decode_log,
        append=args.resume,
    ):
        log_decode_event("decode_log_open", path=decode_log_file)
        eval_model(args)
        log_decode_event("decode_log_close", path=decode_log_file)
