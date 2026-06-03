import argparse
import json
import math
import os

os.environ.setdefault("VILA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")

import random
import shutil
import uuid

import torch
from PIL import Image as PILImage
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import set_seed

import llava
from llava import conversation as clib
from llava.media import Image


def split_list(items, n):
    chunk_size = math.ceil(len(items) / n)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items, n, k):
    return split_list(items, n)[k]


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

    questions = []
    sampled_meta = []
    for idx, entry in enumerate(entries):
        question_id = entry.get("question_id", entry.get("image_id"))
        image_file = entry.get("image")
        if question_id is None or image_file is None:
            raise ValueError(f"Entry {idx} must contain image_id/question_id and image fields: {entry}")
        item = {
            "question_id": int(question_id),
            "image": image_file,
            "text": args.prompt_text,
        }
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
        raise ValueError("Only --dataset coco is supported for this CHAIR adapter.")
    if args.use_existing_sample_file:
        if not args.existing_sample_file:
            raise ValueError("--existing-sample-file is required when --use-existing-sample-file is set")
        return build_questions_from_existing_sample_file(args)

    coco = COCO(args.caption_file_path)
    sampled_img_ids = random.sample(coco.getImgIds(), args.num_samples)
    questions = []
    sampled_meta = []
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


def eval_model(args):
    model_path = os.path.expanduser(args.model_path)
    model = llava.load(
        model_path,
        model_base=args.model_base,
        attn_implementation=os.environ.get("VILA_ATTN_IMPLEMENTATION", "eager"),
    )
    model.eval()

    if args.conv_mode != "auto":
        clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    questions = build_questions(args)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    total = len(questions)
    completed = load_completed_question_ids(answers_file) if args.resume else set()
    if completed:
        questions = [q for q in questions if int(q["question_id"]) not in completed]
        print(f"[resume] {len(completed)} completed answers found; running {len(questions)}/{total} remaining.")

    generation_config = model.default_generation_config
    generation_updates = {
        "do_sample": bool(args.temperature > 0),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
    }
    generation_config.update(**{k: v for k, v in generation_updates.items() if v is not None})

    mode = "a" if args.resume else "w"
    with open(answers_file, mode, encoding="utf-8") as ans_file:
        for sample_idx, line in tqdm(enumerate(questions, start=1), total=len(questions)):
            image_path = os.path.join(args.image_folder, line["image"])
            with PILImage.open(image_path) as img:
                img.verify()
            with torch.inference_mode():
                output = model.generate_content(
                    [Image(image_path), line["text"]],
                    generation_config=generation_config,
                )
            output = str(output).strip()
            print(f"[{sample_idx}/{len(questions)}] question_id={line['question_id']}")
            print(output)
            ans_file.write(
                json.dumps(
                    {
                        "question_id": int(line["question_id"]),
                        "image": line["image"],
                        "prompt": line["text"],
                        "text": output,
                        "answer_id": str(uuid.uuid4()),
                        "model_id": args.model_name,
                        "metadata": {"model_path": args.model_path},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ans_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="vila")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--caption_file_path", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
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
    args = parser.parse_args()
    random.seed(args.seed)
    set_seed(args.seed)
    eval_model(args)
