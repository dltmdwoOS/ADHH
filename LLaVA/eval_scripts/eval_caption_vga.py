import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import StoppingCriteriaList, set_seed

# Use the patched VGA implementation, not the local LLaVA package. VGA modifies
# the LLaMA attention/model forward signatures to accept vl_guidance/attn_coef.
VGA_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "VGA"
sys.path.insert(0, str(VGA_ROOT))

from llava.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates  # noqa: E402
from llava.mm_utils import (  # noqa: E402
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model  # noqa: E402
from llava.utils import disable_torch_init  # noqa: E402
from vcd_utils.greedy_sample import evolve_greedy_sampling, set_vga_tokenizer  # noqa: E402


evolve_greedy_sampling()


def split_list(items, n):
    chunk_size = math.ceil(len(items) / n)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def get_chunk(items, n, k):
    chunks = split_list(items, n)
    if k >= len(chunks):
        return []
    return chunks[k]


def load_completed_question_ids(path):
    completed = set()
    if not path or not os.path.exists(path):
        return completed
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[resume] ignoring malformed line {line_no} in {path}", flush=True)
                continue
            qid = row.get("question_id")
            if qid is not None:
                completed.add(int(qid))
    return completed


def load_questions(args):
    sample_path = os.path.expanduser(args.sample_id_file)
    if sample_path:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            entries = data.get("sentences") or data.get("samples") or data.get("images")
        else:
            entries = data
        if not entries:
            raise ValueError(f"No samples found in {sample_path}")

        coco = COCO(args.caption_file_path)
        questions = []
        for idx, entry in enumerate(entries):
            if isinstance(entry, int):
                image_id = int(entry)
                image = coco.loadImgs(image_id)[0]["file_name"]
            elif isinstance(entry, dict):
                image_id = int(entry.get("question_id", entry.get("image_id")))
                image = entry.get("image") or coco.loadImgs(image_id)[0]["file_name"]
            else:
                raise ValueError(f"Unsupported sample entry {idx}: {entry}")
            questions.append({"question_id": image_id, "image_id": image_id, "image": image, "text": args.prompt_text})
        print(f"Loaded {len(questions)} existing samples from {sample_path}", flush=True)
        return questions

    coco = COCO(args.caption_file_path)
    img_ids = coco.getImgIds()
    sampled = random.sample(img_ids, args.num_samples)
    return [
        {
            "question_id": int(image_id),
            "image_id": int(image_id),
            "image": coco.loadImgs(image_id)[0]["file_name"],
            "text": args.prompt_text,
        }
        for image_id in sampled
    ]


def build_prompt(args, model, question):
    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt(), conv


def eval_model(args):
    disable_torch_init()
    os.makedirs(os.path.dirname(os.path.expanduser(args.answers_file)), exist_ok=True)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)
    tokenizer.padding_side = "right"
    model.model.lm_head = model.lm_head
    set_vga_tokenizer(tokenizer)

    questions = load_questions(args)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    if args.num_samples and args.num_samples > 0:
        questions = questions[: args.num_samples]

    completed = load_completed_question_ids(args.answers_file) if args.resume else set()
    if completed:
        before = len(questions)
        questions = [q for q in questions if int(q["question_id"]) not in completed]
        print(f"[resume] skipped {before - len(questions)} completed samples from {args.answers_file}", flush=True)

    mode = "a" if args.resume else "w"
    ans_file = open(os.path.expanduser(args.answers_file), mode, encoding="utf-8")

    for sample_idx, line in tqdm(enumerate(questions, start=1), total=len(questions)):
        image_file = line["image"]
        question = line["text"]
        prompt, conv = build_prompt(args, model, question)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image_tensor = image_tensor.unsqueeze(0).half().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            outputs = model(
                input_ids[:, :-1],
                images=image_tensor,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.logits
            vis_logits = F.softmax(logits[0, args.image_start:args.image_end, :], dim=-1)
            top_k_scores, _ = torch.topk(vis_logits, args.guidance_topk, dim=-1)
            top_k_scores = top_k_scores.float()
            entropy = (-top_k_scores * torch.log(top_k_scores + 1e-8) / math.log(args.guidance_topk)).sum(-1)
            vl_guidance = (entropy / entropy.sum(0)).to(vis_logits.dtype)

            output_ids = model.generate(
                input_ids[:, -1:],
                images=image_tensor,
                past_key_values=outputs.past_key_values,
                vl_guidance=vl_guidance,
                vis_logits=vis_logits,
                cd_alpha=args.cd_alpha,
                add_layer=list(range(args.start_layer, args.end_layer + 1)),
                attn_coef=args.attn_coef,
                use_add=args.use_add,
                head_balancing=args.head_balancing,
                attn_norm=args.attn_norm,
                do_sample=True,
                sampling=args.sampling,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList([stopping_criteria]),
            )

        text = tokenizer.batch_decode(output_ids[:, 1:], skip_special_tokens=True)[0]
        text = text.split("ASSISTANT:")[-1].strip()
        if text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()

        print(f"[{sample_idx}/{len(questions)}] question_id={line['question_id']}")
        print(text, flush=True)
        ans_file.write(
            json.dumps(
                {
                    "question_id": int(line["question_id"]),
                    "image_id": int(line["image_id"]),
                    "image": image_file,
                    "prompt": question,
                    "model_input_prompt": prompt,
                    "text": text,
                    "output": text,
                    "model_id": model_name,
                    "metadata": {
                        "method": "VGA",
                        "start_layer": args.start_layer,
                        "end_layer": args.end_layer,
                        "attn_coef": args.attn_coef,
                        "head_balancing": args.head_balancing,
                        "cd_alpha": args.cd_alpha,
                        "use_add": args.use_add,
                        "attn_norm": args.attn_norm,
                        "sampling": args.sampling,
                        "image_start": args.image_start,
                        "image_end": args.image_end,
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        ans_file.flush()

    ans_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--caption-file-path", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--sample-id-file", type=str, default=None)
    parser.add_argument("--prompt-text", type=str, default="Please describe this image in detail.")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--image-start", type=int, default=35)
    parser.add_argument("--image-end", type=int, default=611)
    parser.add_argument("--guidance-topk", type=int, default=10)
    parser.add_argument("--start-layer", type=int, default=2)
    parser.add_argument("--end-layer", type=int, default=15)
    parser.add_argument("--attn-coef", type=float, default=0.2)
    parser.add_argument("--cd-alpha", type=float, default=0.02)
    parser.add_argument("--use-add", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--attn-norm", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--sampling", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--head-balancing", type=str, default="simg", choices=["vattn", "battn", "simg", "simv", "simb", "simb-simg", "none"])
    args = parser.parse_args()

    random.seed(args.seed)
    set_seed(args.seed)
    print(json.dumps(vars(args), indent=2), flush=True)
    eval_model(args)


if __name__ == "__main__":
    main()
