#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from PIL import Image


def die(message):
    print(f"[error] {message}", file=sys.stderr)
    raise SystemExit(1)


def token_id(tokenizer, candidates):
    for token in candidates:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            return tid, token
    return None, None


def find_first(ids, value):
    try:
        return ids.index(value)
    except ValueError:
        return None


def find_last(ids, value):
    for idx in range(len(ids) - 1, -1, -1):
        if ids[idx] == value:
            return idx
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Sanity-check Qwen2.5-VL input spans for sys / vision / text-side attention intervention."
    )
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image", default="")
    parser.add_argument("--prompt", default="Please describe this image in detail.")
    args = parser.parse_args()

    try:
        from transformers import AutoProcessor
    except Exception as exc:
        die(
            "transformers is not importable. Use an environment with Qwen2.5-VL support, "
            "e.g. transformers>=4.49 plus qwen-vl-utils. Original import error: "
            f"{exc}"
        )

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = getattr(processor, "tokenizer", processor)

    image_path = Path(args.image) if args.image else None
    if image_path:
        image_obj = Image.open(image_path).convert("RGB")
    else:
        image_obj = Image.new("RGB", (224, 224), color=(128, 128, 128))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    processor_class = processor.__class__.__name__
    if (
        not hasattr(processor, "apply_chat_template")
        or not callable(getattr(processor, "apply_chat_template"))
        or "Tokenizer" in processor_class
    ):
        die(
            f"AutoProcessor resolved to {processor_class}, not a Qwen2.5-VL processor. "
            "This environment is too old for Qwen2.5-VL multimodal processing. "
            "Install a newer transformers build plus qwen-vl-utils before running the span check."
        )

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_obj], return_tensors="pt")
    ids = inputs["input_ids"][0].tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)

    vision_start_id, vision_start_tok = token_id(tokenizer, ["<|vision_start|>", "<|im_start|>"])
    vision_end_id, vision_end_tok = token_id(tokenizer, ["<|vision_end|>", "<|im_end|>"])
    image_pad_id, image_pad_tok = token_id(tokenizer, ["<|image_pad|>", "<image>", "<|image|>"])

    print(f"model_path: {args.model_path}")
    print(f"num_input_tokens: {len(ids)}")
    print(f"vision_start: {vision_start_tok} -> {vision_start_id}")
    print(f"vision_end:   {vision_end_tok} -> {vision_end_id}")
    print(f"image_pad:    {image_pad_tok} -> {image_pad_id}")
    print(f"image_grid_thw: {getattr(inputs.get('image_grid_thw', None), 'tolist', lambda: inputs.get('image_grid_thw', None))()}")

    if vision_start_id is None or vision_end_id is None:
        die("Could not find Qwen vision boundary tokens in the tokenizer.")

    vision_start = find_first(ids, vision_start_id)
    vision_end = find_last(ids, vision_end_id)
    if vision_start is None or vision_end is None or vision_end <= vision_start:
        die("Could not locate a valid vision span in input_ids.")

    image_pad_positions = []
    if image_pad_id is not None:
        image_pad_positions = [i for i, tid in enumerate(ids) if tid == image_pad_id]

    sys_span = (0, vision_start)
    vision_span = (vision_start, vision_end + 1)
    text_side_span = (vision_end + 1, len(ids))

    print("\n[span definition]")
    print(f"system:    [{sys_span[0]}, {sys_span[1]}) len={sys_span[1] - sys_span[0]}")
    print(f"vision:    [{vision_span[0]}, {vision_span[1]}) len={vision_span[1] - vision_span[0]}")
    print(f"text-side: [{text_side_span[0]}, {text_side_span[1]}) len={text_side_span[1] - text_side_span[0]}")
    if image_pad_positions:
        print(
            f"image_pad positions: first={image_pad_positions[0]}, "
            f"last={image_pad_positions[-1]}, count={len(image_pad_positions)}"
        )

    lo = max(0, vision_start - 12)
    hi = min(len(ids), vision_end + 24)
    print("\n[tokens around vision span]")
    for idx in range(lo, hi):
        marker = ""
        if idx == sys_span[1]:
            marker = " <SYS_END/VIS_START>"
        elif idx == vision_span[1]:
            marker = " <VIS_END/TEXT_START>"
        print(f"{idx:04d} {ids[idx]:8d} {toks[idx]}{marker}")


if __name__ == "__main__":
    main()
