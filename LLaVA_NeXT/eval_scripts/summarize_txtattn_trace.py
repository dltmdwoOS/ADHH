import argparse
import json
import os

from eval_scripts.eval_caption import (
    TxtAttnTraceStats,
    buckets_from_txtattn_record,
    load_txtattn_heads,
)


def summarize_trace(args):
    heads = load_txtattn_heads(args.head_file, args.topk)
    stats = TxtAttnTraceStats(heads)
    num_records = 0

    for trace_file in args.trace_file:
        trace_file = os.path.expanduser(trace_file)
        if not os.path.exists(trace_file):
            print(f"[txtattn-summary] skipping missing trace file: {trace_file}")
            continue
        with open(trace_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[txtattn-summary] ignoring malformed trace line {line_no} in {trace_file}")
                    continue
                head_values = record.get("head_values", [])
                if not head_values:
                    continue
                for item in head_values:
                    image_attn = float(item.get("image_attn", 0.0))
                    item["txt_img_ratio"] = float(item.get("I_text", 0.0)) / (image_attn + 1e-12)
                stats.update(buckets_from_txtattn_record(record), head_values)
                num_records += 1

    summary = stats.to_dict()
    summary["config"] = {
        "txtattn_head_file": args.head_file,
        "txtattn_topk": args.topk,
        "num_trace_records": num_records,
        "trace_files": args.trace_file,
        "note": "Rebuilt from txtattn trace jsonl file(s). I_text and txt_img_ratio use the full text-side region image_end:, i.e. question prompt plus generated text.",
    }

    summary_file = os.path.expanduser(args.summary_file)
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[txtattn-summary] wrote {summary_file} from {num_records} trace records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", required=True, nargs="+")
    parser.add_argument("--head-file", required=True)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--summary-file", required=True)
    summarize_trace(parser.parse_args())
