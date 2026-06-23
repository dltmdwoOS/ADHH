import argparse
import json
import os


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value, digits=3):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def load_chair_metrics(method_dir):
    data = load_json(os.path.join(method_dir, "captions_eval_results.json"))
    if not data:
        return {}
    return data.get("overall_metrics", {})


def main(args):
    summary = load_json(os.path.join(args.output_dir, "latency_memory_summary.json"))
    if not summary:
        raise FileNotFoundError(os.path.join(args.output_dir, "latency_memory_summary.json"))

    rows = []
    for item in summary["methods"]:
        method = item["method"]
        chair = load_chair_metrics(os.path.join(args.output_dir, method))
        rows.append(
            {
                "Method": method,
                "CHAIRs": chair.get("CHAIRs"),
                "CHAIRi": chair.get("CHAIRi"),
                "sec/image": item.get("seconds_per_image"),
                "tokens/s": item.get("tokens_per_second"),
                "runtime x": item.get("runtime_multiplier_vs_greedy"),
                "peak alloc delta GB": item.get("peak_generation_delta_allocated_gb"),
                "peak reserved delta GB": item.get("peak_generation_delta_reserved_gb"),
                "memory delta x": item.get("memory_delta_multiplier_vs_greedy"),
                "mean gen tokens": item.get("mean_generated_tokens"),
            }
        )

    markdown = [
        "| Method | CHAIRs | CHAIRi | sec/image | tokens/s | runtime x | peak alloc delta GB | peak reserved delta GB | memory delta x | mean gen tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {Method} | {CHAIRs} | {CHAIRi} | {sec} | {tok} | {rt} | {alloc} | {reserved} | {memx} | {gentok} |".format(
                Method=row["Method"],
                CHAIRs=fmt(row["CHAIRs"]),
                CHAIRi=fmt(row["CHAIRi"]),
                sec=fmt(row["sec/image"]),
                tok=fmt(row["tokens/s"]),
                rt=fmt(row["runtime x"], 2),
                alloc=fmt(row["peak alloc delta GB"]),
                reserved=fmt(row["peak reserved delta GB"]),
                memx=fmt(row["memory delta x"], 2),
                gentok=fmt(row["mean gen tokens"], 1),
            )
        )

    out = {
        "rows": rows,
        "markdown": "\n".join(markdown),
        "notes": [
            "Latency is measured only around model.generate with CUDA synchronization.",
            "CHAIR evaluation is run after caption generation and is not included in latency.",
            "Peak memory deltas are measured after model load and warmup, relative to the pre-generation CUDA allocation baseline.",
        ],
    }
    out_path = os.path.join(args.output_dir, "paper_table_latency_memory.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(out["markdown"])
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    main(parser.parse_args())
