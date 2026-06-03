import argparse
import copy
import json
import math
import os

os.environ.setdefault("VILA_ATTN_IMPLEMENTATION", "eager")
os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")

import random
import uuid

import torch
from tqdm import tqdm
from transformers import set_seed
from pycocotools.coco import COCO

from transformers.models.llama import modeling_llama as llama_mod

from eval_scripts.eval_caption import (
    build_questions,
    generate_with_optional_attentions,
    load_completed_question_ids,
)
import llava
from llava import conversation as clib


def normalize_head_scores(raw_scores, mode):
    if mode == "raw":
        return [min(max(float(s), 0.0), 1.0) for s in raw_scores]
    if mode == "minmax":
        s_min, s_max = min(raw_scores), max(raw_scores)
        if abs(s_max - s_min) < 1e-8:
            return [1.0 for _ in raw_scores]
        return [(float(s) - s_min) / (s_max - s_min) for s in raw_scores]
    if mode == "logminmax":
        logged = [math.log1p(max(float(s), 0.0)) for s in raw_scores]
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


def load_selected_heads(head_file, topk, score_key="score", score_normalize="rank_percentile"):
    with open(os.path.expanduser(head_file), "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "heads" in data:
        records = data["heads"]
    elif isinstance(data, list):
        records = data
    elif isinstance(data, dict) and "hal_heads" in data:
        heads = [[int(l), int(h)] for l, h in data["hal_heads"][:topk]]
        return heads, {f"{l}-{h}": 1.0 for l, h in heads}
    else:
        raise ValueError(f"Unsupported head file format: {head_file}")
    top = records[:topk]
    heads = [[int(x["layer"]), int(x["head"])] if isinstance(x, dict) else [int(x[0]), int(x[1])] for x in top]
    if records and isinstance(records[0], dict):
        score_records = records if score_normalize in ("logminmax", "rank_percentile") else top
        raw_scores = [score_from_head_record(x, score_key) for x in score_records]
        norm_scores = normalize_head_scores(raw_scores, score_normalize)
        norm_by_head = {f"{int(x['layer'])}-{int(x['head'])}": float(ns) for x, ns in zip(score_records, norm_scores)}
        score_map = {f"{l}-{h}": norm_by_head.get(f"{l}-{h}", 1.0) for l, h in heads}
    else:
        score_map = {f"{l}-{h}": 1.0 for l, h in heads}
    return heads, score_map


def _heads_for_layer(config, layer_idx):
    by_layer = getattr(config, "_dynamic_heads_by_layer", None)
    if by_layer is None:
        by_layer = {}
        scores = getattr(config, "intervention_scores", {}) or {}
        for layer, head in getattr(config, "intervention_heads", []) or []:
            layer, head = int(layer), int(head)
            by_layer.setdefault(layer, []).append((head, float(scores.get(f"{layer}-{head}", 1.0))))
        config._dynamic_heads_by_layer = by_layer
    return by_layer.get(int(layer_idx), [])


def _update_stats(config, layer_idx, head_idx, scale, text_mass, head_score, extra):
    if not bool(getattr(config, "log_intervention_stats", False)):
        return
    stats = getattr(config, "_intervention_stats", None)
    if stats is None:
        stats = {"overall": {}, "by_head": {}}
        config._intervention_stats = stats

    def update(bucket):
        n = int(scale.numel())
        scale_f = scale.detach().float()
        text_f = text_mass.detach().float()
        suppression_f = 1.0 - scale_f
        bucket["count"] = bucket.get("count", 0) + n
        bucket["scaled_count"] = bucket.get("scaled_count", 0) + int((scale_f < 0.999).sum().item())
        bucket["near_zero_count"] = bucket.get("near_zero_count", 0) + int((scale_f <= 0.05).sum().item())
        bucket["sum_scale"] = bucket.get("sum_scale", 0.0) + float(scale_f.sum().item())
        bucket["sum_suppression"] = bucket.get("sum_suppression", 0.0) + float(suppression_f.sum().item())
        bucket["sum_text_mass"] = bucket.get("sum_text_mass", 0.0) + float(text_f.sum().item())
        bucket["min_scale"] = min(bucket.get("min_scale", float("inf")), float(scale_f.min().item()))
        bucket["max_scale"] = max(bucket.get("max_scale", float("-inf")), float(scale_f.max().item()))
        for name, tensor in (extra or {}).items():
            value_f = tensor.detach().float()
            bucket[f"sum_{name}"] = bucket.get(f"sum_{name}", 0.0) + float(value_f.sum().item())

    overall = stats["overall"].setdefault("dynamic", {})
    key = f"{int(layer_idx)}-{int(head_idx)}"
    by_head = stats["by_head"].setdefault(key, {"layer": int(layer_idx), "head": int(head_idx), "mode": "dynamic", "head_score": float(head_score)})
    update(overall)
    update(by_head)


def _maybe_reset_dynamic_trace(config, layer_idx):
    if getattr(config, "intervention", "none") != "dynamic":
        return
    if not bool(getattr(config, "log_dynamic_trace", False)):
        return
    if int(layer_idx or 0) == 0:
        config._dynamic_trace_buffer = []


def _append_dynamic_trace(
    config, layer_idx, head_idx, head_score, score_prior, text_mass, img_mass,
    ratio, context_source, context_prior, suppression, scale
):
    if not bool(getattr(config, "log_dynamic_trace", False)):
        return

    buffer = getattr(config, "_dynamic_trace_buffer", None)
    if buffer is None:
        buffer = []
        config._dynamic_trace_buffer = buffer

    text_f = text_mass.detach().float().reshape(-1)
    img_f = img_mass.detach().float().reshape(-1)
    ratio_f = ratio.detach().float().reshape(-1)
    context_f = context_source.detach().float().reshape(-1)
    prior_f = context_prior.detach().float().reshape(-1)
    suppression_f = suppression.detach().float().reshape(-1)
    scale_f = scale.detach().float().reshape(-1)

    for batch_idx in range(suppression_f.numel()):
        buffer.append({
            "layer": int(layer_idx),
            "head": int(head_idx),
            "batch_idx": int(batch_idx),
            "suppression": float(suppression_f[batch_idx].item()),
            "scale": float(scale_f[batch_idx].item()),
            "ratio": float(ratio_f[batch_idx].item()),
            "text_mass": float(text_f[batch_idx].item()),
            "img_mass": float(img_f[batch_idx].item()),
            "context_source": float(context_f[batch_idx].item()),
            "context_prior": float(prior_f[batch_idx].item()),
            "head_score": float(head_score),
            "score_prior": float(score_prior),
        })


def _maybe_print_dynamic_trace(config, layer_idx):
    if getattr(config, "intervention", "none") != "dynamic":
        return
    if not bool(getattr(config, "log_dynamic_trace", False)):
        return

    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is not None and int(layer_idx) != int(num_layers) - 1:
        return

    step = int(getattr(config, "_dynamic_trace_step", 0))
    every = max(int(getattr(config, "dynamic_trace_every", 1)), 1)
    buffer = list(getattr(config, "_dynamic_trace_buffer", []) or [])

    if step % every == 0 and buffer:
        topn = max(int(getattr(config, "dynamic_trace_topn", 10)), 1)
        active = [x for x in buffer if x["suppression"] >= 0.05]
        strong = [x for x in buffer if x["suppression"] >= 0.5]
        near_zero = [x for x in buffer if x["scale"] <= 0.05]
        top = sorted(buffer, key=lambda x: x["suppression"], reverse=True)[:topn]
        mean = lambda key, rows: sum(x[key] for x in rows) / max(len(rows), 1)
        payload = {
            "sample_id": getattr(config, "dynamic_trace_sample_id", None),
            "step": step,
            "candidate_heads": len(buffer),
            "active_heads": len(active),
            "strong_heads": len(strong),
            "near_zero_heads": len(near_zero),
            "context_mode": getattr(config, "dynamic_context_mode", None),
            "redistribute": getattr(config, "dynamic_redistribute", "renorm"),
            "strength": float(getattr(config, "dynamic_strength", 1.0)),
            "tau": float(getattr(config, "dynamic_tau", 0.5)),
            "exp_sharpness": float(getattr(config, "dynamic_exp_sharpness", 6.0)),
            "mean_suppression": mean("suppression", buffer),
            "mean_scale": mean("scale", buffer),
            "mean_ratio": mean("ratio", buffer),
            "mean_text_mass": mean("text_mass", buffer),
            "mean_img_mass": mean("img_mass", buffer),
            "mean_context_prior": mean("context_prior", buffer),
            "top": top,
        }
        print("[DYNAMIC_TRACE] " + json.dumps(payload, ensure_ascii=False), flush=True)

    config._dynamic_trace_step = step + 1
    config._dynamic_trace_buffer = []


def _apply_dynamic_intervention(attn_weights, config, layer_idx):
    if getattr(config, "intervention", "none") != "dynamic":
        return attn_weights
    _maybe_reset_dynamic_trace(config, layer_idx)
    selected = _heads_for_layer(config, layer_idx)
    if not selected:
        _maybe_print_dynamic_trace(config, layer_idx)
        return attn_weights
    img_start = int(getattr(config, "img_start_pos", 0))
    img_length = int(getattr(config, "img_length", 0))
    img_end = img_start + img_length
    text_start = img_end
    if img_length <= 0 or text_start >= attn_weights.size(-1):
        _maybe_print_dynamic_trace(config, layer_idx)
        return attn_weights

    strength = float(getattr(config, "dynamic_strength", 1.0))
    ratio_power = float(getattr(config, "dynamic_ratio_power", 1.0))
    score_power = float(getattr(config, "dynamic_score_power", 1.0))
    tau = float(getattr(config, "dynamic_tau", 0.9))
    sharpness = float(getattr(config, "dynamic_exp_sharpness", 8.0))
    context_mode = getattr(config, "dynamic_context_mode", "ratio_exp")
    redistribute = getattr(config, "dynamic_redistribute", "renorm")
    renorm = bool(getattr(config, "dynamic_renorm", True))
    use_head_scores = bool(getattr(config, "use_head_scores", False))
    eps = 1e-6

    pending = []
    for head, head_score in selected:
        head = int(head)
        if head >= attn_weights.size(1):
            continue
        text_slice = attn_weights[:, head, -1, text_start:]
        text_mass = text_slice.sum(dim=-1)
        img_mass = attn_weights[:, head, -1, img_start:img_end].sum(dim=-1)
        ratio = (text_mass / (text_mass + img_mass + eps)).clamp(0, 1)
        text_context = text_mass.clamp(0, 1)
        score_prior = (max(float(head_score), 0.0) if use_head_scores else 1.0) ** max(score_power, 0.0)
        if context_mode == "ratio_exp":
            context_source = ratio
            context_prior = torch.exp(max(sharpness, 0.0) * (ratio - tau))
        elif context_mode == "ratio_power":
            context_source = ratio
            context_prior = ratio.pow(max(ratio_power, 0.0))
        elif context_mode == "text_exp":
            context_source = text_context
            context_prior = torch.exp(max(sharpness, 0.0) * (text_context - tau))
        else:
            context_source = text_context
            context_prior = text_context.pow(max(ratio_power, 0.0))
        suppression = (strength * score_prior * context_prior).clamp(0, 1)
        scale = 1.0 - suppression

        row_before = attn_weights[:, head, -1, :]
        sys_mass_before = row_before[:, :img_start].sum(dim=-1)
        vision_mass_before = row_before[:, img_start:img_end].sum(dim=-1)
        row_sum_before = row_before.sum(dim=-1)
        scaled_text_slice = text_slice * scale.unsqueeze(-1)
        removed_mass = (text_mass - scaled_text_slice.sum(dim=-1)).clamp_min(0.0)

        if redistribute in ("system", "system_only") and img_start > 0:
            target = attn_weights[:, head, -1, :img_start]
            target_mass = target.sum(dim=-1).clamp_min(eps)
            attn_weights[:, head, -1, :img_start] = target + target * (removed_mass / target_mass).unsqueeze(-1)
        elif redistribute in ("vision", "vision_only") and img_end > img_start:
            target = attn_weights[:, head, -1, img_start:img_end]
            target_mass = target.sum(dim=-1).clamp_min(eps)
            attn_weights[:, head, -1, img_start:img_end] = target + target * (removed_mass / target_mass).unsqueeze(-1)
        elif redistribute == "sysvis" and img_end > img_start:
            target_mass = (attn_weights[:, head, -1, :img_start].sum(dim=-1) + attn_weights[:, head, -1, img_start:img_end].sum(dim=-1)).clamp_min(eps)
            if img_start > 0:
                target = attn_weights[:, head, -1, :img_start]
                attn_weights[:, head, -1, :img_start] = target + (target / target_mass.unsqueeze(-1)) * removed_mass.unsqueeze(-1)
            target = attn_weights[:, head, -1, img_start:img_end]
            attn_weights[:, head, -1, img_start:img_end] = target + (target / target_mass.unsqueeze(-1)) * removed_mass.unsqueeze(-1)

        attn_weights[:, head, -1, text_start:] = scaled_text_slice
        pending.append((head, scale, text_mass, head_score, {
            "img_mass": img_mass,
            "ratio": ratio,
            "context_source": context_source,
            "context_prior": context_prior,
            "dynamic_suppression": suppression,
            "sys_mass_before": sys_mass_before,
            "vision_mass_before": vision_mass_before,
            "text_mass_before": text_mass,
            "row_sum_before": row_sum_before,
            "removed_text_mass": removed_mass,
        }))
        _append_dynamic_trace(
            config, layer_idx, head, head_score, score_prior, text_mass, img_mass,
            ratio, context_source, context_prior, suppression, scale
        )

    if renorm:
        denom = attn_weights[:, :, -1, :].sum(dim=-1, keepdim=True).clamp_min(eps)
        attn_weights[:, :, -1, :] = attn_weights[:, :, -1, :] / denom

    for head, scale, text_mass, head_score, extra in pending:
        row_after = attn_weights[:, head, -1, :]
        extra = dict(extra)
        extra.update({
            "sys_mass_after": row_after[:, :img_start].sum(dim=-1),
            "vision_mass_after": row_after[:, img_start:img_end].sum(dim=-1),
            "text_mass_after": row_after[:, text_start:].sum(dim=-1),
            "row_sum_after": row_after.sum(dim=-1),
        })
        _update_stats(config, layer_idx, head, scale, text_mass, head_score, extra)
    _maybe_print_dynamic_trace(config, layer_idx)
    return attn_weights


def patch_llama_attention_once():
    if getattr(llama_mod.LlamaAttention, "_ha_dynamic_patched", False):
        return
    F = llama_mod.F
    nn = llama_mod.nn
    apply_rotary_pos_emb = llama_mod.apply_rotary_pos_emb
    repeat_kv = llama_mod.repeat_kv
    Cache = llama_mod.Cache
    logger = llama_mod.logger

    def dynamic_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.size()
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            query_states = torch.cat([F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            key_states = torch.cat([F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            value_states = torch.cat([F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is None:
            logger.warning_once("Computing RoPE inside patched LlamaAttention; pass position_embeddings in newer transformers.")
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = _apply_dynamic_intervention(attn_weights, self.config, self.layer_idx)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"attn_output should be {(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}")
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    llama_mod.LlamaAttention.forward = dynamic_forward
    llama_mod.LlamaAttention._ha_dynamic_patched = True


def attach_intervention_config(model, args):
    if args.head_source != "file":
        raise ValueError("VILA dynamic currently expects --head-source file")
    heads, score_map = load_selected_heads(args.head_file, args.topk, args.head_score_key, args.head_score_normalize)
    cfg = model.llm.config
    cfg.intervention = args.intervention
    cfg.intervention_heads = heads
    cfg.intervention_scores = score_map
    cfg.dynamic_strength = args.dynamic_strength
    cfg.dynamic_ratio_power = args.dynamic_ratio_power
    cfg.dynamic_score_power = args.dynamic_score_power
    cfg.dynamic_tau = args.dynamic_tau
    cfg.dynamic_exp_sharpness = args.dynamic_exp_sharpness
    cfg.dynamic_context_mode = args.dynamic_context_mode
    cfg.dynamic_redistribute = args.dynamic_redistribute
    cfg.dynamic_renorm = args.dynamic_renorm
    cfg.use_head_scores = args.use_head_scores
    cfg.log_intervention_stats = args.log_intervention_stats
    cfg.log_dynamic_trace = args.log_dynamic_trace
    cfg.dynamic_trace_topn = args.dynamic_trace_topn
    cfg.dynamic_trace_every = args.dynamic_trace_every
    cfg._intervention_stats = {"overall": {}, "by_head": {}}
    cfg._dynamic_trace_step = 0
    cfg._dynamic_trace_buffer = []
    cfg._dynamic_max_layer = max((int(l) for l, _ in heads), default=0)
    cfg._dynamic_heads_by_layer = None
    return {"heads": heads, "scores": score_map}


def set_sample_span_config(model, input_ids, inputs_embeds, question_id):
    image_token_id = int(model.tokenizer.media_token_ids["image"])
    prompt_ids = input_ids[0].detach().cpu().tolist()
    positions = [idx for idx, token_id in enumerate(prompt_ids) if int(token_id) == image_token_id]
    if len(positions) != 1:
        raise ValueError(f"Expected one image token, found {len(positions)}")
    image_start = int(positions[0])
    image_len = int(inputs_embeds.shape[1]) - (len(prompt_ids) - 1)
    if image_len <= 0:
        raise ValueError(f"Invalid image_len inferred for VILA dynamic: {image_len}")
    model.llm.config.img_start_pos = image_start
    model.llm.config.img_length = image_len
    model.llm.config.dynamic_trace_sample_id = int(question_id)


def generate_dynamic(model, image_path, prompt_text, generation_config, question_id):
    from eval_scripts.eval_caption import prepare_vila_inputs
    input_ids, inputs_embeds, attention_mask = prepare_vila_inputs(model, image_path, prompt_text)
    set_sample_span_config(model, input_ids, inputs_embeds, question_id)
    output_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        use_cache=True,
        output_attentions=False,
        output_scores=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
    )
    caption = model.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption


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


def save_intervention_stats(model, args):
    raw = getattr(model.llm.config, "_intervention_stats", None) or {"overall": {}, "by_head": {}}
    stats = {
        "overall": {mode: finalize_bucket(bucket) for mode, bucket in raw.get("overall", {}).items()},
        "by_head": {key: finalize_bucket(bucket) for key, bucket in raw.get("by_head", {}).items()},
    }
    if args.intervention_stats_file:
        os.makedirs(os.path.dirname(args.intervention_stats_file), exist_ok=True)
        with open(args.intervention_stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


def save_run_config(args, head_cfg):
    out_dir = os.path.dirname(os.path.expanduser(args.answers_file))
    os.makedirs(out_dir, exist_ok=True)
    run_cfg = vars(args).copy()
    run_cfg["selected_heads"] = head_cfg["heads"]
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)


def load_or_sample_ids(args):
    if args.sample_id_file and os.path.exists(args.sample_id_file):
        with open(args.sample_id_file, "r", encoding="utf-8") as f:
            return json.load(f)
    coco = COCO(args.caption_file_path)
    sampled_ids = random.sample(coco.getImgIds(), args.num_samples)
    if args.save_sample_id_file:
        os.makedirs(os.path.dirname(args.save_sample_id_file), exist_ok=True)
        with open(args.save_sample_id_file, "w", encoding="utf-8") as f:
            json.dump(sampled_ids, f, indent=2)
    return sampled_ids


def questions_from_ids(args, sampled_ids):
    coco = COCO(args.caption_file_path)
    id_to_img = {img["id"]: img for img in coco.dataset["images"]}
    questions = []
    for image_id in sampled_ids[: args.num_samples]:
        questions.append({"question_id": int(image_id), "image": id_to_img[int(image_id)]["file_name"], "text": args.prompt_text})
    from eval_scripts.eval_caption import get_chunk
    return get_chunk(questions, args.num_chunks, args.chunk_idx)


def eval_model(args):
    patch_llama_attention_once()
    model = llava.load(
        os.path.expanduser(args.model_path),
        model_base=args.model_base,
        attn_implementation=os.environ.get("VILA_ATTN_IMPLEMENTATION", "eager"),
    )
    model.eval()
    if args.conv_mode != "auto":
        clib.default_conversation = clib.conv_templates[args.conv_mode].copy()
    head_cfg = attach_intervention_config(model, args)
    save_run_config(args, head_cfg)

    sampled_ids = load_or_sample_ids(args)
    questions = questions_from_ids(args, sampled_ids)
    completed = load_completed_question_ids(args.answers_file) if args.resume else set()
    if completed:
        before = len(questions)
        questions = [q for q in questions if int(q["question_id"]) not in completed]
        print(f"[resume] {len(completed)} completed answers found; running {len(questions)}/{before} remaining.")

    generation_config = copy.deepcopy(model.default_generation_config)
    updates = {"do_sample": bool(args.temperature > 0), "temperature": args.temperature, "top_p": args.top_p, "num_beams": args.num_beams, "max_new_tokens": args.max_new_tokens}
    generation_config.update(**{k: v for k, v in updates.items() if v is not None})

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    mode = "a" if args.resume else "w"
    with open(args.answers_file, mode, encoding="utf-8") as ans_file:
        for sample_idx, line in tqdm(enumerate(questions, start=1), total=len(questions)):
            image_path = os.path.join(args.image_folder, line["image"])
            with torch.inference_mode():
                text = generate_dynamic(model, image_path, line["text"], generation_config, line["question_id"])
            print(f"[{sample_idx}/{len(questions)}] question_id={line['question_id']}")
            print(text)
            ans_file.write(json.dumps({
                "question_id": int(line["question_id"]),
                "image": line["image"],
                "prompt": line["text"],
                "text": text,
                "answer_id": str(uuid.uuid4()),
                "model_id": args.model_name,
                "metadata": {"model_path": args.model_path},
            }, ensure_ascii=False) + "\n")
            ans_file.flush()
    save_intervention_stats(model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="vila")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--caption_file_path", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
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
    parser.add_argument("--sample-id-file", type=str, default="")
    parser.add_argument("--save-sample-id-file", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--intervention", type=str, default="dynamic", choices=["none", "dynamic"])
    parser.add_argument("--head-source", type=str, default="file")
    parser.add_argument("--head-file", type=str, required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--head-score-key", type=str, default="global__itext_all__C_toi_HminusG")
    parser.add_argument("--head-score-normalize", type=str, default="rank_percentile")
    parser.add_argument("--use-head-scores", action="store_true")
    parser.add_argument("--dynamic-strength", type=float, default=1.0)
    parser.add_argument("--dynamic-ratio-power", type=float, default=1.0)
    parser.add_argument("--dynamic-score-power", type=float, default=1.0)
    parser.add_argument("--dynamic-tau", type=float, default=0.9)
    parser.add_argument("--dynamic-exp-sharpness", type=float, default=8.0)
    parser.add_argument("--dynamic-context-mode", type=str, default="ratio_exp")
    parser.add_argument("--dynamic-redistribute", type=str, default="renorm")
    parser.add_argument("--dynamic-renorm", dest="dynamic_renorm", action="store_true", default=True)
    parser.add_argument("--no-dynamic-renorm", dest="dynamic_renorm", action="store_false")
    parser.add_argument("--log-intervention-stats", action="store_true")
    parser.add_argument("--intervention-stats-file", type=str, default="")
    parser.add_argument("--log-dynamic-trace", action="store_true")
    parser.add_argument("--dynamic-trace-topn", type=int, default=10)
    parser.add_argument("--dynamic-trace-every", type=int, default=5)
    args = parser.parse_args()
    random.seed(args.seed)
    set_seed(args.seed)
    eval_model(args)
