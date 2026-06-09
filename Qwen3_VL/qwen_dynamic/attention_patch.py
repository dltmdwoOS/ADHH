from collections import defaultdict
from types import MethodType
import json

import torch
from torch import nn


def _group_heads(heads):
    grouped = defaultdict(list)
    for layer, head in heads or []:
        grouped[int(layer)].append(int(head))
    return dict(grouped)


def _lookup_head_score(score_map, layer_idx, head_idx):
    if not score_map:
        return 1.0
    return float(score_map.get(f"{int(layer_idx)}-{int(head_idx)}", score_map.get(f"L{int(layer_idx)}H{int(head_idx)}", 1.0)))


def _heads_for_layer(config, layer_idx):
    heads = getattr(config, "intervention_heads", None) or []
    score_map = getattr(config, "intervention_scores", None) or {}
    out = []
    for layer, head in heads:
        if int(layer) == int(layer_idx):
            out.append((int(head), _lookup_head_score(score_map, layer, head)))
    return out


def _maybe_trace_txtattn_last_row(attn_weights, config, layer_idx):
    if not bool(getattr(config, "enable_txtattn_last_row_trace", False)):
        return
    if layer_idx is None or attn_weights is None:
        return

    layer_idx = int(layer_idx)
    step_idx = int(getattr(config, "_txtattn_trace_current_step", 0))
    last_layer_idx = int(getattr(config, "num_hidden_layers", -1)) - 1

    def maybe_advance_step():
        if layer_idx == last_layer_idx:
            config._txtattn_trace_current_step = step_idx + 1

    heads_by_layer = getattr(config, "txtattn_trace_heads_by_layer", None) or {}
    heads = heads_by_layer.get(layer_idx, [])
    if not heads:
        maybe_advance_step()
        return

    img_start = getattr(config, "img_start_pos", None)
    img_length = getattr(config, "img_length", None)
    if img_start is None or img_length is None:
        maybe_advance_step()
        return

    img_start = int(img_start)
    img_length = int(img_length)
    img_end = img_start + img_length
    att_seq_len = int(attn_weights.size(-1))
    if img_start < 0 or img_length <= 0 or img_end > att_seq_len:
        maybe_advance_step()
        return

    row = attn_weights[0, :, -1, :]
    head_idx = torch.tensor([int(h) for h in heads], device=row.device, dtype=torch.long)
    if int(head_idx.max().item()) >= int(row.size(0)):
        maybe_advance_step()
        return
    selected_rows = row.index_select(0, head_idx).float()

    image_attn = selected_rows[:, img_start:img_end].sum(dim=-1)
    i_text = selected_rows[:, img_end:].sum(dim=-1)
    generated_start = min(max(int(getattr(config, "generated_start_pos", img_end)), img_end), att_seq_len)
    generated_txt_attn = selected_rows[:, generated_start:].sum(dim=-1)
    txt_img_ratio = i_text / (image_attn + 1e-12)

    buffer = getattr(config, "_txtattn_last_row_buffer", None)
    if buffer is None:
        buffer = []
        config._txtattn_last_row_buffer = buffer
    while len(buffer) <= step_idx:
        buffer.append({"layout": None, "head_values": []})

    record = buffer[step_idx]
    if record["layout"] is None:
        record["layout"] = {
            "prompt_visible_len": int(generated_start),
            "generated_len": int(max(att_seq_len - generated_start, 0)),
            "final_att_seq_len": int(att_seq_len),
            "image_start": int(img_start),
            "image_end": int(img_end),
            "image_len": int(img_length),
            "generated_start": int(generated_start),
            "sys_len": int(img_start),
            "txt_len": int(att_seq_len - img_end),
            "trace_mode": "last_row",
            "trace_note": "Qwen trace: system=[0, vision_start), vision=[vision_start, vision_end+1), text-side=[vision_end+1, current_seq_len).",
        }

    for idx, head in enumerate(heads):
        record["head_values"].append(
            {
                "layer": int(layer_idx),
                "head": int(head),
                "I_text": float(i_text[idx].detach().cpu().item()),
                "generated_txt_attn": float(generated_txt_attn[idx].detach().cpu().item()),
                "image_attn": float(image_attn[idx].detach().cpu().item()),
                "txt_img_ratio": float(txt_img_ratio[idx].detach().cpu().item()),
            }
        )

    maybe_advance_step()


def _maybe_reset_dynamic_trace(config, layer_idx):
    if getattr(config, "intervention", "none") not in ("dynamic", "late_boost"):
        return
    if not bool(getattr(config, "log_dynamic_trace", False)):
        return
    if int(layer_idx) == 0:
        config._dynamic_trace_buffer = []


def _append_dynamic_trace(config, layer_idx, head_idx, head_score, score_prior, text_mass, img_mass, ratio, context_source, context_prior, suppression, scale):
    if not bool(getattr(config, "log_dynamic_trace", False)):
        return
    buf = getattr(config, "_dynamic_trace_buffer", None)
    if buf is None:
        buf = []
        config._dynamic_trace_buffer = buf
    buf.append({
        "layer": int(layer_idx),
        "head": int(head_idx),
        "head_score": float(head_score),
        "score_prior": float(score_prior),
        "text_mass": float(text_mass.detach().float().mean().cpu().item()),
        "img_mass": float(img_mass.detach().float().mean().cpu().item()),
        "ratio": float(ratio.detach().float().mean().cpu().item()),
        "context_source": float(context_source.detach().float().mean().cpu().item()),
        "context_prior": float(context_prior.detach().float().mean().cpu().item()),
        "suppression": float(suppression.detach().float().mean().cpu().item()),
        "scale": float(scale.detach().float().mean().cpu().item()),
    })


def _maybe_print_dynamic_trace(config, layer_idx):
    if getattr(config, "intervention", "none") not in ("dynamic", "late_boost"):
        return
    if not bool(getattr(config, "log_dynamic_trace", False)):
        return
    last_layer_idx = int(getattr(config, "num_hidden_layers", -1)) - 1
    if int(layer_idx) != last_layer_idx:
        return
    step = int(getattr(config, "_dynamic_trace_step", 0))
    every = max(int(getattr(config, "dynamic_trace_every", 1)), 1)
    buf = list(getattr(config, "_dynamic_trace_buffer", []) or [])
    if buf and step % every == 0:
        topn = max(int(getattr(config, "dynamic_trace_topn", 10)), 1)
        top = sorted(buf, key=lambda x: x["suppression"], reverse=True)[:topn]
        def mean(k):
            return sum(float(x[k]) for x in buf) / max(len(buf), 1)
        print("[DYNAMIC_TRACE] " + json.dumps({
            "step": step,
            "num_heads": len(buf),
            "mean_suppression": mean("suppression"),
            "mean_scale": mean("scale"),
            "mean_ratio": mean("ratio"),
            "mean_text_mass": mean("text_mass"),
            "mean_img_mass": mean("img_mass"),
            "mean_context_prior": mean("context_prior"),
            "top": top,
        }, ensure_ascii=False), flush=True)
    config._dynamic_trace_step = step + 1
    config._dynamic_trace_buffer = []


def _update_intervention_stats(config, layer_idx, head_idx, mode, scale, text_mass, head_score, extra=None):
    if not bool(getattr(config, "log_intervention_stats", False)):
        return
    stats = getattr(config, "_intervention_stats", None)
    if stats is None:
        stats = {"overall": {}, "by_head": {}}
        config._intervention_stats = stats

    def update_bucket(bucket):
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
        if extra:
            for name, tensor in extra.items():
                bucket[f"sum_{name}"] = bucket.get(f"sum_{name}", 0.0) + float(tensor.detach().float().sum().item())

    overall = stats["overall"].setdefault(mode, {})
    key = f"{int(layer_idx)}-{int(head_idx)}"
    by_head = stats["by_head"].setdefault(key, {"layer": int(layer_idx), "head": int(head_idx), "mode": mode, "head_score": float(head_score)})
    update_bucket(overall)
    update_bucket(by_head)


def _apply_text_intervention(attn_weights, config, layer_idx):
    mode = getattr(config, "intervention", "none")
    if mode == "none":
        return attn_weights
    if mode != "dynamic":
        return attn_weights

    _maybe_reset_dynamic_trace(config, layer_idx)
    selected = _heads_for_layer(config, layer_idx)
    if not selected:
        _maybe_print_dynamic_trace(config, layer_idx)
        return attn_weights

    img_start = getattr(config, "img_start_pos", None)
    img_length = getattr(config, "img_length", None)
    if img_start is None or img_length is None:
        _maybe_print_dynamic_trace(config, layer_idx)
        return attn_weights

    img_start = int(img_start)
    img_end = img_start + int(img_length)
    text_start = img_end
    if text_start >= int(attn_weights.size(-1)):
        _maybe_print_dynamic_trace(config, layer_idx)
        return attn_weights

    dynamic_strength = float(getattr(config, "dynamic_strength", 1.0))
    dynamic_ratio_power = float(getattr(config, "dynamic_ratio_power", 1.0))
    dynamic_score_power = float(getattr(config, "dynamic_score_power", 1.0))
    dynamic_tau = float(getattr(config, "dynamic_tau", 0.5))
    dynamic_exp_sharpness = float(getattr(config, "dynamic_exp_sharpness", 6.0))
    dynamic_late_boost_start = int(getattr(config, "dynamic_late_boost_start", -1))
    dynamic_late_boost_end = int(getattr(config, "dynamic_late_boost_end", 128))
    dynamic_late_boost_mode = getattr(config, "dynamic_late_boost_mode", "linear")
    dynamic_late_tau = float(getattr(config, "dynamic_late_tau", -1.0))
    dynamic_context_mode = getattr(config, "dynamic_context_mode", "ratio_exp")
    dynamic_redistribute = getattr(config, "dynamic_redistribute", "renorm")
    dynamic_renorm = bool(getattr(config, "dynamic_renorm", True))
    use_head_scores = bool(getattr(config, "use_head_scores", False))
    eps = 1e-6
    generation_step = int(getattr(config, "_dynamic_trace_step", 0))
    effective_dynamic_tau = dynamic_tau
    if mode == "late_boost" and dynamic_late_boost_start >= 0 and dynamic_late_tau >= 0.0 and generation_step >= dynamic_late_boost_start:
        if dynamic_late_boost_mode == "linear":
            late_end = max(dynamic_late_boost_end, dynamic_late_boost_start + 1)
            progress = min(max((generation_step - dynamic_late_boost_start) / max(late_end - dynamic_late_boost_start, 1), 0.0), 1.0)
            effective_dynamic_tau = dynamic_tau + progress * (dynamic_late_tau - dynamic_tau)
        else:
            effective_dynamic_tau = dynamic_late_tau
    config._dynamic_effective_tau = effective_dynamic_tau

    for head, head_score in selected:
        if int(head) >= int(attn_weights.size(1)):
            continue
        text_slice = attn_weights[:, head, -1, text_start:]
        text_mass = text_slice.sum(dim=-1)
        img_mass = attn_weights[:, head, -1, img_start:img_end].sum(dim=-1)
        ratio = (text_mass / (text_mass + img_mass + eps)).clamp(0, 1)
        text_context = text_mass.clamp(0, 1)

        score_prior = head_score if use_head_scores else 1.0
        score_prior = max(float(score_prior), 0.0) ** max(dynamic_score_power, 0.0)

        if dynamic_context_mode == "ratio_power":
            context_source = ratio
            context_prior = ratio.pow(max(dynamic_ratio_power, 0.0))
        elif dynamic_context_mode == "ratio_exp":
            context_source = ratio
            context_prior = torch.exp(max(dynamic_exp_sharpness, 0.0) * (ratio - effective_dynamic_tau))
        elif dynamic_context_mode == "text_exp":
            context_source = text_context
            context_prior = torch.exp(max(dynamic_exp_sharpness, 0.0) * (text_context - effective_dynamic_tau))
        else:
            context_source = text_context
            context_prior = text_context.pow(max(dynamic_ratio_power, 0.0))

        suppression = (dynamic_strength * score_prior * context_prior).clamp(0, 1)
        scale = 1.0 - suppression
        scaled_text_slice = text_slice * scale.unsqueeze(-1)

        if dynamic_redistribute in ("system", "system_only", "vision", "vision_only", "sysvis"):
            removed_mass = (text_mass - scaled_text_slice.sum(dim=-1)).clamp_min(0.0)
            if dynamic_redistribute in ("system", "system_only") and img_start > 0:
                target_slice = attn_weights[:, head, -1, :img_start]
                target_mass = target_slice.sum(dim=-1).clamp_min(eps)
                attn_weights[:, head, -1, :img_start] = target_slice + target_slice * (removed_mass / target_mass).unsqueeze(-1)
            elif dynamic_redistribute in ("vision", "vision_only") and img_end > img_start:
                target_slice = attn_weights[:, head, -1, img_start:img_end]
                target_mass = target_slice.sum(dim=-1).clamp_min(eps)
                attn_weights[:, head, -1, img_start:img_end] = target_slice + target_slice * (removed_mass / target_mass).unsqueeze(-1)
            elif dynamic_redistribute == "sysvis" and img_end > img_start:
                target_mass = (attn_weights[:, head, -1, :img_start].sum(dim=-1) + attn_weights[:, head, -1, img_start:img_end].sum(dim=-1)).clamp_min(eps)
                if img_start > 0:
                    target_slice = attn_weights[:, head, -1, :img_start]
                    attn_weights[:, head, -1, :img_start] = target_slice + (target_slice / target_mass.unsqueeze(-1)) * removed_mass.unsqueeze(-1)
                target_slice = attn_weights[:, head, -1, img_start:img_end]
                attn_weights[:, head, -1, img_start:img_end] = target_slice + (target_slice / target_mass.unsqueeze(-1)) * removed_mass.unsqueeze(-1)

        attn_weights[:, head, -1, text_start:] = scaled_text_slice
        _update_intervention_stats(config, layer_idx, head, mode, scale, text_mass, head_score, {
            "img_mass": img_mass,
            "ratio": ratio,
            "context_source": context_source,
            "context_prior": context_prior,
            "dynamic_suppression": suppression,
        })
        _append_dynamic_trace(config, layer_idx, head, head_score, score_prior, text_mass, img_mass, ratio, context_source, context_prior, suppression, scale)

    if dynamic_renorm:
        denom = attn_weights[:, :, -1, :].sum(dim=-1, keepdim=True).clamp_min(eps)
        attn_weights[:, :, -1, :] = attn_weights[:, :, -1, :] / denom
    _maybe_print_dynamic_trace(config, layer_idx)
    return attn_weights


def _repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    bsz, num_kv, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_kv * n_rep, slen, head_dim)


def _manual_attention_from_states(module, query_states, key_states, value_states, attention_mask):
    key_states = _repeat_kv(key_states, module.num_key_value_groups)
    value_states = _repeat_kv(value_states, module.num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * module.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights, value_states


def _manual_qwen_forward(self, *args, **kwargs):
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_mod

    hidden_states = args[0] if args else kwargs.pop("hidden_states")
    attention_mask = kwargs.get("attention_mask", None)
    past_key_values = kwargs.get("past_key_values", kwargs.get("past_key_value", None))
    cache_position = kwargs.get("cache_position", None)
    output_attentions = bool(kwargs.get("output_attentions", False))
    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is None:
        raise RuntimeError("Qwen dynamic intervention requires position_embeddings from Qwen2.5-VL forward.")

    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    rope_cfg = getattr(self, "rope_scaling", None) or getattr(self.config, "rope_scaling", None) or getattr(self.config, "rope_parameters", None) or {}
    mrope_section = rope_cfg.get("mrope_section") if isinstance(rope_cfg, dict) else getattr(rope_cfg, "mrope_section", None)
    if mrope_section is None:
        raise RuntimeError("Qwen dynamic intervention could not locate mrope_section for multimodal RoPE.")
    query_states, key_states = qwen_mod.apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, mrope_section
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        try:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        except TypeError:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attn_output, attn_weights, _ = _manual_attention_from_states(self, query_states, key_states, value_states, attention_mask)
    attn_weights = _apply_text_intervention(attn_weights, self.config, self.layer_idx)
    _maybe_trace_txtattn_last_row(attn_weights[:, :, -1:, :], self.config, self.layer_idx)
    if getattr(self.config, "intervention", "none") != "none":
        # Only the final query row is intervened; keep earlier prompt rows from the original manual attention.
        key_states_rep = _repeat_kv(key_states, self.num_key_value_groups)
        value_states_rep = _repeat_kv(value_states, self.num_key_value_groups)
        last_output = torch.matmul(attn_weights[:, :, -1:, :], value_states_rep)
        attn_output[:, -1:, :, :] = last_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights if output_attentions else None


def _wrap_attention_forward(module):
    if getattr(module, "_ha_txtattn_wrapped", False):
        return False
    original_forward = module.forward

    def wrapped_forward(self, *args, **kwargs):
        config = getattr(self, "config", None)
        trace_active = bool(getattr(config, "enable_txtattn_last_row_trace", False)) if config is not None else False
        intervention_active = getattr(config, "intervention", "none") != "none" if config is not None else False
        if intervention_active:
            return _manual_qwen_forward(self, *args, **kwargs)

        requested_output_attentions = bool(kwargs.get("output_attentions", False))
        if trace_active:
            kwargs["output_attentions"] = True

        out = original_forward(*args, **kwargs)
        if trace_active:
            if isinstance(out, tuple) and len(out) >= 2:
                attn_weights = out[1]
                _maybe_trace_txtattn_last_row(attn_weights, config, getattr(self, "layer_idx", None))
                if not requested_output_attentions:
                    out = (out[0], None, *out[2:])
            else:
                print(f"[txtattn-warning] attention module {self.__class__.__name__} did not return attn weights")
        return out

    module.forward = MethodType(wrapped_forward, module)
    module._ha_txtattn_wrapped = True
    return True


def install_qwen25_attention_patch():
    try:
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl  # noqa: F401
    except Exception as exc:
        raise RuntimeError("Could not import Qwen2.5-VL modeling code. Use a transformers version with Qwen2.5-VL support.") from exc
    return None


def patch_qwen_attention_modules(model):
    patched = 0
    candidates = 0
    for module in model.modules():
        if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")) and hasattr(module, "layer_idx"):
            candidates += 1
            if _wrap_attention_forward(module):
                patched += 1
    print(f"[txtattn-patch] wrapped {patched}/{candidates} Qwen text attention modules")
    if candidates == 0:
        print("[txtattn-warning] found no Qwen text attention modules to wrap")
    return patched


def _all_configs(model):
    configs = []
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and cfg not in configs:
            configs.append(cfg)
    cfg = getattr(model, "config", None)
    if cfg is not None and cfg not in configs:
        configs.append(cfg)
    return configs


def configure_txtattn_trace(model, heads, img_start_pos, img_length, generated_start_pos=None):
    grouped = _group_heads(heads)
    shared_buffer = []
    for cfg in _all_configs(model):
        cfg.enable_txtattn_last_row_trace = True
        cfg.txtattn_trace_heads_by_layer = grouped
        cfg.img_start_pos = int(img_start_pos)
        cfg.img_length = int(img_length)
        cfg.generated_start_pos = int(generated_start_pos if generated_start_pos is not None else img_start_pos + img_length)
        cfg._txtattn_last_row_buffer = shared_buffer
        cfg._txtattn_trace_current_step = 0


def configure_dynamic_intervention(model, heads, scores, img_start_pos, img_length, args):
    shared_stats = None
    for cfg in _all_configs(model):
        existing_stats = getattr(cfg, "_intervention_stats", None)
        if existing_stats is not None:
            shared_stats = existing_stats
            break
    if shared_stats is None:
        shared_stats = {"overall": {}, "by_head": {}}
    shared_dynamic_buffer = []
    for cfg in _all_configs(model):
        cfg.intervention = args.intervention
        cfg.intervention_heads = heads
        cfg.intervention_scores = scores
        cfg.img_start_pos = int(img_start_pos)
        cfg.img_length = int(img_length)
        cfg.generated_start_pos = int(img_start_pos + img_length)
        cfg.dynamic_strength = args.dynamic_strength
        cfg.dynamic_ratio_power = args.dynamic_ratio_power
        cfg.dynamic_score_power = args.dynamic_score_power
        cfg.dynamic_tau = args.dynamic_tau
        cfg.dynamic_exp_sharpness = args.dynamic_exp_sharpness
        cfg.dynamic_late_boost_start = args.dynamic_late_boost_start
        cfg.dynamic_late_boost_end = args.dynamic_late_boost_end if args.dynamic_late_boost_end > 0 else getattr(args, "max_new_tokens", 128)
        cfg.dynamic_late_boost_mode = args.dynamic_late_boost_mode
        cfg.dynamic_late_tau = args.dynamic_late_tau
        cfg.dynamic_context_mode = args.dynamic_context_mode
        cfg.dynamic_redistribute = args.dynamic_redistribute
        cfg.dynamic_renorm = args.dynamic_renorm
        cfg.use_head_scores = args.use_head_scores
        cfg.log_intervention_stats = args.log_intervention_stats
        cfg._intervention_stats = shared_stats
        cfg.log_dynamic_trace = args.log_dynamic_trace
        cfg.dynamic_trace_topn = args.dynamic_trace_topn
        cfg.dynamic_trace_every = args.dynamic_trace_every
        cfg._dynamic_trace_step = 0
        cfg._dynamic_trace_buffer = shared_dynamic_buffer



def disable_txtattn_trace(model):
    for cfg in _all_configs(model):
        cfg.enable_txtattn_last_row_trace = False


def disable_intervention(model):
    for cfg in _all_configs(model):
        cfg.intervention = "none"


def get_txtattn_trace_buffer(model):
    best = []
    for cfg in _all_configs(model):
        buffer = getattr(cfg, "_txtattn_last_row_buffer", None)
        if buffer and len(buffer) > len(best):
            best = buffer
    return best


def get_intervention_stats(model):
    for cfg in _all_configs(model):
        stats = getattr(cfg, "_intervention_stats", None)
        if stats:
            return stats
    return {"overall": {}, "by_head": {}}
