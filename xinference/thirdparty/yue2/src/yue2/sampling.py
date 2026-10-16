"""Request-local sampling, preserving mode-specific historical arithmetic."""
from __future__ import annotations
import time
import torch
from .protocol import EOD, ABC_END, MUSIC_END, CODEC_OFFSET, CODEC_SIZE, CONTEXT


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def window_penalty(logits, recent_ids, penalty):
    if penalty == 1.0 or len(recent_ids) == 0:
        return logits
    recent = torch.as_tensor(recent_ids, dtype=torch.long, device=logits.device).reshape(1, -1)
    freq = torch.zeros_like(logits)
    freq.scatter_add_(-1, recent, torch.ones_like(recent, dtype=logits.dtype))
    alpha = penalty ** freq
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def distribution(logits, sampling, history, step, phase, legacy_off=False):
    # vLLM's symbolic processor receives FP32 logits; historical off uses BF16.
    scores = logits.clone() if legacy_off else logits.float().clone()
    end = ABC_END if phase == "abc" else MUSIC_END
    allowed = torch.full_like(scores, float("-inf"))
    if phase == "abc":
        allowed[..., :EOD] = 0
    else:
        allowed[..., CODEC_OFFSET:CODEC_OFFSET + CODEC_SIZE] = 0
    allowed[..., end] = 0
    scores = scores + allowed
    if step < sampling.min_tokens:
        scores[..., end] = -torch.inf
    scores = window_penalty(scores, history[-sampling.penalty_window:], sampling.repetition_penalty)
    if sampling.temperature == 0:
        return scores
    if sampling.temperature != 1:
        scores = scores / sampling.temperature
    threshold = scores.topk(min(sampling.top_k, scores.shape[-1])).values[..., -1, None]
    scores = scores.masked_fill(scores < threshold, -torch.inf)
    if sampling.top_p < 1:
        values, indices = scores.sort(descending=True)
        probabilities = values.softmax(-1)
        removed = probabilities.cumsum(-1) - probabilities > sampling.top_p
        removed[..., :3 if legacy_off else 1] = False
        values = values.masked_fill(removed, -torch.inf)
        scores = values.scatter(-1, indices, values)
    return scores


@torch.inference_mode()
def generate_tokens(model, prefix, sampling, seed, phase, negative=None, cfg_scale=1.0,
                    legacy_off=False, cancelled=None, on_token=None, use_cuda_graph=True):
    from .modeling_yue2 import StaticKVCache
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    if negative is not None and len(negative) + sampling.max_tokens > CONTEXT:
        raise ValueError("Negative prefix + generation budget exceeds context")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")
    # The two stages deliberately reset their request-local seed, matching the preset.
    rng_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    generator = torch.Generator(device=rng_device).manual_seed(seed)
    config = model.config

    def prefill(ids):
        cache = StaticKVCache(num_layers=config.num_hidden_layers, batch_size=1,
                              num_kv_heads=config.num_key_value_heads,
                              max_seq_len=len(ids) + sampling.max_tokens,
                              head_dim=config.head_dim, dtype=dtype, device=device)
        output = model(torch.tensor([ids], device=device), past_key_values=cache,
                       use_cache=True, logits_to_keep=1)
        return output.logits[:, -1, :], output.past_key_values

    graph = None
    positive_cache = negative_cache = None
    graph_enabled = use_cuda_graph and device.type == "cuda" and not getattr(model, "_yue2_fp8_originals", {})
    synchronize(device)
    start = time.perf_counter()
    try:
        if graph_enabled:
            from .cuda_graph import GraphAR
            graph = GraphAR(model, [prefix] if cfg_scale == 1 else [prefix, negative], sampling.max_tokens)
            logits = graph.prefill()
            conditional = logits[:1]
            unconditional = logits[1:] if cfg_scale != 1 else None
        else:
            conditional, positive_cache = prefill(prefix)
            unconditional = None
            if cfg_scale != 1.0:
                unconditional, negative_cache = prefill(negative)
        synchronize(device)
        prefill_seconds = time.perf_counter() - start
        history, first, eos = [], None, False
        end = ABC_END if phase == "abc" else MUSIC_END
        for step in range(sampling.max_tokens):
            if cancelled is not None and cancelled():
                raise InterruptedError(f"Cancelled during {phase}")
            # Preserve historical BF16 CFG subtraction/multiply/add before upcast.
            logits = conditional if cfg_scale == 1.0 else unconditional + cfg_scale * (conditional - unconditional)
            scores = distribution(logits, sampling, history, step, phase, legacy_off)
            if sampling.temperature == 0:
                next_id = scores.argmax(-1, keepdim=True)
            else:
                probabilities = scores.softmax(-1)
                if device.type == "mps":
                    next_id = torch.multinomial(probabilities.cpu(), 1, generator=generator).to(device)
                else:
                    next_id = torch.multinomial(probabilities, 1, generator=generator)
            token = int(next_id.item())
            if first is None:
                first = time.perf_counter() - start
            if on_token is not None:
                on_token(phase, token)
            if token == end:
                eos = True
                break
            history.append(token)
            if step + 1 < sampling.max_tokens:
                if graph is not None:
                    branch_logits = graph.step(next_id)
                    conditional = branch_logits[:1]
                    unconditional = branch_logits[1:] if cfg_scale != 1 else None
                else:
                    conditional = model(next_id, past_key_values=positive_cache, use_cache=True,
                                        logits_to_keep=1).logits[:, -1, :]
                    if negative_cache is not None:
                        unconditional = model(next_id, past_key_values=negative_cache, use_cache=True,
                                              logits_to_keep=1).logits[:, -1, :]
        synchronize(device)
        seconds = time.perf_counter() - start
        count = len(history) + int(eos)
        timing = {"seconds": seconds, "prefill_seconds": prefill_seconds,
                  "ttft_seconds": first, "output_tokens": count, "content_tokens": len(history),
                  "output_tps": count / seconds, "prefix_tokens": len(prefix),
                  "cfg_branches": 1 if cfg_scale == 1 else 2,
                  "execution": "cuda_graph" if graph is not None else "eager",
                  "attention": graph.attention_backend if graph is not None else "sdpa"}
        if use_cuda_graph and not graph_enabled:
            timing["graph_fallback_reason"] = "fp8_not_graph_validated" if getattr(model, "_yue2_fp8_originals", {}) else "non_cuda_device"
        return history, timing, not eos
    finally:
        if graph is not None:
            graph.close()
        positive_cache = negative_cache = None
