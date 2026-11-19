from __future__ import annotations

AUDIO_COMP_START_TOKEN = "<|audio_comp_start|>"
AUDIO_COMP_SPAN_TOKEN = "<|audio_comp_span|>"
AUDIO_COMP_END_TOKEN = "<|audio_comp_end|>"
AUDIO_GEN_START_TOKEN = "<|audio_gen_start|>"
AUDIO_GEN_SPAN_TOKEN = "<|audio_gen_span|>"
AUDIO_GEN_END_TOKEN = "<|audio_gen_end|>"
TEXT_COND_END_TOKEN = "<|text_cond_end|>"


def require_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id < 0:
        raise ValueError(f"Artifact tokenizer is missing required special token: {token}")
    return int(token_id)


__all__ = [
    "AUDIO_COMP_END_TOKEN",
    "AUDIO_COMP_SPAN_TOKEN",
    "AUDIO_COMP_START_TOKEN",
    "AUDIO_GEN_END_TOKEN",
    "AUDIO_GEN_SPAN_TOKEN",
    "AUDIO_GEN_START_TOKEN",
    "TEXT_COND_END_TOKEN",
    "require_token_id",
]
