# Copyright (c) Ant Group. All rights reserved.
"""Token-level input contract for Ming-3.0-tts.

The released checkpoint learned one exact token layout. Rebuilding it by
hand in every caller is how silent quality regressions happen, so the
sequence is assembled here and the tokenizer is validated against it at load
time rather than producing plausible-sounding but wrong audio.
"""

from __future__ import annotations

from typing import List, Tuple


# --- Dense (Qwen2-based 0.5b) -------------------------------------------
DENSE_SYSTEM_PREFIX = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
DENSE_USER_PREFIX = "<|im_start|>user\n"
DENSE_USER_SUFFIX = "<|im_end|>\n"
DENSE_ASSISTANT_PREFIX = "<|im_start|>assistant\n"
DENSE_SPEAKER_START = "<|vision_start|>"
DENSE_SPEAKER_PAD = "<|vision_pad|>"
DENSE_SPEAKER_END = "<|vision_end|>\n"
# The checkpoint was trained with the assistant-side caption terminated by
# <|im_end|>, not <|endoftext|>.
DENSE_INSTRUCTION_EOS = "<|im_end|>"

# Token IDs the released 0.5b checkpoint was trained with. Names alone are not
# enough: a stock Qwen2 tokenizer with the same extra tokens added in a
# different order gives every name the right spelling and the wrong embedding
# row, which degrades audio without raising anything.
DENSE_TOKEN_IDS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<|vision_start|>": 151652,
    "<|vision_end|>": 151653,
    "<|vision_pad|>": 151654,
    "<audioPatch>": 151705,
    "<audio>": 151706,
}

# --- Shared --------------------------------------------------------------
TEXT_INPUT_PREFIX = " Text input:\n"
AUDIO_START = "<audio>"
AUDIO_PATCH = "<audioPatch>"

# A BGM caption carries its own field list and must not get the text prefix.
_BGM_MARKERS = ("Genre: ", "Mood: ", "Instrument: ", "Theme: ", "Duration: ")


def _dialect(model_type):
    """Return the token vocabulary for this checkpoint family.

    Only the dense layout is implemented. Falling back to the legacy MoE
    constants for an unrecognised checkpoint would produce a sequence that
    tokenizes without error and conditions the model on something it never
    saw, so an unsupported ``model_type`` is refused instead.
    """
    if str(model_type) == "dense":
        return {
            "speaker_start": DENSE_SPEAKER_START,
            "speaker_pad": DENSE_SPEAKER_PAD,
            "speaker_end": DENSE_SPEAKER_END,
            "instruction_eos": DENSE_INSTRUCTION_EOS,
        }
    raise NotImplementedError(
        f"model_type={model_type!r} is not part of Ming-3.0-tts-0.5b. "
        "Ming-3.0-tts-tiny is a different architecture on its own tokenizer "
        "and prompt layout; use ming_tiny.MingTinyTTS for it."
    )


def required_tokens(model_type) -> Tuple[str, ...]:
    tokens = _dialect(model_type)
    return (
        AUDIO_START,
        AUDIO_PATCH,
        tokens["speaker_start"],
        tokens["speaker_pad"],
        tokens["instruction_eos"],
    )


def expected_token_ids(model_type):
    """Return the trained token IDs, or ``{}`` when they are not pinned."""
    return dict(DENSE_TOKEN_IDS) if str(model_type) == "dense" else {}


def validate_tokenizer(tokenizer, model_type) -> None:
    """Fail loudly when the tokenizer cannot express the trained layout.

    A tokenizer that silently splits ``<audio>`` into sub-words still
    produces audio, just conditioned on a sequence the model never saw.
    """
    broken = {}
    for token in required_tokens(model_type):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
            broken[token] = token_id
            continue
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            broken[token] = encoded
    if broken:
        details = ", ".join(f"{name!r} -> {got!r}" for name, got in broken.items())
        raise ValueError(
            f"Tokenizer does not match the Ming-3.0-tts {model_type} contract; "
            f"these tokens are missing or split: {details}. Load the "
            "tokenizer shipped with the checkpoint."
        )

    mismatched = {
        token: (want, tokenizer.convert_tokens_to_ids(token))
        for token, want in expected_token_ids(model_type).items()
        if tokenizer.convert_tokens_to_ids(token) != want
    }
    if mismatched:
        details = ", ".join(
            f"{token!r}: expected {want}, got {got}"
            for token, (want, got) in sorted(mismatched.items())
        )
        raise ValueError(
            f"Tokenizer IDs do not match Ming-3.0-tts {model_type} training: "
            f"{details}. These IDs index embedding rows, so a name-only match "
            "is not enough. Load the tokenizer shipped with the checkpoint."
        )


def build_input_ids(
    tokenizer,
    model_type,
    prompt: str,
    text: str,
    *,
    num_speakers: int = 0,
    instruction: str = None,
    prompt_text: str = None,
    prompt_latent_steps: int = 0,
) -> Tuple[List[int], List[int], int]:
    """Assemble the input sequence and the rows to overwrite with features.

    Returns the token ids, the index of each ``speaker_pad`` row, and the
    index of the first prompt-latent row (``-1`` when there is no acoustic
    prompt).
    """
    tokens = _dialect(model_type)
    encode = lambda value: tokenizer.encode(value, add_special_tokens=False)

    speaker_block: List[int] = []
    for index in range(num_speakers):
        speaker_block.extend(
            encode(f"  speaker_{index + 1}:")
            + encode(tokens["speaker_start"])
            + encode(tokens["speaker_pad"])
            + encode(tokens["speaker_end"])
        )

    instruction_block: List[int] = []
    if instruction is not None:
        instruction_block = encode(instruction) + encode(tokens["instruction_eos"])

    prompt_text_block: List[int] = []
    latent_block: List[int] = []
    if prompt_latent_steps > 0 and prompt_text is not None:
        prompt_text_block = encode(prompt_text)
        latent_block = encode(AUDIO_PATCH) * prompt_latent_steps

    text_prefix = [] if all(m in text for m in _BGM_MARKERS) else encode(TEXT_INPUT_PREFIX)

    head = (
        encode(DENSE_SYSTEM_PREFIX)
        + encode(DENSE_USER_PREFIX)
        + encode(prompt)
    )
    turn_end = encode(DENSE_USER_SUFFIX) + encode(DENSE_ASSISTANT_PREFIX)

    input_ids = (
        head
        + speaker_block
        + text_prefix
        + prompt_text_block
        + encode(text)
        + turn_end
        + instruction_block
        + encode(AUDIO_START)
    )

    speaker_pad_id = tokenizer.convert_tokens_to_ids(tokens["speaker_pad"])
    speaker_start_id = tokenizer.convert_tokens_to_ids(tokens["speaker_start"])
    speaker_positions = [
        position + 1
        for position, token_id in enumerate(input_ids)
        if token_id == speaker_start_id
    ]
    if len(speaker_positions) != num_speakers:
        raise RuntimeError(
            f"expected {num_speakers} speaker slots, found "
            f"{len(speaker_positions)}"
        )
    for position in speaker_positions:
        if input_ids[position] != speaker_pad_id:
            raise RuntimeError("speaker slot is not followed by its pad token")

    latent_position = len(input_ids) if latent_block else -1
    input_ids.extend(latent_block)
    return input_ids, speaker_positions, latent_position


__all__ = [
    "AUDIO_PATCH",
    "AUDIO_START",
    "DENSE_INSTRUCTION_EOS",
    "DENSE_TOKEN_IDS",
    "TEXT_INPUT_PREFIX",
    "build_input_ids",
    "expected_token_ids",
    "required_tokens",
    "validate_tokenizer",
]
