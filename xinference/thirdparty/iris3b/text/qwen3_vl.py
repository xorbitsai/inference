# Modified for Xinference: use package-relative imports.
"""Qwen3-VL language-stack text encoder."""

import logging
from pathlib import Path

import torch

from ..config import TextEncoderConfig
from ..registry import TEXT_ENCODERS
from .base import TextEncoder, TextEncoding

# Text-conditioning chat template. Only the language stack is used; no image
# placeholders or pixel values enter this encoder.
_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial "
    "relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
_PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

logger = logging.getLogger(__name__)
# rate limit for the overflow warning, counted in encode calls
_OVERFLOW_LOG_INTERVAL = 100


@TEXT_ENCODERS.register("qwen3_vl")
class Qwen3VLTextEncoder(TextEncoder):
    """Frozen Qwen3-VL language stack producing fixed-length prompt states."""

    def __init__(self, cfg: TextEncoderConfig, device: str | torch.device = "cpu"):
        try:
            from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3VLTextEncoder requires transformers>=4.57.0 with native Qwen3-VL support"
            ) from exc

        self.cfg = cfg
        self.dim = cfg.dim
        self.max_length = cfg.max_length
        self.hidden_layers = tuple(cfg.hidden_layers)
        self.device = torch.device(device)

        if cfg.on_caption_overflow not in ("warn", "error", "silent"):
            raise ValueError(
                "text_encoder.on_caption_overflow must be warn, error or silent, got "
                f"{cfg.on_caption_overflow!r}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained)
        # The padded batch is assembled here instead of by the tokenizer: the
        # template suffix is appended after the caption is truncated, so the
        # assistant-turn marker can never be the part that gets cut.
        prefix_ids = self.tokenizer.encode(_PROMPT_PREFIX, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(_PROMPT_SUFFIX, add_special_tokens=False)
        if not prefix_ids or not suffix_ids:
            raise ValueError("Qwen3-VL prompt template produced an empty token prefix or suffix")
        self._prefix_ids = torch.tensor(prefix_ids, dtype=torch.long)
        self._prefix_len = len(prefix_ids)
        self._suffix_ids = torch.tensor(suffix_ids, dtype=torch.long)
        self._suffix_len = len(suffix_ids)
        self.caption_budget = self.max_length - self._suffix_len
        if self.caption_budget < 1:
            raise ValueError(
                f"text_encoder.max_length={self.max_length} leaves no room for a caption: the "
                f"assistant-turn suffix alone is {self._suffix_len} tokens"
            )
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Qwen3-VL tokenizer exposes neither pad_token_id nor eos_token_id")
        self._pad_id = int(pad_id)

        # caption-overflow counters; the trainer reads these for logging
        self.calls = 0
        self.overflow_rows = 0
        self.overflow_tokens = 0
        self.max_caption_tokens = 0
        self._last_overflow_log = -_OVERFLOW_LOG_INTERVAL

        dtype = getattr(torch, cfg.dtype)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.pretrained,
            dtype=dtype,
            attn_implementation=cfg.attn_implementation,
        )
        decoder = model.get_decoder()
        del model

        hidden_size = getattr(decoder.config, "hidden_size", None)
        if hidden_size != self.dim:
            raise ValueError(
                f"Qwen3-VL checkpoint hidden size is {hidden_size}, but text_encoder.dim={self.dim}"
            )
        num_hidden_layers = getattr(decoder.config, "num_hidden_layers", None)
        if self.hidden_layers:
            if tuple(sorted(set(self.hidden_layers))) != self.hidden_layers:
                raise ValueError("text_encoder.hidden_layers must be sorted and unique")
            if num_hidden_layers is None:
                raise ValueError("Qwen3-VL checkpoint does not report num_hidden_layers")
            if self.hidden_layers[0] < 1 or self.hidden_layers[-1] > num_hidden_layers:
                raise ValueError(
                    "text_encoder.hidden_layers must use 1-based indices in "
                    f"[1, {num_hidden_layers}], got {list(self.hidden_layers)}"
                )

        self.decoder = decoder.to(self.device).eval()
        self.decoder.requires_grad_(False)
        if cfg.compile:
            self.decoder.compile()

    def _tokenize_captions(self, prompts: list[str]) -> list[list[int]]:
        """Tokenize captions alone and account for anything past the budget."""
        caption_ids = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        lengths = [len(ids) for ids in caption_ids]
        longest = max(lengths, default=0)
        self.calls += 1
        self.max_caption_tokens = max(self.max_caption_tokens, longest)
        dropped = [n - self.caption_budget for n in lengths if n > self.caption_budget]
        if not dropped:
            return caption_ids
        self.overflow_rows += len(dropped)
        self.overflow_tokens += sum(dropped)
        if self.cfg.on_caption_overflow == "error":
            raise ValueError(
                f"caption tokenizes to {longest} tokens but only {self.caption_budget} fit "
                f"(text_encoder.max_length={self.max_length} minus {self._suffix_len} chat-template "
                "tokens); raise text_encoder.max_length, shorten the caption, or set "
                "text_encoder.on_caption_overflow=warn to truncate it"
            )
        if (
            self.cfg.on_caption_overflow == "warn"
            and self.calls - self._last_overflow_log >= _OVERFLOW_LOG_INTERVAL
        ):
            self._last_overflow_log = self.calls
            logger.warning(
                "caption overflow: %d of %d rows truncated (longest %d tokens, budget %d); "
                "cumulative %d rows / %d dropped tokens over %d calls, longest caption seen %d",
                len(dropped),
                len(lengths),
                longest,
                self.caption_budget,
                self.overflow_rows,
                self.overflow_tokens,
                self.calls,
                self.max_caption_tokens,
            )
        return caption_ids

    def _run(self, prompts: list[str]) -> TextEncoding:
        caption_ids = self._tokenize_captions(prompts)
        start = self._prefix_len
        stop = start + self.max_length
        input_ids = torch.full((len(prompts), stop), self._pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), stop), dtype=torch.long)
        input_ids[:, :start] = self._prefix_ids
        for row, ids in enumerate(caption_ids):
            n = min(len(ids), self.caption_budget)
            if n:
                input_ids[row, start : start + n] = torch.tensor(ids[:n], dtype=torch.long)
            input_ids[row, start + n : start + n + self._suffix_len] = self._suffix_ids
            attention_mask[row, : start + n + self._suffix_len] = 1

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        output = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=bool(self.hidden_layers),
        )
        mask = attention_mask[:, start:stop].to(torch.int64)
        if self.hidden_layers:
            if output.hidden_states is None or len(output.hidden_states) <= self.hidden_layers[-1]:
                raise RuntimeError("Qwen3-VL decoder did not return the requested hidden layers")
            embeddings = torch.stack(
                [output.hidden_states[layer][:, start:stop] for layer in self.hidden_layers],
                dim=2,
            )
            embeddings = embeddings * mask[:, :, None, None].to(embeddings.dtype)
        else:
            embeddings = output.last_hidden_state[:, start:stop]
            embeddings = embeddings * mask.unsqueeze(-1).to(embeddings.dtype)
        return TextEncoding(embeddings=embeddings, mask=mask)

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> TextEncoding:
        return self._run(prompts)

    @torch.no_grad()
    def null(self, negative_prompt: str = "") -> TextEncoding:
        """Encode CFG null/negative text with the same template as positive prompts."""
        cache = self._null_cache_path() if negative_prompt == "" else None
        if cache is not None and cache.is_file():
            payload = torch.load(cache, map_location="cpu")
            return TextEncoding(
                embeddings=payload["embeddings"].to(self.device),
                mask=payload["mask"].to(self.device),
            )

        encoding = self._run([negative_prompt])
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"embeddings": encoding.embeddings.cpu(), "mask": encoding.mask.cpu()},
                cache,
            )
        return encoding

    def _null_cache_path(self) -> Path:
        name = self.cfg.pretrained.replace("/", "-")
        layers = "" if not self.hidden_layers else "_layers-" + "-".join(map(str, self.hidden_layers))
        return Path(self.cfg.null_embed_dir) / (
            f"null_embed_qwen3vl_{name}_{self.max_length}token_{self.dim}{layers}.pth"
        )

    def to(self, device) -> "Qwen3VLTextEncoder":
        self.device = torch.device(device)
        self.decoder = self.decoder.to(self.device)
        return self
