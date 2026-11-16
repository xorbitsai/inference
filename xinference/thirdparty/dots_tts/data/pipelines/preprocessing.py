from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_EDGE_SILENCE_MS = 250.0
DEFAULT_EDGE_SILENCE_TOP_DB = 30.0


def align_length(num_samples: int, multiple_of: int | None) -> int:
    if multiple_of is None or multiple_of <= 0:
        return int(num_samples)
    if num_samples % multiple_of == 0:
        return int(num_samples)
    return int(((num_samples + multiple_of - 1) // multiple_of) * multiple_of)


def pad_waveform_align_only(
    waveform: torch.Tensor,
    *,
    multiple_of: int | None,
) -> torch.Tensor:
    if multiple_of is None or multiple_of <= 0:
        return waveform

    target_length = align_length(waveform.size(-1), multiple_of)
    delta = target_length - waveform.size(-1)
    if delta <= 0:
        return waveform

    return F.pad(waveform, (0, delta), "constant", 0.0)


def normalize_edge_silence_duration(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    target_silence_duration_ms: float = DEFAULT_EDGE_SILENCE_MS,
    top_db: float = DEFAULT_EDGE_SILENCE_TOP_DB,
) -> torch.Tensor:
    mono_waveform = waveform[0]
    target_samples = int(round(float(sample_rate) * float(target_silence_duration_ms) / 1000.0))
    amplitude = mono_waveform.abs()
    peak = float(amplitude.max().item())
    if peak <= 0.0:
        waveform = waveform[..., :target_samples]
        current_length = int(waveform.size(-1))
        if current_length < target_samples:
            waveform = F.pad(waveform, (0, target_samples - current_length), "constant", 0.0)
        return waveform

    threshold = peak * (10.0 ** (-float(top_db) / 20.0))
    non_silent = torch.nonzero(amplitude > threshold, as_tuple=False).flatten()
    first_non_silent = int(non_silent[0].item())
    last_non_silent = int(non_silent[-1].item())

    leading_silence_samples = first_non_silent
    trailing_silence_samples = int(mono_waveform.numel()) - last_non_silent - 1

    leading_delta = target_samples - leading_silence_samples
    if leading_delta > 0:
        waveform = F.pad(waveform, (leading_delta, 0), "constant", 0.0)
    else:
        trim_from_start = min(-leading_delta, int(waveform.size(-1)))
        waveform = waveform[..., trim_from_start:]

    trailing_delta = target_samples - trailing_silence_samples
    if trailing_delta > 0:
        return F.pad(waveform, (0, trailing_delta), "constant", 0.0)

    trim_from_end = min(-trailing_delta, int(waveform.size(-1)))
    if trim_from_end <= 0:
        return waveform
    return waveform[..., :-trim_from_end]


def compute_num_audio_tokens(
    num_samples: int, *, audio_samples_per_llm_token: int
) -> int:
    if num_samples % audio_samples_per_llm_token != 0:
        raise ValueError(
            f"Waveform length {num_samples} is not aligned to token hop {audio_samples_per_llm_token}."
        )
    return num_samples // audio_samples_per_llm_token
