"""Audio helpers used by the retained train/infer pipeline."""

from __future__ import annotations

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as Kaldi
import torchaudio.functional as AF


EDIT_EDGE_SILENCE_MS = 250.0
EDIT_EDGE_SILENCE_TOP_DB = 30.0


def high_quality_resample(x, orig_sr, target_sr):
    return AF.resample(
        x,
        orig_freq=orig_sr,
        new_freq=target_sr,
        lowpass_filter_width=64,
        rolloff=0.95,
        resampling_method="sinc_interp_kaiser",
    )


def prepare_edit_source_audio(
    audio_path: str,
    *,
    target_sample_rate: int,
    samples_per_llm_token: int,
) -> torch.Tensor:
    """Load and align one Edit source waveform for schedule prefill.

    Returns mono float32 audio with shape ``[1, samples]``. The waveform is
    resampled, normalized to 250 ms of silence at each edge, and padded to an
    exact LLM audio-token boundary. Edit retains its established 128-tap
    resampling contract without changing the public TTS prompt resampler.
    """

    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive.")
    if samples_per_llm_token <= 0:
        raise ValueError("samples_per_llm_token must be positive.")

    audio_data, sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(audio_data.T)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.contiguous()
    if waveform.numel() == 0:
        raise ValueError("Edit source audio must not be empty.")

    waveform = AF.resample(
        waveform,
        orig_freq=int(sample_rate),
        new_freq=int(target_sample_rate),
        lowpass_filter_width=128,
        rolloff=0.95,
        resampling_method="sinc_interp_kaiser",
    )
    waveform = _normalize_edit_edge_silence(
        waveform,
        sample_rate=int(target_sample_rate),
    )
    return _pad_edit_waveform_to_token_boundary(
        waveform,
        multiple_of=int(samples_per_llm_token),
    )


def _normalize_edit_edge_silence(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
) -> torch.Tensor:
    mono_waveform = waveform[0]
    target_samples = int(
        round(float(sample_rate) * EDIT_EDGE_SILENCE_MS / 1000.0)
    )
    amplitude = mono_waveform.abs()
    peak = float(amplitude.max().item())
    if peak <= 0.0:
        waveform = waveform[..., :target_samples]
        current_length = int(waveform.size(-1))
        if current_length < target_samples:
            waveform = F.pad(
                waveform,
                (0, target_samples - current_length),
                "constant",
                0.0,
            )
        return waveform

    threshold = peak * (10.0 ** (-EDIT_EDGE_SILENCE_TOP_DB / 20.0))
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


def _pad_edit_waveform_to_token_boundary(
    waveform: torch.Tensor,
    *,
    multiple_of: int,
) -> torch.Tensor:
    remainder = int(waveform.size(-1)) % multiple_of
    if remainder == 0:
        return waveform
    return F.pad(
        waveform,
        (0, multiple_of - remainder),
        "constant",
        0.0,
    )


def extract_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_mels: int,
    dither: float = 0.0,
    mean_norm: bool = False,
) -> torch.Tensor:
    if waveform.ndim == 1:
        feature_input = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        feature_input = waveform if waveform.size(0) == 1 else waveform[0:1, :]
    else:
        raise ValueError(
            f"FBank expects a 1D or 2D waveform, got shape {tuple(waveform.shape)}."
        )
    features = Kaldi.fbank(
        feature_input,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
        dither=dither,
    )
    if mean_norm:
        features = features - features.mean(dim=0, keepdim=True)
    return features
