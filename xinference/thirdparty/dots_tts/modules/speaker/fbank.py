from __future__ import annotations

import torch

from ...utils.audio import extract_fbank, high_quality_resample

_SPEAKER_FBANK_SAMPLE_RATE = 16000
_SPEAKER_FBANK_N_MELS = 80
_SPEAKER_FBANK_MEAN_NORM = True
_SPEAKER_FBANK_DITHER = 0.0


def extract_speaker_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
) -> torch.Tensor:
    feature_input = waveform
    if sample_rate != _SPEAKER_FBANK_SAMPLE_RATE:
        feature_input = high_quality_resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=_SPEAKER_FBANK_SAMPLE_RATE,
        )
    return extract_fbank(
        feature_input,
        sample_rate=_SPEAKER_FBANK_SAMPLE_RATE,
        n_mels=_SPEAKER_FBANK_N_MELS,
        dither=_SPEAKER_FBANK_DITHER,
        mean_norm=_SPEAKER_FBANK_MEAN_NORM,
    )
