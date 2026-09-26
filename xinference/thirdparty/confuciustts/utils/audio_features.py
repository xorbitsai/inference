from functools import lru_cache
from typing import Optional

import torch
import torchaudio
from librosa.filters import mel as librosa_mel_fn


_FBANK_TRANSFORMS: dict = {}


@lru_cache(maxsize=16)
def _get_mel_transform(
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    f_min: float,
    f_max: Optional[float],
    device: str,
) -> torchaudio.transforms.MelSpectrogram:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=1.0,
        center=True,
        pad_mode="reflect",
        mel_scale="slaney",
        norm="slaney",
    ).to(torch.device(device))


@lru_cache(maxsize=16)
def _get_mel_spectrogram_basis(
    n_fft: int,
    n_mels: int,
    sample_rate: int,
    win_length: int,
    fmin: float,
    fmax: Optional[float],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    mel = librosa_mel_fn(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    mel_basis = torch.from_numpy(mel).float().to(torch.device(device))
    hann_window = torch.hann_window(win_length).to(torch.device(device))
    return mel_basis, hann_window


def extract_mel(
    waveform: torch.Tensor,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    log_eps: float = 1e-5,
) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    tx = _get_mel_transform(
        sample_rate,
        n_fft,
        win_length,
        hop_length,
        n_mels,
        f_min,
        f_max,
        str(waveform.device),
    )
    mel = tx(waveform)
    mel = torch.log(mel.clamp_min(log_eps))
    return mel.squeeze(0).transpose(0, 1).contiguous()


def mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
) -> torch.Tensor:
    device = audio.device
    mel_basis, hann_window = _get_mel_spectrogram_basis(
        n_fft, n_mels, sample_rate, win_length, fmin, fmax, str(device)
    )

    pad = (n_fft - hop_length) // 2
    y = torch.nn.functional.pad(audio.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.view_as_real(torch.stft(
        y, n_fft, hop_length=hop_length, win_length=win_length,
        window=hann_window, center=False, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True,
    ))
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(mel_basis, spec)
    return torch.log(torch.clamp(spec, min=1e-5))


def extract_fbank(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_mels: int = 80,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
) -> torch.Tensor:
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    feat = torchaudio.compliance.kaldi.fbank(
        waveform,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=0.0,
    )
    feat = feat - feat.mean(dim=0, keepdim=True)
    return feat
