import math
import random

import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from ...modules.speaker.campplus import CAMPPlus
from ...modules.speaker.fbank import (
    _SPEAKER_FBANK_N_MELS,
    _SPEAKER_FBANK_SAMPLE_RATE,
    extract_speaker_fbank,
)


class SpeakerXVectorFeatures(nn.Module):
    """
    Speaker embedding extractor based on 3D-Speaker CAM++.
    """

    def __init__(
        self,
        sample_rate=_SPEAKER_FBANK_SAMPLE_RATE,
        campplus_embedding_size=512,
        max_audio_seconds=10.0,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.max_audio_seconds = float(max_audio_seconds)
        self.model = CAMPPlus(
            feat_dim=_SPEAKER_FBANK_N_MELS,
            embedding_size=campplus_embedding_size,
        )
        self.resample = None
        if self.sample_rate != _SPEAKER_FBANK_SAMPLE_RATE:
            self.resample = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=_SPEAKER_FBANK_SAMPLE_RATE,
            )

        for param in self.model.parameters():
            param.requires_grad = False

    @staticmethod
    def _normalize_lengths(lengths, batch_size, max_length, device, *, min_length):
        if lengths is None:
            return torch.full(
                (batch_size,),
                max_length,
                device=device,
                dtype=torch.long,
            )
        return lengths.to(device=device, dtype=torch.long).clamp(
            min=min_length,
            max=max_length,
        )

    def _crop_audio(self, audio, audio_lengths=None):
        original_lengths = self._normalize_lengths(
            audio_lengths,
            audio.size(0),
            audio.size(-1),
            audio.device,
            min_length=0,
        )
        if self.max_audio_seconds <= 0:
            return audio, original_lengths, original_lengths, torch.zeros_like(
                original_lengths
            )

        max_input_length = round(self.sample_rate * self.max_audio_seconds)
        cropped_audio = []
        cropped_lengths = []
        starts = []

        for index, total_length_tensor in enumerate(original_lengths):
            total_length = int(total_length_tensor.item())
            cropped_length = min(total_length, max_input_length)
            start = (
                random.randint(0, total_length - cropped_length)
                if total_length > cropped_length
                else 0
            )
            cropped_audio.append(audio[index, start : start + cropped_length])
            cropped_lengths.append(cropped_length)
            starts.append(start)

        return pad_sequence(
            cropped_audio,
            batch_first=True,
            padding_value=0.0,
        ), original_lengths, torch.tensor(
            cropped_lengths,
            device=audio.device,
            dtype=torch.long,
        ), torch.tensor(starts, device=audio.device, dtype=torch.long)

    def _crop_fbank(
        self,
        fbank,
        fbank_lengths,
        original_audio_lengths,
        cropped_audio_lengths,
        starts,
    ):
        original_fbank_lengths = self._normalize_lengths(
            fbank_lengths,
            fbank.size(0),
            fbank.size(1),
            fbank.device,
            min_length=1,
        )
        cropped_fbank = []
        cropped_fbank_lengths = []

        for index, total_feat_length_tensor in enumerate(original_fbank_lengths):
            total_audio_length = int(original_audio_lengths[index].item())
            total_feat_length = int(total_feat_length_tensor.item())
            start_audio = int(starts[index].item())
            end_audio = start_audio + int(cropped_audio_lengths[index].item())

            if total_audio_length > 0:
                start_feat = math.floor(
                    start_audio * total_feat_length / total_audio_length
                )
                end_feat = math.ceil(end_audio * total_feat_length / total_audio_length)
            else:
                start_feat = 0
                end_feat = 1

            start_feat = min(start_feat, total_feat_length - 1)
            end_feat = min(max(end_feat, start_feat + 1), total_feat_length)
            cropped_fbank.append(fbank[index, start_feat:end_feat])
            cropped_fbank_lengths.append(end_feat - start_feat)

        return pad_sequence(
            cropped_fbank,
            batch_first=True,
            padding_value=0.0,
        ), torch.tensor(
            cropped_fbank_lengths,
            device=fbank.device,
            dtype=torch.long,
        )

    def _extract_fbank_batch(self, audio, audio_lengths):
        if self.resample is not None:
            audio = self.resample(audio)
            audio_lengths = torch.ceil(
                audio_lengths.float()
                * (_SPEAKER_FBANK_SAMPLE_RATE / self.sample_rate)
            ).long()

        audio_cpu = audio.detach().cpu()
        features = []

        for index, valid_length_tensor in enumerate(audio_lengths):
            valid_length = int(valid_length_tensor.item())
            waveform = audio_cpu[index, :valid_length]
            if waveform.numel() == 0:
                waveform = audio_cpu.new_zeros(1)
            features.append(
                extract_speaker_fbank(
                    waveform,
                    sample_rate=_SPEAKER_FBANK_SAMPLE_RATE,
                )
            )

        fbank_lengths = torch.tensor(
            [feature.size(0) for feature in features],
            device=audio.device,
            dtype=torch.long,
        )
        fbank = pad_sequence(
            features,
            batch_first=True,
            padding_value=0.0,
        ).to(device=audio.device, dtype=audio.dtype)
        return fbank, fbank_lengths

    @torch.no_grad()
    @torch.autocast(enabled=False, device_type="cuda")
    def forward(
        self, audio, audio_lengths=None, fbank=None, fbank_lengths=None, **_kwargs
    ):
        self.model.eval()
        audio = audio.float()
        if audio.dim() == 3:
            if audio.size(1) != 1:
                raise ValueError(
                    f"Speaker encoder expects mono audio, got shape {tuple(audio.shape)}."
                )
            audio = audio[:, 0]
        elif audio.dim() != 2:
            raise ValueError(
                f"Speaker encoder expects a 2D or 3D audio tensor, got shape {tuple(audio.shape)}."
            )

        audio, original_audio_lengths, cropped_audio_lengths, starts = self._crop_audio(
            audio,
            audio_lengths=audio_lengths,
        )

        if fbank is None:
            fbank, fbank_lengths = self._extract_fbank_batch(
                audio,
                cropped_audio_lengths,
            )
        else:
            if not isinstance(fbank, torch.Tensor):
                raise TypeError("Speaker encoder expects `fbank` to be a torch.Tensor.")
            if fbank.dim() != 3 or fbank.size(0) != audio.size(0):
                raise ValueError(
                    f"Speaker encoder expects `fbank` with shape (B, T, F) and matching batch size, got {tuple(fbank.shape)}."
                )
            fbank, fbank_lengths = self._crop_fbank(
                fbank.to(device=audio.device, dtype=torch.float32),
                fbank_lengths,
                original_audio_lengths,
                cropped_audio_lengths,
                starts,
            )

        return self.model(fbank, lengths=fbank_lengths)
