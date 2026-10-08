from __future__ import annotations

import logging
from collections.abc import Iterable

import torch

logger = logging.getLogger(__name__)

IRODORI_WATERMARK_PAYLOAD = (73, 82, 68, 84, 83)  # "IRDTS"


def _as_single_channel_vector(audio: torch.Tensor) -> torch.Tensor | None:
    squeezed = audio.detach().float().squeeze()
    if squeezed.ndim == 0 or squeezed.numel() == 0:
        return None
    if squeezed.ndim == 1:
        return squeezed
    return squeezed.reshape(-1)


def _match_original_rank(audio: torch.Tensor, *, reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim == 2:
        return audio.reshape(1, -1)
    return audio.reshape(-1)


class SilentCipherWatermarker:
    def __init__(self, *, device: str, model_type: str = "44.1k") -> None:
        self.model = self._load_backend(device=device, model_type=model_type)

    @staticmethod
    def _load_backend(*, device: str, model_type: str):
        try:
            import silentcipher
        except ImportError:
            logger.warning(
                "SilentCipher package is unavailable; generated audio will not be watermarked."
            )
            return None

        try:
            return silentcipher.get_model(model_type=model_type, device=device)
        except Exception as exc:
            logger.warning(
                "SilentCipher model could not be loaded (%s); generated audio will not be "
                "watermarked.",
                exc,
            )
            return None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def encode_one(
        self,
        audio: torch.Tensor,
        *,
        sample_rate: int,
        payload: Iterable[int] = IRODORI_WATERMARK_PAYLOAD,
    ) -> torch.Tensor:
        if self.model is None:
            return audio

        vector = _as_single_channel_vector(audio)
        if vector is None:
            return audio

        encoded, _ = self.model.encode_wav(
            vector.to(self.model.device),
            int(sample_rate),
            list(payload),
            calc_sdr=False,
        )
        encoded_audio = torch.as_tensor(encoded, dtype=torch.float32, device="cpu")
        return _match_original_rank(encoded_audio, reference=audio)

    def encode_batch(self, audios: list[torch.Tensor], *, sample_rate: int) -> list[torch.Tensor]:
        if self.model is None:
            return audios
        return [self.encode_one(audio, sample_rate=sample_rate) for audio in audios]
