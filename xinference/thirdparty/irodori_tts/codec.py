from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from huggingface_hub import hf_hub_download

_CODEC_DEFAULT = object()


def patchify_latent(latent: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert latent from (B, T, D) -> (B, T//patch, D*patch).
    Extra tail tokens are dropped.
    """
    if patch_size <= 1:
        return latent
    bsz, seq_len, dim = latent.shape
    usable = (seq_len // patch_size) * patch_size
    latent = latent[:, :usable]
    latent = latent.reshape(bsz, usable // patch_size, dim * patch_size)
    return latent


def unpatchify_latent(patched: torch.Tensor, patch_size: int, latent_dim: int) -> torch.Tensor:
    """
    Convert latent from (B, T_p, D*patch) -> (B, T_p*patch, D).
    """
    if patch_size <= 1:
        return patched
    return patched.reshape(patched.shape[0], patched.shape[1] * patch_size, latent_dim)


@dataclass
class DACVAECodec:
    model: torch.nn.Module
    sample_rate: int
    latent_dim: int
    device: torch.device
    dtype: torch.dtype
    deterministic_encode: bool
    deterministic_decode: bool
    normalize_db: float | None

    @classmethod
    def load(
        cls,
        repo_id: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        deterministic_encode: bool = True,
        deterministic_decode: bool = True,
        normalize_db: float | None = -16.0,
    ) -> DACVAECodec:
        # Prefer installed package; fallback to local clone at ../dacvae.
        try:
            from dacvae import DACVAE
        except ImportError:
            local_repo = Path(__file__).resolve().parents[2] / "dacvae"
            if local_repo.exists():
                sys.path.insert(0, str(local_repo))
            from dacvae import DACVAE

        location = str(repo_id).strip()
        if location.startswith("hf://"):
            location = location[len("hf://") :]
        if not Path(location).exists() and "/" in location and not location.endswith(".pth"):
            try:
                location = hf_hub_download(repo_id=location, filename="weights.pth")
                print(f"[codec] dacvae: hf://{repo_id} -> {location}", flush=True)
            except Exception:
                # Let DACVAE.load surface a clearer error if this is not a valid path/repo.
                pass

        model = DACVAE.load(location).eval().to(device)
        if dtype is not None:
            model = model.to(dtype=dtype)

        decoder = getattr(model, "decoder", None)
        if decoder is not None and hasattr(decoder, "alpha"):
            decoder.alpha = 0.0
            if hasattr(decoder, "wm_model"):
                # Irodori checkpoints were trained without the DACVAE watermark branch.
                # Keep decode output mono while skipping that encode/decode path.
                def _watermark_passthrough(
                    x: torch.Tensor,
                    message: torch.Tensor | None = None,
                    _decoder=decoder,
                ) -> torch.Tensor:
                    del message
                    return _decoder.wm_model.encoder_block.forward_no_conv(x)

                decoder.watermark = _watermark_passthrough

        if deterministic_decode:
            cls._configure_deterministic_decode(model=model, device=device)

        model_dtype = next(model.parameters()).dtype
        # Infer latent dimension by encoding a tiny random signal.
        dummy = torch.zeros(1, 1, 2048, device=device, dtype=model_dtype)
        with torch.inference_mode():
            z = model.encode(dummy)  # (B, D, T)
        return cls(
            model=model,
            sample_rate=int(model.sample_rate),
            latent_dim=int(z.shape[1]),
            device=torch.device(device),
            dtype=model_dtype,
            deterministic_encode=bool(deterministic_encode),
            deterministic_decode=bool(deterministic_decode),
            normalize_db=None if normalize_db is None else float(normalize_db),
        )

    @staticmethod
    def _configure_deterministic_decode(model: torch.nn.Module, device: str | torch.device) -> None:
        decoder = getattr(model, "decoder", None)
        wm_model = getattr(decoder, "wm_model", None)
        msg_processor = getattr(wm_model, "msg_processor", None)
        if msg_processor is None:
            return
        nbits = int(msg_processor.nbits)
        message_device = torch.device(device)

        def _fixed_message(batch_size: int) -> torch.Tensor:
            return torch.zeros((batch_size, nbits), dtype=torch.float32, device=message_device)

        wm_model.random_message = _fixed_message

    @staticmethod
    def _normalize_loudness(
        wav: torch.Tensor, sample_rate: int, target_db: float | None
    ) -> torch.Tensor:
        if target_db is None:
            return wav
        wav_device = wav.device
        wav = wav.to(dtype=torch.float32)
        if wav.ndim == 2:
            if wav.shape[0] == 1:
                wav = wav[0]
            elif wav.shape[1] == 1:
                wav = wav[:, 0]
            else:
                wav = wav.mean(dim=0)
        if wav.ndim != 1:
            raise ValueError(
                "normalize_loudness expects a mono waveform with shape (T,) "
                f"or singleton-channel (1, T)/(T, 1), got {tuple(wav.shape)}"
            )

        try:
            from audiotools import AudioSignal
        except Exception as exc:
            raise RuntimeError(
                "audiotools is required when normalize_db is set. "
                "Install audiotools or disable normalize_db."
            ) from exc

        signal = AudioSignal(wav.unsqueeze(0).unsqueeze(0), int(sample_rate))
        signal.normalize(float(target_db))
        signal.ensure_max_of_audio()
        normalized = signal.audio_data
        if not isinstance(normalized, torch.Tensor):
            normalized = torch.as_tensor(normalized)
        normalized = normalized.to(dtype=torch.float32, device=wav_device)
        normalized = normalized.squeeze()
        if normalized.ndim != 1:
            raise RuntimeError(
                "audiotools normalization returned an unexpected waveform shape "
                f"{tuple(normalized.shape)}"
            )
        return normalized

    @torch.inference_mode()
    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        *,
        normalize_db: float | None | object = _CODEC_DEFAULT,
        ensure_max: bool | None = None,
    ) -> torch.Tensor:
        """
        Input:
          waveform: (B, C, T) or (C, T)
          normalize_db: Optional target loudness (LUFS-like dB) applied before encode
          ensure_max: If True and normalize_db is None, scale down only when abs peak exceeds 1.0
        Output:
          latent: (B, T_latent, D_latent)
        """
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError(f"Expected waveform ndim=3, got shape={tuple(waveform.shape)}")

        if waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        if normalize_db is _CODEC_DEFAULT:
            effective_normalize_db = self.normalize_db
        elif normalize_db is None:
            effective_normalize_db = None
        else:
            effective_normalize_db = float(normalize_db)
        # audiotools normalization already applies ensure_max_of_audio(), so codec-side
        # peak scaling is only needed when normalization is disabled.
        effective_ensure_max = (
            effective_normalize_db is None and bool(ensure_max) if ensure_max is not None else False
        )

        waveform = waveform.to(dtype=torch.float32)
        if effective_normalize_db is not None or effective_ensure_max:
            # Keep behavior deterministic per utterance by normalizing each waveform independently.
            processed: list[torch.Tensor] = []
            for wav in waveform.squeeze(1):
                if effective_normalize_db is not None:
                    wav = self._normalize_loudness(
                        wav, sample_rate=self.sample_rate, target_db=effective_normalize_db
                    )
                wav = wav.squeeze()
                if wav.ndim != 1:
                    raise RuntimeError(
                        "Expected mono per-item waveform after preprocessing, "
                        f"got shape={tuple(wav.shape)}"
                    )
                if effective_ensure_max:
                    peak = wav.abs().max()
                    if torch.isfinite(peak) and peak > 1.0:
                        wav = wav * (1.0 / float(peak))
                processed.append(wav)
            waveform = torch.stack(processed, dim=0).unsqueeze(1)

        waveform = waveform.to(self.device, dtype=self.dtype)
        if self.deterministic_encode:
            required_paths_present = (
                hasattr(self.model, "encoder")
                and hasattr(self.model, "_pad")
                and hasattr(self.model, "quantizer")
                and hasattr(self.model.quantizer, "in_proj")
            )
            if not required_paths_present:
                raise RuntimeError(
                    "deterministic_encode=True requires encoder/_pad/quantizer.in_proj on DACVAE model."
                )
            z = self.model.encoder(self.model._pad(waveform))
            mean, _scale = self.model.quantizer.in_proj(z).chunk(2, dim=1)
            encoded = mean
        else:
            encoded = self.model.encode(waveform)  # (B, D, T)
        return encoded.transpose(1, 2).contiguous()  # (B, T, D)

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Input:
          latent: (B, T, D)
        Output:
          audio: (B, 1, samples)
        """
        if latent.ndim != 3:
            raise ValueError(f"Expected latent ndim=3, got shape={tuple(latent.shape)}")
        z = latent.transpose(1, 2).contiguous().to(self.device, dtype=self.dtype)  # (B, D, T)
        return self.model.decode(z)

    def encode_file(self, path: str | Path) -> torch.Tensor:
        try:
            wav, sr = torchaudio.load(str(path))
        except RuntimeError:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            wav = torch.from_numpy(data)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.T
        wav = wav.unsqueeze(0)  # (1, C, T)
        return self.encode_waveform(wav, sr).cpu()
