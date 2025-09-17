import typing

import julius
import numpy as np
import torch
import torchaudio

from . import util


class EffectMixin:
    GAIN_FACTOR = np.log(10) / 20
    """Gain factor for converting between amplitude and decibels."""
    CODEC_PRESETS = {
        "8-bit": {"format": "wav", "encoding": "ULAW", "bits_per_sample": 8},
        "GSM-FR": {"format": "gsm"},
        "MP3": {"format": "mp3", "compression": -9},
        "Vorbis": {"format": "vorbis", "compression": -1},
        "Ogg": {
            "format": "ogg",
            "compression": -1,
        },
        "Amr-nb": {"format": "amr-nb"},
    }
    """Presets for applying codecs via torchaudio."""

    def mix(
        self,
        other,
        snr: typing.Union[torch.Tensor, np.ndarray, float] = 10,
        other_eq: typing.Union[torch.Tensor, np.ndarray] = None,
    ):
        """Mixes noise with signal at specified
        signal-to-noise ratio. Optionally, the
        other signal can be equalized in-place.


        Parameters
        ----------
        other : AudioSignal
            AudioSignal object to mix with.
        snr : typing.Union[torch.Tensor, np.ndarray, float], optional
            Signal to noise ratio, by default 10
        other_eq : typing.Union[torch.Tensor, np.ndarray], optional
            EQ curve to apply to other signal, if any, by default None

        Returns
        -------
        AudioSignal
            In-place modification of AudioSignal.
        """
        snr = util.ensure_tensor(snr).to(self.device)

        pad_len = max(0, self.signal_length - other.signal_length)
        other.zero_pad(0, pad_len)
        other.truncate_samples(self.signal_length)
        if other_eq is not None:
            other = other.equalizer(other_eq)

        tgt_loudness = self.loudness() - snr
        other = other.normalize(tgt_loudness)

        self.audio_data = self.audio_data + other.audio_data
        return self

    def convolve(self, other, start_at_max: bool = True):
        """Convolves self with other.
        This function uses FFTs to do the convolution.

        Parameters
        ----------
        other : AudioSignal
            Signal to convolve with.
        start_at_max : bool, optional
            Whether to start at the max value of other signal, to
            avoid inducing delays, by default True

        Returns
        -------
        AudioSignal
            Convolved signal, in-place.
        """
        from . import AudioSignal

        pad_len = self.signal_length - other.signal_length

        if pad_len > 0:
            other.zero_pad(0, pad_len)
        else:
            other.truncate_samples(self.signal_length)

        if start_at_max:
            # Use roll to rotate over the max for every item
            # so that the impulse responses don't induce any
            # delay.
            idx = other.audio_data.abs().argmax(axis=-1)
            irs = torch.zeros_like(other.audio_data)
            for i in range(other.batch_size):
                irs[i] = torch.roll(other.audio_data[i], -idx[i].item(), -1)
            other = AudioSignal(irs, other.sample_rate)

        delta = torch.zeros_like(other.audio_data)
        delta[..., 0] = 1

        length = self.signal_length
        delta_fft = torch.fft.rfft(delta, length)
        other_fft = torch.fft.rfft(other.audio_data, length)
        self_fft = torch.fft.rfft(self.audio_data, length)

        convolved_fft = other_fft * self_fft
        convolved_audio = torch.fft.irfft(convolved_fft, length)

        delta_convolved_fft = other_fft * delta_fft
        delta_audio = torch.fft.irfft(delta_convolved_fft, length)

        # Use the delta to rescale the audio exactly as needed.
        delta_max = delta_audio.abs().max(dim=-1, keepdims=True)[0]
        scale = 1 / delta_max.clamp(1e-5)
        convolved_audio = convolved_audio * scale

        self.audio_data = convolved_audio

        return self

    def apply_ir(
        self,
        ir,
        drr: typing.Union[torch.Tensor, np.ndarray, float] = None,
        ir_eq: typing.Union[torch.Tensor, np.ndarray] = None,
        use_original_phase: bool = False,
    ):
        """Applies an impulse response to the signal. If ` is`ir_eq``
        is specified, the impulse response is equalized before
        it is applied, using the given curve.

        Parameters
        ----------
        ir : AudioSignal
            Impulse response to convolve with.
        drr : typing.Union[torch.Tensor, np.ndarray, float], optional
            Direct-to-reverberant ratio that impulse response will be
            altered to, if specified, by default None
        ir_eq : typing.Union[torch.Tensor, np.ndarray], optional
            Equalization that will be applied to impulse response
            if specified, by default None
        use_original_phase : bool, optional
            Whether to use the original phase, instead of the convolved
            phase, by default False

        Returns
        -------
        AudioSignal
            Signal with impulse response applied to it
        """
        if ir_eq is not None:
            ir = ir.equalizer(ir_eq)
        if drr is not None:
            ir = ir.alter_drr(drr)

        # Save the peak before
        max_spk = self.audio_data.abs().max(dim=-1, keepdims=True).values

        # Augment the impulse response to simulate microphone effects
        # and with varying direct-to-reverberant ratio.
        phase = self.phase
        self.convolve(ir)

        # Use the input phase
        if use_original_phase:
            self.stft()
            self.stft_data = self.magnitude * torch.exp(1j * phase)
            self.istft()

        # Rescale to the input's amplitude
        max_transformed = self.audio_data.abs().max(dim=-1, keepdims=True).values
        scale_factor = max_spk.clamp(1e-8) / max_transformed.clamp(1e-8)
        self = self * scale_factor

        return self

    def ensure_max_of_audio(self, max: float = 1.0):
        """Ensures that ``abs(audio_data) <= max``.

        Parameters
        ----------
        max : float, optional
            Max absolute value of signal, by default 1.0

        Returns
        -------
        AudioSignal
            Signal with values scaled between -max and max.
        """
        peak = self.audio_data.abs().max(dim=-1, keepdims=True)[0]
        peak_gain = torch.ones_like(peak)
        peak_gain[peak > max] = max / peak[peak > max]
        self.audio_data = self.audio_data * peak_gain
        return self

    def normalize(self, db: typing.Union[torch.Tensor, np.ndarray, float] = -24.0):
        """Normalizes the signal's volume to the specified db, in LUFS.
        This is GPU-compatible, making for very fast loudness normalization.

        Parameters
        ----------
        db : typing.Union[torch.Tensor, np.ndarray, float], optional
            Loudness to normalize to, by default -24.0

        Returns
        -------
        AudioSignal
            Normalized audio signal.
        """
        db = util.ensure_tensor(db).to(self.device)
        ref_db = self.loudness()
        gain = db - ref_db
        gain = torch.exp(gain * self.GAIN_FACTOR)

        self.audio_data = self.audio_data * gain[:, None, None]
        return self

    def volume_change(self, db: typing.Union[torch.Tensor, np.ndarray, float]):
        """Change volume of signal by some amount, in dB.

        Parameters
        ----------
        db : typing.Union[torch.Tensor, np.ndarray, float]
            Amount to change volume by.

        Returns
        -------
        AudioSignal
            Signal at new volume.
        """
        db = util.ensure_tensor(db, ndim=1).to(self.device)
        gain = torch.exp(db * self.GAIN_FACTOR)
        self.audio_data = self.audio_data * gain[:, None, None]
        return self

    def _to_2d(self):
        waveform = self.audio_data.reshape(-1, self.signal_length)
        return waveform

    def _to_3d(self, waveform):
        return waveform.reshape(self.batch_size, self.num_channels, -1)

    def pitch_shift(self, n_semitones: int, quick: bool = True):
        """Pitch shift the signal. All items in the batch
        get the same pitch shift.

        Parameters
        ----------
        n_semitones : int
            How many semitones to shift the signal by.
        quick : bool, optional
            Using quick pitch shifting, by default True

        Returns
        -------
        AudioSignal
            Pitch shifted audio signal.
        """
        device = self.device
        effects = [
            ["pitch", str(n_semitones * 100)],
            ["rate", str(self.sample_rate)],
        ]
        if quick:
            effects[0].insert(1, "-q")

        waveform = self._to_2d().cpu()
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects, channels_first=True
        )
        self.sample_rate = sample_rate
        self.audio_data = self._to_3d(waveform)
        return self.to(device)

    def time_stretch(self, factor: float, quick: bool = True):
        """Time stretch the audio signal.

        Parameters
        ----------
        factor : float
            Factor by which to stretch the AudioSignal. Typically
            between 0.8 and 1.2.
        quick : bool, optional
            Whether to use quick time stretching, by default True

        Returns
        -------
        AudioSignal
            Time-stretched AudioSignal.
        """
        device = self.device
        effects = [
            ["tempo", str(factor)],
            ["rate", str(self.sample_rate)],
        ]
        if quick:
            effects[0].insert(1, "-q")

        waveform = self._to_2d().cpu()
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, self.sample_rate, effects, channels_first=True
        )
        self.sample_rate = sample_rate
        self.audio_data = self._to_3d(waveform)
        return self.to(device)

    def apply_codec(
        self,
        preset: str = None,
        format: str = "wav",
        encoding: str = None,
        bits_per_sample: int = None,
        compression: int = None,
    ):  # pragma: no cover
        """Applies an audio codec to the signal.

        Parameters
        ----------
        preset : str, optional
            One of the keys in ``self.CODEC_PRESETS``, by default None
        format : str, optional
            Format for audio codec, by default "wav"
        encoding : str, optional
            Encoding to use, by default None
        bits_per_sample : int, optional
            How many bits per sample, by default None
        compression : int, optional
            Compression amount of codec, by default None

        Returns
        -------
        AudioSignal
            AudioSignal with codec applied.

        Raises
        ------
        ValueError
            If preset is not in ``self.CODEC_PRESETS``, an error
            is thrown.
        """
        torchaudio_version_070 = "0.7" in torchaudio.__version__
        if torchaudio_version_070:
            return self

        kwargs = {
            "format": format,
            "encoding": encoding,
            "bits_per_sample": bits_per_sample,
            "compression": compression,
        }

        if preset is not None:
            if preset in self.CODEC_PRESETS:
                kwargs = self.CODEC_PRESETS[preset]
            else:
                raise ValueError(
                    f"Unknown preset: {preset}. "
                    f"Known presets: {list(self.CODEC_PRESETS.keys())}"
                )

        waveform = self._to_2d()
        if kwargs["format"] in ["vorbis", "mp3", "ogg", "amr-nb"]:
            # Apply it in a for loop
            augmented = torch.cat(
                [
                    torchaudio.functional.apply_codec(
                        waveform[i][None, :], self.sample_rate, **kwargs
                    )
                    for i in range(waveform.shape[0])
                ],
                dim=0,
            )
        else:
            augmented = torchaudio.functional.apply_codec(
                waveform, self.sample_rate, **kwargs
            )
        augmented = self._to_3d(augmented)

        self.audio_data = augmented
        return self

    def mel_filterbank(self, n_bands: int):
        """Breaks signal into mel bands.

        Parameters
        ----------
        n_bands : int
            Number of mel bands to use.

        Returns
        -------
        torch.Tensor
            Mel-filtered bands, with last axis being the band index.
        """
        filterbank = (
            julius.SplitBands(self.sample_rate, n_bands).float().to(self.device)
        )
        filtered = filterbank(self.audio_data)
        return filtered.permute(1, 2, 3, 0)

    def equalizer(self, db: typing.Union[torch.Tensor, np.ndarray]):
        """Applies a mel-spaced equalizer to the audio signal.

        Parameters
        ----------
        db : typing.Union[torch.Tensor, np.ndarray]
            EQ curve to apply.

        Returns
        -------
        AudioSignal
            AudioSignal with equalization applied.
        """
        db = util.ensure_tensor(db)
        n_bands = db.shape[-1]
        fbank = self.mel_filterbank(n_bands)

        # If there's a batch dimension, make sure it's the same.
        if db.ndim == 2:
            if db.shape[0] != 1:
                assert db.shape[0] == fbank.shape[0]
        else:
            db = db.unsqueeze(0)

        weights = (10**db).to(self.device).float()
        fbank = fbank * weights[:, None, None, :]
        eq_audio_data = fbank.sum(-1)
        self.audio_data = eq_audio_data
        return self

    def clip_distortion(
        self, clip_percentile: typing.Union[torch.Tensor, np.ndarray, float]
    ):
        """Clips the signal at a given percentile. The higher it is,
        the lower the threshold for clipping.

        Parameters
        ----------
        clip_percentile : typing.Union[torch.Tensor, np.ndarray, float]
            Values are between 0.0 to 1.0. Typical values are 0.1 or below.

        Returns
        -------
        AudioSignal
            Audio signal with clipped audio data.
        """
        clip_percentile = util.ensure_tensor(clip_percentile, ndim=1)
        min_thresh = torch.quantile(self.audio_data, clip_percentile / 2, dim=-1)
        max_thresh = torch.quantile(self.audio_data, 1 - (clip_percentile / 2), dim=-1)

        nc = self.audio_data.shape[1]
        min_thresh = min_thresh[:, :nc, :]
        max_thresh = max_thresh[:, :nc, :]

        self.audio_data = self.audio_data.clamp(min_thresh, max_thresh)

        return self

    def quantization(
        self, quantization_channels: typing.Union[torch.Tensor, np.ndarray, int]
    ):
        """Applies quantization to the input waveform.

        Parameters
        ----------
        quantization_channels : typing.Union[torch.Tensor, np.ndarray, int]
            Number of evenly spaced quantization channels to quantize
            to.

        Returns
        -------
        AudioSignal
            Quantized AudioSignal.
        """
        quantization_channels = util.ensure_tensor(quantization_channels, ndim=3)

        x = self.audio_data
        x = (x + 1) / 2
        x = x * quantization_channels
        x = x.floor()
        x = x / quantization_channels
        x = 2 * x - 1

        residual = (self.audio_data - x).detach()
        self.audio_data = self.audio_data - residual
        return self

    def mulaw_quantization(
        self, quantization_channels: typing.Union[torch.Tensor, np.ndarray, int]
    ):
        """Applies mu-law quantization to the input waveform.

        Parameters
        ----------
        quantization_channels : typing.Union[torch.Tensor, np.ndarray, int]
            Number of mu-law spaced quantization channels to quantize
            to.

        Returns
        -------
        AudioSignal
            Quantized AudioSignal.
        """
        mu = quantization_channels - 1.0
        mu = util.ensure_tensor(mu, ndim=3)

        x = self.audio_data

        # quantize
        x = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
        x = ((x + 1) / 2 * mu + 0.5).to(torch.int64)

        # unquantize
        x = (x / mu) * 2 - 1.0
        x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu

        residual = (self.audio_data - x).detach()
        self.audio_data = self.audio_data - residual
        return self

    def __matmul__(self, other):
        return self.convolve(other)


class ImpulseResponseMixin:
    """These functions are generally only used with AudioSignals that are derived
    from impulse responses, not other sources like music or speech. These methods
    are used to replicate the data augmentation described in [1].

    1.  Bryan, Nicholas J. "Impulse response data augmentation and deep
        neural networks for blind room acoustic parameter estimation."
        ICASSP 2020-2020 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP). IEEE, 2020.
    """

    def decompose_ir(self):
        """Decomposes an impulse response into early and late
        field responses.
        """
        # Equations 1 and 2
        # -----------------
        # Breaking up into early
        # response + late field response.

        td = torch.argmax(self.audio_data, dim=-1, keepdim=True)
        t0 = int(self.sample_rate * 0.0025)

        idx = torch.arange(self.audio_data.shape[-1], device=self.device)[None, None, :]
        idx = idx.expand(self.batch_size, -1, -1)
        early_idx = (idx >= td - t0) * (idx <= td + t0)

        early_response = torch.zeros_like(self.audio_data, device=self.device)
        early_response[early_idx] = self.audio_data[early_idx]

        late_idx = ~early_idx
        late_field = torch.zeros_like(self.audio_data, device=self.device)
        late_field[late_idx] = self.audio_data[late_idx]

        # Equation 4
        # ----------
        # Decompose early response into windowed
        # direct path and windowed residual.

        window = torch.zeros_like(self.audio_data, device=self.device)
        for idx in range(self.batch_size):
            window_idx = early_idx[idx, 0].nonzero()
            window[idx, ..., window_idx] = self.get_window(
                "hann", window_idx.shape[-1], self.device
            )
        return early_response, late_field, window

    def measure_drr(self):
        """Measures the direct-to-reverberant ratio of the impulse
        response.

        Returns
        -------
        float
            Direct-to-reverberant ratio
        """
        early_response, late_field, _ = self.decompose_ir()
        num = (early_response**2).sum(dim=-1)
        den = (late_field**2).sum(dim=-1)
        drr = 10 * torch.log10(num / den)
        return drr

    @staticmethod
    def solve_alpha(early_response, late_field, wd, target_drr):
        """Used to solve for the alpha value, which is used
        to alter the drr.
        """
        # Equation 5
        # ----------
        # Apply the good ol' quadratic formula.

        wd_sq = wd**2
        wd_sq_1 = (1 - wd) ** 2
        e_sq = early_response**2
        l_sq = late_field**2
        a = (wd_sq * e_sq).sum(dim=-1)
        b = (2 * (1 - wd) * wd * e_sq).sum(dim=-1)
        c = (wd_sq_1 * e_sq).sum(dim=-1) - torch.pow(10, target_drr / 10) * l_sq.sum(
            dim=-1
        )

        expr = ((b**2) - 4 * a * c).sqrt()
        alpha = torch.maximum(
            (-b - expr) / (2 * a),
            (-b + expr) / (2 * a),
        )
        return alpha

    def alter_drr(self, drr: typing.Union[torch.Tensor, np.ndarray, float]):
        """Alters the direct-to-reverberant ratio of the impulse response.

        Parameters
        ----------
        drr : typing.Union[torch.Tensor, np.ndarray, float]
            Direct-to-reverberant ratio that impulse response will be
            altered to, if specified, by default None

        Returns
        -------
        AudioSignal
            Altered impulse response.
        """
        drr = util.ensure_tensor(drr, 2, self.batch_size).to(self.device)

        early_response, late_field, window = self.decompose_ir()
        alpha = self.solve_alpha(early_response, late_field, window, drr)
        min_alpha = (
            late_field.abs().max(dim=-1)[0] / early_response.abs().max(dim=-1)[0]
        )
        alpha = torch.maximum(alpha, min_alpha)[..., None]

        aug_ir_data = (
            alpha * window * early_response
            + ((1 - window) * early_response)
            + late_field
        )
        self.audio_data = aug_ir_data
        self.ensure_max_of_audio()
        return self
