import torch
import torch.nn.functional as F
from torch import nn

from ...core import AudioSignal
from ...core import STFTParams
from ...core import util


class SpectralGate(nn.Module):
    """Spectral gating algorithm for noise reduction,
    as in Audacity/Ocenaudio. The steps are as follows:

    1.  An FFT is calculated over the noise audio clip
    2.  Statistics are calculated over FFT of the the noise
        (in frequency)
    3.  A threshold is calculated based upon the statistics
        of the noise (and the desired sensitivity of the algorithm)
    4.  An FFT is calculated over the signal
    5.  A mask is determined by comparing the signal FFT to the
        threshold
    6.  The mask is smoothed with a filter over frequency and time
    7.  The mask is appled to the FFT of the signal, and is inverted

    Implementation inspired by Tim Sainburg's noisereduce:

    https://timsainburg.com/noise-reduction-python.html

    Parameters
    ----------
    n_freq : int, optional
        Number of frequency bins to smooth by, by default 3
    n_time : int, optional
        Number of time bins to smooth by, by default 5
    """

    def __init__(self, n_freq: int = 3, n_time: int = 5):
        super().__init__()

        smoothing_filter = torch.outer(
            torch.cat(
                [
                    torch.linspace(0, 1, n_freq + 2)[:-1],
                    torch.linspace(1, 0, n_freq + 2),
                ]
            )[..., 1:-1],
            torch.cat(
                [
                    torch.linspace(0, 1, n_time + 2)[:-1],
                    torch.linspace(1, 0, n_time + 2),
                ]
            )[..., 1:-1],
        )
        smoothing_filter = smoothing_filter / smoothing_filter.sum()
        smoothing_filter = smoothing_filter.unsqueeze(0).unsqueeze(0)
        self.register_buffer("smoothing_filter", smoothing_filter)

    def forward(
        self,
        audio_signal: AudioSignal,
        nz_signal: AudioSignal,
        denoise_amount: float = 1.0,
        n_std: float = 3.0,
        win_length: int = 2048,
        hop_length: int = 512,
    ):
        """Perform noise reduction.

        Parameters
        ----------
        audio_signal : AudioSignal
            Audio signal that noise will be removed from.
        nz_signal : AudioSignal, optional
            Noise signal to compute noise statistics from.
        denoise_amount : float, optional
            Amount to denoise by, by default 1.0
        n_std : float, optional
            Number of standard deviations above which to consider
            noise, by default 3.0
        win_length : int, optional
            Length of window for STFT, by default 2048
        hop_length : int, optional
            Hop length for STFT, by default 512

        Returns
        -------
        AudioSignal
            Denoised audio signal.
        """
        stft_params = STFTParams(win_length, hop_length, "sqrt_hann")

        audio_signal = audio_signal.clone()
        audio_signal.stft_data = None
        audio_signal.stft_params = stft_params

        nz_signal = nz_signal.clone()
        nz_signal.stft_params = stft_params

        nz_stft_db = 20 * nz_signal.magnitude.clamp(1e-4).log10()
        nz_freq_mean = nz_stft_db.mean(keepdim=True, dim=-1)
        nz_freq_std = nz_stft_db.std(keepdim=True, dim=-1)

        nz_thresh = nz_freq_mean + nz_freq_std * n_std

        stft_db = 20 * audio_signal.magnitude.clamp(1e-4).log10()
        nb, nac, nf, nt = stft_db.shape
        db_thresh = nz_thresh.expand(nb, nac, -1, nt)

        stft_mask = (stft_db < db_thresh).float()
        shape = stft_mask.shape

        stft_mask = stft_mask.reshape(nb * nac, 1, nf, nt)
        pad_tuple = (
            self.smoothing_filter.shape[-2] // 2,
            self.smoothing_filter.shape[-1] // 2,
        )
        stft_mask = F.conv2d(stft_mask, self.smoothing_filter, padding=pad_tuple)
        stft_mask = stft_mask.reshape(*shape)
        stft_mask *= util.ensure_tensor(denoise_amount, ndim=stft_mask.ndim).to(
            audio_signal.device
        )
        stft_mask = 1 - stft_mask

        audio_signal.stft_data *= stft_mask
        audio_signal.istft()

        return audio_signal
