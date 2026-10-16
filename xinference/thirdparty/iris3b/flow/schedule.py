"""Discrete rectified-flow noise schedule with resolution shift.

The training schedule is a 1000-point sigma grid derived from
``1 - linspace(1.0, 0.001, N)`` (i.e. ascending 0 -> 0.999), remapped by the
SD3-style resolution shift

    sigma' = shift * sigma / (1 + (shift - 1) * sigma)

Model time is defined as ``1000 * sigma'`` - the network always sees the
*shifted* noise level. During training the value is integer-truncated
(int64 cast); samplers pass floats.
"""

import math

import torch

# "flux" law: mu is affine in image token count, anchored on
# (256 tokens -> 0.5) and (4096 tokens -> 1.15).
_FLUX_MU_SLOPE = (1.15 - 0.5) / (4096 - 256)
_FLUX_MU_INTERCEPT = 0.5 - _FLUX_MU_SLOPE * 256


def shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1 + (shift - 1) * sigma)


def resolution_shift(tokens: int, law: str, base_shift: float, base_tokens: int = 256) -> float:
    """Resolve ``flow.shift`` for a stage whose sequence length is ``tokens``.

    SD3 (arXiv:2403.03206 Eq. 23-25) derives ``alpha = sqrt(m / n)`` for a
    resolution change from n to m scalars, i.e. alpha scales with the LINEAR
    side length: anchored at 256px, that is 1x / 2x / 4x at 256 / 512 / 1024.
    ``flux`` is the alternative calibration, an affine map in token count
    (``mu = a * L + b``, ``alpha = exp(mu)``) fitted on 256..4096 tokens,
    which grows far more slowly: 1.65 / 1.88 / 3.16 at the same rungs.

    The result is a STAGE-level scalar, not a per-sample one: it is computed
    once from a nominal resolution and reused across the whole shape
    distribution -- at constant area the aspect axis moves alpha by only ~4%,
    and a per-sample alpha would make the frozen validation grid depend on the
    shape mix.
    """
    if law == "none":
        return base_shift
    if tokens <= 0:
        raise ValueError(f"token count must be positive, got {tokens}")
    if law == "sd3":
        return base_shift * math.sqrt(tokens / base_tokens)
    if law == "flux":
        mu = _FLUX_MU_SLOPE * tokens + _FLUX_MU_INTERCEPT
        return math.exp(mu)
    raise ValueError(f"unknown flow.shift_law '{law}' (none | sd3 | flux)")


class FlowSchedule:
    def __init__(self, num_timesteps: int = 1000, shift: float = 1.0):
        self.num_timesteps = num_timesteps
        self.shift = shift
        base = 1.0 - torch.linspace(1.0, 0.001, num_timesteps, dtype=torch.float64)
        sigmas64 = shift_sigma(base, shift)
        self.sigmas = sigmas64.float()  # [N], ascending, sigma[0] = 0
        # integer model-time map used at train time (truncating cast)
        self.model_times = (sigmas64 * num_timesteps).to(torch.int64)
        # per-device copies of the (tiny, constant) lookup tables so the hot
        # loop never re-issues H2D transfers
        self._tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        tables = self._tables.get(key)
        if tables is None:
            tables = (self.sigmas.to(device), self.model_times.to(device))
            self._tables[key] = tables
        return tables

    def add_noise(
        self, x0: torch.Tensor, noise: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate ``x_t = (1 - sigma) * x0 + sigma * noise`` at grid indices.

        Returns (x_t, model_time) where model_time is the integer-truncated
        ``1000 * sigma'`` the network is conditioned on.
        """
        sigmas, model_times = self._on(x0.device)
        sigma = sigmas[idx].view(-1, *([1] * (x0.ndim - 1))).to(x0.dtype)
        x_t = (1 - sigma) * x0 + sigma * noise
        t_model = model_times[idx].float()
        return x_t, t_model

    def sigma_at(self, idx: torch.Tensor) -> torch.Tensor:
        return self._on(idx.device)[0][idx]
