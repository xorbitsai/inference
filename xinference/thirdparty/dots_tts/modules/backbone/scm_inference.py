"""Inference-only sCM support for two-step artifacts."""

from __future__ import annotations

import math
from typing import Callable

import torch

from ...modules.backbone.dit_inference import (
    FLOW_MATCHING_MODE,
    DiTInferenceContext,
    DiTSolver,
    EagerDiTRunner,
    ODESchedule,
    _resolve_kv_attention_backend,
)


def _broadcast(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return values.reshape(-1, *([1] * (target.ndim - 1)))


def _coefficients(
    tau: torch.Tensor, fm_sigma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tau = tau.float()
    cos_tau = torch.cos(tau)
    sin_tau = torch.sin(tau)
    residual_scale = 1.0 - fm_sigma
    scale = sin_tau + residual_scale * cos_tau
    scale_derivative = cos_tau - residual_scale * sin_tau
    return cos_tau, sin_tau, scale, scale_derivative


def _to_flow_matching(
    x_tau: torch.Tensor, tau: torch.Tensor, fm_sigma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_tau, _sin_tau, scale, _scale_derivative = _coefficients(tau, fm_sigma)
    return x_tau.float() / _broadcast(scale, x_tau), cos_tau / scale


def _denoise(
    flow_state: torch.Tensor,
    flow_velocity: torch.Tensor,
    tau: torch.Tensor,
    fm_sigma: float,
) -> torch.Tensor:
    cos_tau, sin_tau, scale, scale_derivative = _coefficients(tau, fm_sigma)
    x_tau = _broadcast(scale, flow_state) * flow_state.float()
    trig_velocity = _broadcast(
        scale_derivative, flow_state
    ) * flow_state.float() - flow_velocity.float() / _broadcast(scale, flow_state)
    return (
        _broadcast(cos_tau, x_tau) * x_tau - _broadcast(sin_tau, x_tau) * trig_velocity
    )


def _renoise(denoised: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return _broadcast(torch.cos(tau), denoised) * denoised + _broadcast(
        torch.sin(tau), denoised
    ) * torch.randn_like(denoised)


def _tau_grid(tau_mid: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [math.pi / 2, tau_mid],
        device=device,
        dtype=torch.float32,
    )


class _SCMSchedule(ODESchedule):
    """Two-step TrigFlow denoise/re-noise schedule."""

    def __init__(self, taus: torch.Tensor, flow_times: torch.Tensor, fm_sigma: float):
        super().__init__(mode=FLOW_MATCHING_MODE, times=flow_times)
        self.taus = taus
        self.fm_sigma = fm_sigma

    def advance(
        self,
        flow_state: torch.Tensor,
        flow_velocity: torch.Tensor,
        *,
        ode_idx: int,
        batch_size: int,
        flow_dt: torch.Tensor | None,
    ) -> torch.Tensor:
        del batch_size, flow_dt
        tau = self.taus[ode_idx].expand(flow_state.size(0))
        denoised = _denoise(
            flow_state,
            flow_velocity,
            tau,
            self.fm_sigma,
        )
        if ode_idx == self.taus.numel() - 1:
            return denoised.to(dtype=flow_state.dtype)

        next_tau = self.taus[ode_idx + 1].expand(flow_state.size(0))
        next_x_tau = _renoise(denoised, next_tau)
        next_flow_state, _next_flow_time = _to_flow_matching(
            next_x_tau,
            next_tau,
            self.fm_sigma,
        )
        return next_flow_state.to(dtype=flow_state.dtype)


class _SCMEagerDiTRunner(EagerDiTRunner):
    """Full-compute sCM runner used when optimization is disabled."""

    def __init__(self, *, context: DiTInferenceContext, tau_mid: float):
        super().__init__(context=context)
        self.tau_mid = tau_mid

    def _velocity(
        self,
        t: torch.Tensor,
        flow_state: torch.Tensor,
        *,
        input_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor,
        g_cond: torch.Tensor,
    ) -> torch.Tensor:
        latent_start = input_sequence.size(1) - self.latent_patch_size
        x = self._splat_noise(self.coordinate_proj(flow_state), input_sequence)
        return self.dit(
            x=x,
            timesteps=t.reshape(1).expand(x.size(0)),
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond.to(device=x.device, dtype=x.dtype),
        )[:, latent_start:]

    def decode_next(
        self,
        *,
        sequence: torch.Tensor,
        fm_seq_len: int,
        g_cond: torch.Tensor,
        nfe: int,
        meanflow: bool,
        cfg_sequence: torch.Tensor | None = None,
        guidance_scale: float | None = None,
        ode_method: str = "euler",
    ) -> torch.Tensor:
        del nfe, meanflow, cfg_sequence, guidance_scale, ode_method
        input_sequence, _cfg_input, attn_mask, pos_ids = self._prepare_inputs(
            sequence=sequence,
            fm_seq_len=fm_seq_len,
        )
        taus = _tau_grid(self.tau_mid, sequence.device)
        x_tau = torch.randn(
            (sequence.size(0), self.latent_patch_size, self.latent_dim),
            dtype=sequence.dtype,
            device=sequence.device,
        )

        for ode_idx in range(taus.numel()):
            tau = taus[ode_idx].expand(sequence.size(0))
            flow_state, flow_time = _to_flow_matching(x_tau, tau, self.fm_sigma)
            flow_velocity = self._velocity(
                flow_time[0],
                flow_state.to(dtype=sequence.dtype),
                input_sequence=input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                g_cond=g_cond,
            )
            denoised = _denoise(
                flow_state,
                flow_velocity,
                tau,
                self.fm_sigma,
            )
            if ode_idx == taus.numel() - 1:
                return denoised.to(dtype=sequence.dtype)
            next_tau = taus[ode_idx + 1].expand(sequence.size(0))
            x_tau = _renoise(denoised, next_tau).to(dtype=sequence.dtype)
        raise AssertionError("unreachable sCM sampling loop")


class SCMDiTSolver(DiTSolver):
    """Two-step sCM solver for eager and KV-cache inference."""

    def __init__(
        self,
        context: DiTInferenceContext,
        *,
        optimize: bool,
        bucket_resolver: Callable[[int], int],
        tau_mid: float,
    ):
        if not 0.0 < tau_mid < math.pi / 2:
            raise ValueError("tau_mid must lie strictly inside (0, pi/2)")
        if not 0.0 <= context.fm_sigma < 1.0:
            raise ValueError("fm_sigma must lie inside [0, 1)")
        super().__init__(
            context,
            optimize=optimize,
            bucket_resolver=bucket_resolver,
        )
        self.branch_multiplier = 1
        self.tau_mid = tau_mid

    def _backend_for(self, device: torch.device) -> str:
        key = device.type
        if key not in self._backend_by_device_type:
            self._backend_by_device_type[key] = _resolve_kv_attention_backend(
                optimize=self.optimize,
                on_cuda=key == "cuda",
                default_backend="sdpa",
            )
        return self._backend_by_device_type[key]

    def _get_eager_runner(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> _SCMEagerDiTRunner:
        key = (str(device), dtype)
        runner = self._eager_runners.get(key)
        if runner is None:
            runner = _SCMEagerDiTRunner(context=self.context, tau_mid=self.tau_mid)
            self._eager_runners[key] = runner
        return runner

    def _validate_decode_args(
        self,
        *,
        fm_seq_len: int,
        nfe: int,
        cfg_sequence: torch.Tensor | None,
        guidance_scale: float | None,
    ) -> None:
        del cfg_sequence
        if fm_seq_len <= 0:
            raise RuntimeError(
                "Cannot decode audio before any conditioning state has been prefetched."
            )
        if nfe != 2:
            raise ValueError(f"sCM requires nfe=2, got {nfe}")
        if guidance_scale is None or guidance_scale != 0.0:
            raise ValueError("sCM requires guidance_scale=0")

    def _prepare_schedule_flow_matching(
        self, *, g_cond: torch.Tensor, nfe: int
    ) -> tuple[torch.Tensor, _SCMSchedule]:
        del nfe
        taus = _tau_grid(self.tau_mid, g_cond.device)
        dummy = torch.zeros(2, 1, 1, device=g_cond.device)
        _flow_state, flow_times = _to_flow_matching(
            dummy,
            taus,
            self.context.fm_sigma,
        )

        dit = self.context.dit
        with torch.no_grad():
            time_inputs = flow_times[:, None].expand(2, g_cond.size(0)).reshape(-1)
            condition = dit.time_embedder(time_inputs.to(dtype=g_cond.dtype))
            g_flat = g_cond[None].expand(2, -1, -1).reshape(2 * g_cond.size(0), -1)
            mods = dit.fused_adaln(condition + g_flat)
        return (
            mods.reshape(2, g_cond.size(0), -1),
            _SCMSchedule(taus, flow_times, self.context.fm_sigma),
        )
