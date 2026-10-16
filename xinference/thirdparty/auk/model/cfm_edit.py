"""
ein notation:
b - batch
n - sequence
nt - text sequence
d - dimension
"""
# ruff: noqa: F722 F821

from __future__ import annotations

import logging
from random import random
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from auk.model.utils import exists, lens_to_mask


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CFMEdit(nn.Module):
    def __init__(
        self,
        transformer,
        text_encoder,
        text_processor,
        num_channels: int,  # VAE latent dim
        odeint_kwargs: dict = dict(method="euler"),  # 'euler' | 'midpoint'
        audio_drop_prob: float = 0.3,
        cond_drop_prob: float = 0.2,
        t_sampling: str = "uniform",  # 'uniform' | 'logistic_normal'
        P_mean: float = -0.8,
        P_std: float = 0.8,
        **_ignored,  # tolerate extra config keys (e.g. legacy schedule fields)
    ):
        super().__init__()

        self.transformer = transformer
        self.dim = transformer.dim
        self.num_channels = num_channels

        # classifier-free guidance drop rates (training only)
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # flow-matching time sampling (training only)
        self.t_sampling = t_sampling
        self.P_mean = P_mean
        self.P_std = P_std

        # ODE solver options (inference)
        self.odeint_kwargs = odeint_kwargs

        # --- text encoder (LLM), frozen ---
        self.text_encoder = text_encoder
        self.text_processor = text_processor
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

        num_layers = self.text_encoder.config.text_config.num_hidden_layers
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))  # logits for softmax
        self.layer_scale = nn.Parameter(torch.ones(1))  # learnable scalar multiplier

    @property
    def device(self):
        return next(self.parameters()).device

    def sample_time(self, batch: int, dtype, device) -> torch.Tensor:
        if self.t_sampling == "uniform":
            return torch.rand((batch,), dtype=dtype, device=device)
        elif self.t_sampling == "logistic_normal":
            z = torch.randn((batch,), dtype=dtype, device=device) * self.P_std + self.P_mean
            return torch.sigmoid(z)
        else:
            raise ValueError(f"Unknown t_sampling: {self.t_sampling}")

    @staticmethod
    def build_cond_inputs(messages_batch, processor):
        formatted_texts = processor.apply_chat_template(
            messages_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        # auto-detect multimodal content (audio/image/video) anywhere in the batch
        has_mm = any(
            isinstance(c, dict) and "type" in c and c["type"] in ("audio", "image", "video")
            for msgs in messages_batch
            for m in msgs
            for c in m["content"]
        )
        if has_mm:
            from qwen_omni_utils import process_mm_info

            mm_audios, mm_images, mm_videos = process_mm_info(messages_batch, use_audio_in_video=True)
            return processor(
                text=formatted_texts,
                audio=mm_audios,
                images=mm_images,
                videos=mm_videos,
                padding=True,
                return_tensors="pt",
                use_audio_in_video=True,
            )
        return processor(
            text=formatted_texts,
            padding=True,
            return_tensors="pt",
        )

    def encode_text(self, cond_inputs, device) -> torch.Tensor:
        # Move every tensor in cond_inputs to the target device (BatchFeature.to handles this).
        if hasattr(cond_inputs, "to"):
            cond_inputs = cond_inputs.to(device)
        else:
            cond_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond_inputs.items()}

        attention_mask = cond_inputs["attention_mask"]

        # --- text encoder forward (frozen) ---
        with torch.no_grad():
            outputs = self.text_encoder(
                **cond_inputs,
                output_hidden_states=True,
            )
        all_hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors

        # --- layer fusion: ELMo-style learned weighted average with per-layer LayerNorm ---
        _, _, d_llm = all_hidden_states[0].shape
        stacked = torch.stack([F.layer_norm(h, [d_llm]) for h in all_hidden_states[1:]], dim=0)  # (num_layers, B, seq, d_llm)
        weights = F.softmax(self.layer_weights, dim=0)
        hidden = (stacked * weights[:, None, None, None]).sum(dim=0) * self.layer_scale
        return hidden, attention_mask.bool()

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"],  # VAE latent reference [B, Np, D]
        text,  # BatchFeature / dict with input_ids+attention_mask (+ optional input_features)
        duration: int | int["b"],
        *,
        lens: int["b"] | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        t_grid: list[float] | None = None,  # explicit sampling times (e.g. DMD student time_grid); overrides steps+sway
        seed: int | None = None,
        max_duration=65536,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,
        use_epss=True,
        no_ref_audio=False,
    ):
        # ODE state is target-only; ref audio prepended inside backbone.
        self.eval()

        cond = cond.to(next(self.parameters()).dtype)
        batch, cond_seq_len, device = *cond.shape[:2], cond.device

        ref_latent = cond  # [B, Np, D]
        if not exists(lens):
            ref_lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
        else:
            ref_lens = lens
        ref_mask = lens_to_mask(ref_lens, length=cond_seq_len)

        if no_ref_audio:
            ref_latent = torch.zeros_like(ref_latent)

        text_embeds, context_mask = self.encode_text(text, device)

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
        duration = duration.clamp(max=max_duration)
        target_duration = (duration - ref_lens).clamp(min=1)
        duration = ref_lens + target_duration  # reconcile after clamp

        if batch > 1:
            target_mask = lens_to_mask(target_duration)
        else:
            target_mask = None

        def fn(t, x):
            if cfg_strength < 1e-5:
                return self.transformer(
                    x=x,
                    text=text_embeds,
                    time=t,
                    mask=target_mask,
                    c_mask=context_mask,
                    ref=ref_latent,
                    ref_mask=ref_mask,
                    drop_audio_cond=False,
                    drop_text=False,
                    cache=True,
                )
            pred_cfg = self.transformer(
                x=x,
                text=text_embeds,
                time=t,
                mask=target_mask,
                c_mask=context_mask,
                ref=ref_latent,
                ref_mask=ref_mask,
                cfg_infer=True,
                cache=True,
            )
            v_cond, v_uncond = torch.chunk(pred_cfg, 2, dim=0)
            return v_cond + (v_cond - v_uncond) * cfg_strength

        y0 = []
        for dur in target_duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=ref_latent.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=torch.float32)
        if t_grid is not None:
            t = torch.tensor(t_grid, device=self.device, dtype=torch.float32)
        elif sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()
        sampled = trajectory[-1]  # [B, max_target_dur, D]

        # assemble [ref | generated] to match caller expectations
        total_max_dur = duration.amax().item()
        out = torch.zeros(batch, total_max_dur, self.num_channels, device=device, dtype=sampled.dtype)
        for i in range(batch):
            rl = ref_lens[i].item()
            out[i, :rl] = ref_latent[i, :rl]
            td = target_duration[i].item()
            out[i, rl : rl + td] = sampled[i, :td]

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"],  # VAE latent target
        text,  # BatchFeature / dict with input_ids+attention_mask (+ optional input_features)
        *,
        ref_latent: float["b np d"],
        ref_lens: int["b"] | None = None,
        lens: int["b"] | None = None,
        noise_scheduler: str | None = None,
        time: float["b"] | None = None,  # None -> sample_time; else use injected t (eval per-t grid)
        x0: float["b n d"] | None = None,  # None -> randn; else use injected shared noise (eval)
        apply_cond_drop: bool = True,  # False -> full condition, no CFG drop (eval)
    ):
        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        # encode text via LLM (text must be a pre-tokenized cond_inputs BatchFeature/dict)
        text_embeds, context_mask = self.encode_text(text, device)

        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)
        mask = lens_to_mask(lens, length=seq_len)

        # latent is x1
        x1 = inp
        # x0 is gaussian noise
        x0 = torch.randn_like(x1) if x0 is None else x0

        # time step
        time = self.sample_time(batch, dtype, self.device) if time is None else time

        # sample xt
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # cfg training with a drop rate
        if apply_cond_drop:
            drop_audio_cond = random() < self.audio_drop_prob
            if random() < self.cond_drop_prob:
                drop_audio_cond = True
                drop_text = True
            else:
                drop_text = False
        else:
            drop_audio_cond = False
            drop_text = False

        # prompt audio is prepended in sequence dim; entire target is ODE state.
        ref_seq_len = ref_latent.shape[1]
        if ref_lens is None:
            ref_lens = torch.full((batch,), ref_seq_len, device=device, dtype=torch.long)
        ref_mask = lens_to_mask(ref_lens, length=ref_seq_len)

        v_pred = self.transformer(
            x=φ,
            text=text_embeds,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            mask=mask,
            c_mask=context_mask,
            ref=ref_latent,
            ref_mask=ref_mask,
        )

        # velocity prediction + flow-matching loss (gated by padding mask)
        x_pred = φ + (1.0 - t) * v_pred
        loss = F.mse_loss(v_pred, flow, reduction="none")
        if mask is not None:
            loss = loss[mask]

        return loss.mean(), None, x_pred
