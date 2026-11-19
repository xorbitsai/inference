import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from transformers import Qwen2Config, Qwen2ForCausalLM

from ...models.dots_tts.config import ModelConfig
from ...modules.backbone.dit import DiT
from ...modules.backbone.encoder import VAESemanticEncoder
from ...utils.tokenizer import (
    AUDIO_COMP_SPAN_TOKEN,
    AUDIO_GEN_SPAN_TOKEN,
    TEXT_COND_END_TOKEN,
    require_token_id,
)
from ...utils.util import get_mask_from_lengths, mask_data


@dataclass(frozen=True)
class DotsTtsForwardOutput:
    llm_logits: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    eos_out: torch.Tensor


class DotsTtsCore(nn.Module):
    # region Module construction
    def __init__(
        self,
        config: ModelConfig,
        llm_config: Qwen2Config,
        tokenizer=None,
        *,
        latent_stats_path,
    ):
        super().__init__()
        self.config = config
        self.fm_hidden_size = config.DiT.hidden_size
        self.hidden_patch_size = 1
        self.cfg_droprate = config.get("cfg_droprate", 0.2)
        self.latent_patch_size = config.patch_size
        self.latent_dim = config.latent_dim
        self.xvec_dim = config.campplus_embedding_size
        self.xvec_drop_rate = config.get("xvec_drop_rate", 0.2)

        # Setup tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be provided before building the model.")
        if llm_config is None:
            raise RuntimeError("LLM config must be provided before building the model.")
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        self.audio_gen_span_id = require_token_id(self.tokenizer, AUDIO_GEN_SPAN_TOKEN)
        self.audio_comp_span_id = require_token_id(
            self.tokenizer, AUDIO_COMP_SPAN_TOKEN
        )
        self.text_cond_end_id = require_token_id(self.tokenizer, TEXT_COND_END_TOKEN)

        # Setup LLM with language modeling head so we can obtain logits directly
        llm_config = copy.deepcopy(llm_config)
        llm_config.vocab_size = len(self.tokenizer)
        self.llm = Qwen2ForCausalLM._from_config(
            llm_config,
            dtype=torch.float32,
        )
        self.llm_hidden_size = self.llm.config.hidden_size

        self.patch_encoder = VAESemanticEncoder(
            in_dim=self.latent_dim,
            out_dim=self.llm_hidden_size,
            config=config,
        )

        # Setup Flow matching related modules
        self.hidden_proj = nn.Linear(self.llm_hidden_size, self.fm_hidden_size)
        self.latent_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.coordinate_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.xvec_proj = nn.Sequential(
            nn.Linear(self.xvec_dim, self.fm_hidden_size),
            nn.LayerNorm(self.fm_hidden_size),
        )
        self.meanflow_config = config.meanflow if config.meanflow is not None else None
        self.mode = (
            "meanflow"
            if self.meanflow_config is not None and self.meanflow_config.enabled
            else "flow_matching"
        )
        dit_mode = (
            "meanflow"
            if self.mode == "meanflow"
            and self.meanflow_config.use_duration_embedding
            else "flow_matching"
        )
        self.velocity_field_predictor = DiT(
            in_dim=self.fm_hidden_size,
            out_dim=self.latent_dim,
            transformer_config=config.DiT,
            mode=dit_mode,
        )

        # Setup eos predictor
        self.eos_proj = nn.Sequential(
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
            nn.SiLU(),
            nn.Linear(self.llm_hidden_size, 2),
        )

        # Helpers
        self.fm_helper = FlowMatchingHelper(sigma=config.get("fm_sigma", 0.0))
        self.causal_helper = CausalHelper()
        self.io_helper = IOHelper(latent_stats_path=latent_stats_path)
        self.audio_span_token_ids: list[int] = [
            self.audio_gen_span_id,
            self.audio_comp_span_id,
        ]
    # endregion Module construction

    # region Training forward path
    def forward(self, data: dict[str, Any]) -> DotsTtsForwardOutput:
        input_ids: torch.Tensor = data["input_ids"]
        input_ids_lengths: torch.Tensor = data["input_ids_lengths"]
        input_span_mask: torch.Tensor = data["input_span_mask"]
        output_span_mask: torch.Tensor = data["output_span_mask"]
        batch_size = input_ids.size(0)
        device = input_ids.device

        latents: torch.Tensor | None = data.get("latents")
        latents_sampled: torch.Tensor | None = data.get("latents_sampled")
        latent_lengths: torch.Tensor | None = data.get("latent_lengths")
        has_latents = latents is not None or latents_sampled is not None

        patch_embeddings: torch.Tensor | None
        valid_patch_counts: torch.Tensor | None
        if has_latents:
            if latents_sampled is None:
                latents_sampled = self.io_helper.sample_from_latent(latents)
            patch_embeddings = self.patch_encoder(
                latents_sampled, x_lens=latent_lengths
            )
            valid_patch_counts = latent_lengths // self.latent_patch_size
            latents_sampled = self.io_helper.normalize(latents_sampled)
        else:
            latents_sampled = None
            patch_embeddings = None
            valid_patch_counts = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )

        input_span_counts = input_span_mask.sum(dim=1)
        if input_span_counts.sum() > 0 and patch_embeddings is None:
            raise RuntimeError(
                "Found audio span tokens but no latents provided to compute patch embeddings."
            )

        # Token embeddings with audio span replacement
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        if patch_embeddings is not None:
            inputs_embeds = inputs_embeds.clone()
            patch_embeddings = patch_embeddings.to(inputs_embeds.dtype)
            for b in range(batch_size):
                span_num = input_span_counts[b].item()
                if span_num == 0:
                    continue
                expected = valid_patch_counts[b].item()
                if expected != span_num:
                    raise RuntimeError(
                        f"Mismatch between span tokens ({span_num}) and latent patches ({expected}) for sample {b}."
                    )
                indices = input_span_mask[b].nonzero(as_tuple=False).squeeze(-1)
                inputs_embeds[b, indices, :] = patch_embeddings[b, :span_num, :]

        # LLM forward pass to obtain logits & hidden states
        _llm_attn_mask, llm_seq_mask, _ = self.causal_helper.create_causal_mask_and_pos(
            seq_lens=input_ids_lengths, max_len=input_ids.size(1)
        )
        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=llm_seq_mask.long(),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        llm_logits = llm_outputs.logits  # [B, L, V]
        llm_hidden = llm_outputs.hidden_states[-1]  # [B, L, H]

        # eos prediction, before cfg masking
        eos = self.eos_proj(llm_hidden.detach())

        # Flow matching forward
        total_patches = int(output_span_mask.sum().item())
        if total_patches > 0 and latents_sampled is None:
            raise RuntimeError("Flow matching requested but latents are missing.")
        if total_patches > 0:
            xvec_cond = self.xvec_proj(data["xvector"])
            vocal_mask = data.get("vocal_mask")
            if vocal_mask is None:
                vocal_mask = torch.ones((batch_size,), device=device, dtype=torch.bool)
            xvec_drop_mask = (
                torch.empty((batch_size,), device=device, dtype=torch.float32).uniform_(
                    0, 1
                )
                < self.xvec_drop_rate
            )
            xvec_drop_mask = xvec_drop_mask & vocal_mask
            xvec_cond = mask_data(xvec_cond, xvec_drop_mask)

            hiddens_for_fm = torch.where(
                output_span_mask.unsqueeze(-1), llm_hidden, inputs_embeds
            )

            # Prepare DiT inputs
            (
                fm_seq,
                target,
                fm_attn_mask,
                fm_seq_mask,
                fm_pos_ids,
                times,
                fm_prefix_lengths,
                fm_gen_lengths,
                fm_gen_patch_size,
            ) = self.io_helper.prepare_inputs_for_dit(
                hiddens=hiddens_for_fm,
                hidden_lens=input_ids_lengths,
                latents=latents_sampled,
                latent_lens=latent_lengths,
                hidden_proj=self.hidden_proj,
                latent_proj=self.latent_proj,
                noisy_proj=self.coordinate_proj,
                span_mask=output_span_mask,
                hidden_patch_size=self.hidden_patch_size,
                latent_patch_size=self.latent_patch_size,
                fm_helper=self.fm_helper,
                cfg_droprate=self.cfg_droprate,
            )

            # Predict velocity field
            vt = self.velocity_field_predictor(
                x=fm_seq,
                timesteps=times,
                pos_ids=fm_pos_ids,
                mask=fm_seq_mask,
                attn_mask=fm_attn_mask,
                return_hidden_stats=False,
                g_cond=xvec_cond,
            )

            # Get predictions and targets
            pred = self.io_helper.get_dit_outputs(
                pred_v=vt,
                fm_prefix_lengths=fm_prefix_lengths,
                fm_gen_lengths=fm_gen_lengths,
                fm_gen_patch_size=fm_gen_patch_size,
                latent_patch_size=self.latent_patch_size,
            )
        else:
            # Dummy forward for velocity_field_predictor to keep gradients connected in DDP
            dummy_length = self.latent_patch_size
            dummy_seq_h = llm_hidden.new_zeros((1, dummy_length, self.llm_hidden_size))
            dummy_seq_h = self.hidden_proj(dummy_seq_h) * 0.0  # dummy op for ddp
            dummy_seq_l = llm_hidden.new_zeros((1, dummy_length, self.latent_dim))
            dummy_seq_l = self.latent_proj(dummy_seq_l) * 0.0  # dummy op for ddp
            dummy_seq_c = llm_hidden.new_zeros((1, dummy_length, self.latent_dim))
            dummy_seq_c = self.coordinate_proj(dummy_seq_c) * 0.0  # dummy op for ddp
            dummy_seq = dummy_seq_h + dummy_seq_l + dummy_seq_c
            dummy_times = torch.zeros((1,), device=device, dtype=torch.float32)
            dummy_attn_mask = torch.ones(
                (1, dummy_length, dummy_length), device=device, dtype=torch.bool
            )
            dummy_out = self.velocity_field_predictor(
                x=dummy_seq,
                timesteps=dummy_times,
                attn_mask=dummy_attn_mask,
            )
            pred = dummy_out[:, -self.latent_patch_size :, :]
            target = pred.detach()

        return DotsTtsForwardOutput(
            llm_logits=llm_logits,
            pred=pred,
            target=target,
            eos_out=eos,
        )
    # endregion Training forward path

    # region Autoregressive and flow-matching inference steps
    def _llm_base_model(self) -> nn.Module:
        base_model = getattr(self.llm, "base_model", None)
        if base_model is not None and base_model is not self.llm:
            return base_model

        base_model_prefix = getattr(self.llm, "base_model_prefix", None)
        if base_model_prefix:
            prefixed_model = getattr(self.llm, base_model_prefix, None)
            if prefixed_model is not None and prefixed_model is not self.llm:
                return prefixed_model

        raise RuntimeError(
            "Qwen2ForCausalLM did not expose a base transformer model. "
            "Add an adapter for this architecture before optimized inference."
        )

    @staticmethod
    def _last_hidden_state(outputs: Any) -> torch.Tensor | None:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            return hidden
        if isinstance(outputs, tuple) and outputs:
            first = outputs[0]
            if isinstance(first, torch.Tensor):
                return first
        return None

    def _compute_lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        output_embeddings = self.llm.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("LLM does not expose output embeddings.")
        return output_embeddings(hidden)

    @torch.no_grad()
    def fm_solver_step(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        input_sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        hidden_size: int,
        patch_size: int,
        g_cond: torch.Tensor | None,
        guidance_scale: torch.Tensor | float,
    ) -> torch.Tensor:
        batch_size = input_sequence.size(0)
        if input_sequence.shape != cfg_sequence.shape:
            raise ValueError(
                "FM input_sequence and cfg_sequence must share the same shape."
            )
        if input_sequence.size(1) < patch_size:
            raise ValueError(
                "FM input sequence must reserve at least one latent patch slot."
            )
        latent_start = input_sequence.size(1) - patch_size
        z = self.coordinate_proj(z)
        z_c = input_sequence.clone()
        z_c[:, latent_start:] = z
        z_branches = [z_c]
        g_cond_t = (
            None if g_cond is None else g_cond.to(device=z_c.device, dtype=z_c.dtype)
        )
        g_cond_branches = None if g_cond_t is None else [g_cond_t]

        z_cfg = cfg_sequence.clone()
        z_cfg[:, latent_start:] = z
        z_branches.append(z_cfg)
        if g_cond_branches is not None:
            g_cond_branches.append(torch.zeros_like(g_cond_t))

        z_z = torch.cat(z_branches, dim=0)
        t_t = t.reshape(1).repeat(len(z_branches))
        if g_cond_branches is not None:
            g_cond_t = torch.cat(g_cond_branches, dim=0)
        vt = self.velocity_field_predictor(
            x=z_z,
            timesteps=t_t,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond_t,
            hidden_size=patch_size * 2 + hidden_size,
            patch_size=patch_size + 1,
        )
        vt = vt[:, latent_start:]
        vt_c = vt[:batch_size]
        vt_u = vt[batch_size:]
        if not torch.is_tensor(guidance_scale):
            guidance_scale = vt_c.new_tensor(float(guidance_scale))
        else:
            guidance_scale = guidance_scale.to(device=vt_c.device, dtype=vt_c.dtype)
        return vt_c + guidance_scale * (vt_c - vt_u)

    @torch.no_grad()
    def step_llm(
        self,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any | None]:
        provided = int(inputs_embeds is not None) + int(input_ids is not None)
        if provided != 1:
            raise ValueError(
                "Exactly one of inputs_embeds or input_ids must be provided to step_llm()."
            )

        if inputs_embeds is not None:
            pass
        else:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = outputs.hidden_states[-1]
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        return inputs_embeds, hidden, logits, past_key_values

    @torch.no_grad()
    def _meanflow_step_fm(
        self,
        *,
        input_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        patch_size: int,
        g_cond: torch.Tensor | None = None,
        nfe: int = 2,
        solver_step: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if nfe <= 0:
            raise ValueError(f"MeanFlow nfe must be positive, got {nfe}.")
        batch_size = input_sequence.size(0)
        device = input_sequence.device
        dtype = input_sequence.dtype
        solver_step = self.meanflow_solver_step if solver_step is None else solver_step
        z = (
            torch.randn(
                (batch_size, patch_size, self.latent_dim),
                device=device,
                dtype=dtype,
            )
        )
        times = torch.linspace(0.0, 1.0, nfe + 1, device=device, dtype=dtype)

        for step in range(nfe):
            t = times[step].expand(batch_size)
            dt = (times[step + 1] - times[step]).expand(batch_size)
            z = solver_step(
                z,
                t=t,
                dt=dt,
                input_sequence=input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                patch_size=patch_size,
                g_cond=g_cond,
            ).clone()
        return z

    @torch.no_grad()
    def meanflow_solver_step(
        self,
        z: torch.Tensor,
        *,
        t: torch.Tensor,
        dt: torch.Tensor,
        input_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        patch_size: int,
        g_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_sequence.size(1) < patch_size:
            raise ValueError(
                "MeanFlow input sequence must reserve at least one latent patch slot."
            )
        latent_start = input_sequence.size(1) - patch_size
        z_proj = self.coordinate_proj(z)
        z_c = input_sequence.clone()
        z_c[:, latent_start:] = z_proj
        vt = self.velocity_field_predictor(
            x=z_c,
            timesteps=t,
            duration=dt,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond,
        )
        velocity = vt[:, latent_start:]
        return z + velocity * dt.view(-1, 1, 1)

    @torch.no_grad()
    def _flow_matching_step_fm(
        self,
        *,
        input_sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        hidden_size: int,
        patch_size: int,
        g_cond: torch.Tensor | None = None,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 3.0,
        solver_step: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = input_sequence.size(0)
        num_evals = 0
        solver_step = self.fm_solver_step if solver_step is None else solver_step
        guidance_scale_tensor = input_sequence.new_tensor(float(guidance_scale))

        # Prepare ODE solver
        def solver(t, z):
            nonlocal num_evals
            num_evals += 1
            return solver_step(
                t,
                z,
                input_sequence=input_sequence,
                cfg_sequence=cfg_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                hidden_size=hidden_size,
                patch_size=patch_size,
                g_cond=g_cond,
                guidance_scale=guidance_scale_tensor,
            )

        # Prepare noise as initial coordinate
        noise = torch.randn(
            (batch_size, patch_size, self.latent_dim),
            dtype=input_sequence.dtype,
            device=input_sequence.device,
        )
        # Solve
        times = torch.tensor(
            [0.0, 1.0], dtype=input_sequence.dtype, device=input_sequence.device
        )
        if ode_method in ["euler", "midpoint", "rk4"]:  # fixed step size methods
            options = {"step_size": 1.0 / num_steps}
        else:
            logger.warning(
                "Using adaptive step size ODE solver for FM, NFE is not guaranteed: "
                "ode_method={}",
                ode_method,
            )
            options = {}
        trajectory = odeint(
            func=solver,
            y0=noise,
            t=times,
            atol=1e-5,
            rtol=1e-5,
            method=ode_method,
            options=options,
        )
        # print(f"Expected NFE: {num_steps}, Actual NFE: {num_evals}")
        return trajectory[-1]

    @torch.no_grad()
    def step_fm(
        self,
        input_sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        hidden_size: int,
        patch_size: int,
        g_cond: torch.Tensor | None = None,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 3.0,
        solver_step: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.mode == "meanflow":
            return self._meanflow_step_fm(
                input_sequence=input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                patch_size=patch_size,
                g_cond=g_cond,
                nfe=num_steps,
                solver_step=solver_step,
            )

        return self._flow_matching_step_fm(
            input_sequence=input_sequence,
            cfg_sequence=cfg_sequence,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            hidden_size=hidden_size,
            patch_size=patch_size,
            g_cond=g_cond,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            solver_step=solver_step,
        )
    # endregion Autoregressive and flow-matching inference steps


class FlowMatchingHelper:
    """
    Base helper for computing x_t and u_t, given target x_1 and noise x_0
    ref:  Flow matching for generative modeling, Lipman
    """

    def __init__(self, sigma=1e-5):
        self.sigma = sigma

    def compute_mu_t(self, x1, t):
        return t * x1

    def compute_sigma_t(self, t):
        return 1 - (1 - self.sigma) * t

    def sample_x_t(self, x0, x1, t):
        mu_t = self.compute_mu_t(x1, t)
        sigma_t = self.compute_sigma_t(t)
        return mu_t + sigma_t * x0

    def compute_u_t(self, x0, x1):
        return x1 - (1 - self.sigma) * x0

    def compute_xt_ut(self, x1, t=None, x0=None):
        if x0 is None:
            x0 = torch.randn_like(x1, device=x1.device)
        if t is None:
            t = torch.rand(x1.size(0), dtype=x1.dtype, device=x1.device)
        times = t
        t = t.reshape(-1, *([1] * (x1.dim() - 1)))
        xt = self.sample_x_t(x0, x1, t)
        ut = self.compute_u_t(x0, x1)
        return xt, ut, times


class CausalHelper:
    def create_causal_mask_and_pos(self, seq_lens, max_len):
        seq_mask = get_mask_from_lengths(seq_lens, max_len=max_len).unsqueeze(1)
        causal_mask = (
            torch.ones((max_len, max_len), device=seq_lens.device).triu(1).bool()
        )
        causal_mask = ~causal_mask.unsqueeze(0)
        attn_mask = seq_mask & causal_mask
        return attn_mask, seq_mask.squeeze(1), None

    def create_causal_chunk_mask_and_pos(
        self,
        batch_size,
        C_lens,
        Z_lens,
        span_mask,
        patch_size=8,
    ):
        device = C_lens.device
        total_lens = C_lens + Z_lens
        attn_mask = torch.zeros(
            (batch_size, total_lens.max(), total_lens.max()),
            device=device,
            dtype=torch.bool,
        )
        pos_ids = []
        # | C2C |     |
        # | Z2C | Z2Z |
        for i in range(batch_size):
            C_len = C_lens[i]
            Z_len = Z_lens[i]

            # C2C parts are standard causal attention
            attn_mask[i, :C_len, :C_len] = (
                torch.ones((C_len, C_len), device=device, dtype=torch.bool)
                .triu(1)
                .logical_not()
            )
            # Position ids in C parts are 0, 1, 2, ..., n
            c_pos = torch.arange(C_len, device=device, dtype=torch.float32)

            # Z2Z parts are block diag attention
            assert Z_len % patch_size == 0, "Z_len must be multiple of patch_size"
            attn_mask[i, C_len : C_len + Z_len, C_len : C_len + Z_len] = (
                torch.block_diag(
                    *[
                        torch.ones(
                            (patch_size, patch_size), device=device, dtype=torch.bool
                        )
                    ]
                    * (Z_len // patch_size)
                )
            )

            # Z2C parts is full attention before current patch latents
            # build according to span_mask
            j_indices = torch.arange(Z_len, device=device)
            patch_indices = j_indices // patch_size
            patch_in_c_indices = torch.where(span_mask[i])[0][patch_indices]
            attn_mask[
                i,
                C_len + j_indices.unsqueeze(1),
                torch.arange(C_len, device=device).unsqueeze(0),
            ] = torch.arange(C_len, device=device).unsqueeze(
                0
            ) < patch_in_c_indices.unsqueeze(1)
            # Position ids in Z parts start from current patch latents index in C parts
            z_pos = (patch_in_c_indices + j_indices % patch_size).to(torch.float32)
            pos_ids.append(torch.cat([c_pos, z_pos]))
        seq_mask = get_mask_from_lengths(total_lens, max_len=total_lens.max().item())
        pos_ids = pad_sequence(pos_ids, batch_first=True, padding_value=0.0).to(
            C_lens.device
        )
        return attn_mask, seq_mask, pos_ids


class IOHelper:
    def __init__(self, latent_stats_path=None):
        if latent_stats_path is not None:
            latent_stats = torch.load(latent_stats_path, weights_only=False)
            self.global_mean = torch.as_tensor(latent_stats["mean"])
            self.global_var = torch.as_tensor(latent_stats["var"])
        else:
            self.global_mean = None
            self.global_var = None

    def normalize(self, x):
        if self.global_mean is not None and self.global_var is not None:
            x = (x - self.global_mean.to(x.device)) / torch.sqrt(
                self.global_var.to(x.device)
            )
        return x

    def denormalize(self, x):
        """Restore physical-space latents in FP32 for the audio decoder."""
        x = x.float()
        if self.global_mean is not None and self.global_var is not None:
            mean = self.global_mean.to(device=x.device, dtype=torch.float32)
            var = self.global_var.to(device=x.device, dtype=torch.float32)
            x = x * torch.sqrt(var) + mean
        return x

    @staticmethod
    def sample_from_latent(latent):
        mean, log_std = latent.chunk(2, 1)
        z = mean + torch.randn_like(mean) * torch.exp(log_std)
        return z.transpose(1, 2)

    @staticmethod
    def prepare_inputs_for_dit(
        hiddens,
        hidden_lens,
        latents,
        latent_lens,
        hidden_proj,
        latent_proj,
        noisy_proj,
        span_mask,
        hidden_patch_size,
        latent_patch_size,
        fm_helper,
        cfg_droprate=-1,
    ):
        assert hidden_patch_size == 1, "Hidden patch size > 1 is not supported."

        B, _, _, device = *hiddens.shape, hiddens.device

        # Gather span hidden states for flow matching using span_mask
        span_hidden_list = []
        for b in range(B):
            indices = span_mask[b].nonzero(as_tuple=False).squeeze(-1)
            span_hidden_list.append(hiddens[b, indices, :])
        hiddens = pad_sequence(span_hidden_list, batch_first=True, padding_value=0.0)
        hidden_lens = torch.tensor(
            [t.size(0) for t in span_hidden_list], device=device, dtype=torch.long
        )

        # Update span_mask to be all True for the new lengths
        max_len = hiddens.size(1)
        span_mask = torch.arange(max_len, device=device).expand(
            B, max_len
        ) < hidden_lens.unsqueeze(1)

        # Prepare history latents
        history_latents = latent_proj(latents)
        fm_dim = history_latents.shape[-1]
        assert (latent_patch_size * history_latents.size(1) % latents.size(1)) == 0
        latent_history_patch_size = (
            latent_patch_size * history_latents.size(1) // latents.size(1)
        )

        # Prepare llm hidden with cfg masking
        cfg_mask = (
            torch.empty((B,), dtype=torch.float, device=latents.device).uniform_(0, 1)
            < cfg_droprate
        )
        hiddens = hidden_proj(mask_data(hiddens, cfg_mask))

        # Prepare noise latents
        xt, ut, times = fm_helper.compute_xt_ut(latents)
        projected_noise = noisy_proj(xt)

        # Initialize empty fm_seq
        hist_chunk_size = hidden_patch_size + latent_history_patch_size
        valid_patch_counts = latent_lens // latent_patch_size
        fm_prefix_lengths = hidden_lens + valid_patch_counts * (
            hist_chunk_size - hidden_patch_size
        )
        fm_gen_lengths = latent_lens + valid_patch_counts * hidden_patch_size
        fm_gen_patch_size = hidden_patch_size + latent_patch_size
        fm_seq_lengths = fm_prefix_lengths + fm_gen_lengths
        fm_seq = torch.zeros(
            (B, fm_seq_lengths.max().item(), fm_dim),
            dtype=history_latents.dtype,
            device=device,
        )
        fm_target = []
        patch_context_lengths = []
        history_latent_span_mask = torch.zeros(
            (B, fm_seq_lengths.max().item()), dtype=torch.bool, device=device
        )  # to mark start positions of each history latents

        # Fill fm_seq
        for b in range(B):
            # Step 1: Interleave hiddens at span positions with patched_latents
            interleaved = []
            span_mask_b = span_mask[b, : hidden_lens[b]]
            interleaved.append(
                hiddens[b, : hidden_lens[b]][span_mask_b].reshape(
                    valid_patch_counts[b], hidden_patch_size, fm_dim
                )
            )
            interleaved.append(
                history_latents[
                    b, : valid_patch_counts[b] * latent_history_patch_size, :
                ].reshape(valid_patch_counts[b], latent_history_patch_size, fm_dim)
            )
            interleaved = torch.cat(interleaved, dim=1)
            interleaved = rearrange(
                interleaved, "n h d -> (n h) d"
            )  # [num_spans*hist_chunk_size, D]

            # Step 2: Build mapping from input positions to fm positions
            position_increment = torch.where(
                span_mask_b, hist_chunk_size, 1
            )  # span->hist_chunk_size, non-span->1
            fm_seq_positions = (
                torch.cumsum(position_increment, dim=0) - position_increment
            )

            # Step 3: Scatter non-span hiddens
            non_span_mask = ~span_mask_b
            non_span_indices = fm_seq_positions[non_span_mask]  # [num_non_spans]
            fm_seq[b, non_span_indices, :] = hiddens[b, : hidden_lens[b]][
                non_span_mask, :
            ]

            # Step 4: Scatter interleaved span tokens
            span_indices = fm_seq_positions[span_mask_b]  # [num_spans]
            span_indices_expanded = torch.stack(
                [span_indices + i for i in range(hist_chunk_size)], dim=1
            )  # [num_spans, hist_chunk_size]
            span_indices_flat = span_indices_expanded.reshape(
                -1
            )  # [num_spans*hist_chunk_size]
            fm_seq[b, span_indices_flat, :] = interleaved
            history_latent_span_mask[b, span_indices] = True
            patch_context_lengths.append(span_indices.clone())

            # Step 5: Fill with noise latents at the end
            noise_part = []
            span_mask_b = span_mask[b, : hidden_lens[b]]
            noise_part.append(
                hiddens[b, : hidden_lens[b]][span_mask_b].reshape(
                    valid_patch_counts[b], hidden_patch_size, fm_dim
                )
            )
            noise_part.append(
                projected_noise[b, : latent_lens[b], :].reshape(
                    valid_patch_counts[b], latent_patch_size, fm_dim
                )
            )
            noise_part = torch.cat(noise_part, dim=1)
            noise_part = rearrange(noise_part, "n h d -> (n h) d")
            noise_start = fm_seq_positions[-1] + position_increment[-1]
            noise_end = noise_start + fm_gen_lengths[b]
            fm_seq[b, noise_start:noise_end, :] = noise_part

            # Step 6: prepare fm_target
            ut_b = ut[b, : latent_lens[b], :]
            fm_target.append(rearrange(ut_b, "(n p) d -> n p d", p=latent_patch_size))

        # Construct fm_attn_mask and fm_pos_ids
        fm_attn_mask, fm_seq_mask, fm_pos_ids = (
            CausalHelper().create_causal_chunk_mask_and_pos(
                batch_size=B,
                C_lens=fm_prefix_lengths,
                Z_lens=fm_gen_lengths,
                span_mask=history_latent_span_mask,
                patch_size=fm_gen_patch_size,
            )
        )
        fm_prefix_lengths = fm_prefix_lengths.unsqueeze(1)
        fm_gen_lengths = fm_gen_lengths.unsqueeze(1)
        fm_target = torch.cat(fm_target, dim=0)
        results = [
            fm_seq,
            fm_target,
            fm_attn_mask,
            fm_seq_mask,
            fm_pos_ids,
            times,
            fm_prefix_lengths,
            fm_gen_lengths,
            fm_gen_patch_size,
        ]
        return tuple(results)

    @staticmethod
    def prepare_meanflow_inputs_for_dit(
        *,
        hiddens: torch.Tensor,
        latents: torch.Tensor,
        latent_lens: torch.Tensor,
        hidden_proj,
        latent_proj,
        noisy_proj,
        span_mask: torch.Tensor,
        hidden_patch_size: int,
        latent_patch_size: int,
        cfg_mask: torch.Tensor,
        noise_latents: torch.Tensor,
    ) -> dict[str, Any]:
        if hidden_patch_size != 1:
            raise ValueError("MeanFlow training only supports hidden_patch_size=1.")

        batch_size = hiddens.size(0)
        device = hiddens.device

        span_hidden_list = []
        for batch_idx in range(batch_size):
            indices = span_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            span_hidden_list.append(hiddens[batch_idx, indices, :])
        hiddens = pad_sequence(span_hidden_list, batch_first=True, padding_value=0.0)
        hidden_lens = torch.tensor(
            [item.size(0) for item in span_hidden_list],
            device=device,
            dtype=torch.long,
        )

        history_latents = latent_proj(latents)
        fm_dim = history_latents.shape[-1]
        latent_history_patch_size = (
            latent_patch_size * history_latents.size(1) // latents.size(1)
        )

        hiddens = hidden_proj(mask_data(hiddens, cfg_mask))
        projected_noise = noisy_proj(noise_latents)

        hist_chunk_size = hidden_patch_size + latent_history_patch_size
        valid_patch_counts = latent_lens // latent_patch_size
        fm_prefix_lengths = hidden_lens + valid_patch_counts * (
            hist_chunk_size - hidden_patch_size
        )
        fm_gen_lengths = latent_lens + valid_patch_counts * hidden_patch_size
        fm_gen_patch_size = hidden_patch_size + latent_patch_size
        fm_seq_lengths = fm_prefix_lengths + fm_gen_lengths
        fm_seq = torch.zeros(
            (batch_size, int(fm_seq_lengths.max().item()), fm_dim),
            dtype=history_latents.dtype,
            device=device,
        )
        history_latent_span_mask = torch.zeros(
            (batch_size, int(fm_seq_lengths.max().item())),
            dtype=torch.bool,
            device=device,
        )
        noise_region_starts = []

        for batch_idx in range(batch_size):
            patch_count = int(valid_patch_counts[batch_idx].item())
            hidden_len = int(hidden_lens[batch_idx].item())
            if patch_count <= 0 or hidden_len <= 0:
                noise_region_starts.append(0)
                continue

            hidden_block = hiddens[batch_idx, :hidden_len].reshape(
                patch_count,
                hidden_patch_size,
                fm_dim,
            )
            history_block = history_latents[
                batch_idx,
                : patch_count * latent_history_patch_size,
                :,
            ].reshape(patch_count, latent_history_patch_size, fm_dim)
            interleaved = rearrange(
                torch.cat([hidden_block, history_block], dim=1),
                "n h d -> (n h) d",
            )

            span_indices = torch.arange(patch_count, device=device) * hist_chunk_size
            span_indices_expanded = torch.stack(
                [span_indices + idx for idx in range(hist_chunk_size)],
                dim=1,
            )
            fm_seq[batch_idx, span_indices_expanded.reshape(-1), :] = interleaved
            history_latent_span_mask[batch_idx, span_indices] = True

            noise_start = patch_count * hist_chunk_size
            noise_region_starts.append(noise_start)
            noise_part = torch.cat(
                [
                    hidden_block,
                    projected_noise[
                        batch_idx,
                        : patch_count * latent_patch_size,
                        :,
                    ].reshape(patch_count, latent_patch_size, fm_dim),
                ],
                dim=1,
            )
            noise_part = rearrange(noise_part, "n h d -> (n h) d")
            noise_end = noise_start + int(fm_gen_lengths[batch_idx].item())
            fm_seq[batch_idx, noise_start:noise_end, :] = noise_part

        fm_attn_mask, fm_seq_mask, fm_pos_ids = (
            CausalHelper().create_causal_chunk_mask_and_pos(
                batch_size=batch_size,
                C_lens=fm_prefix_lengths,
                Z_lens=fm_gen_lengths,
                span_mask=history_latent_span_mask,
                patch_size=fm_gen_patch_size,
            )
        )
        return {
            "fm_seq": fm_seq,
            "fm_attn_mask": fm_attn_mask,
            "fm_seq_mask": fm_seq_mask,
            "fm_pos_ids": fm_pos_ids,
            "fm_prefix_lengths": fm_prefix_lengths.unsqueeze(1),
            "fm_gen_lengths": fm_gen_lengths.unsqueeze(1),
            "fm_gen_patch_size": fm_gen_patch_size,
            "noise_region_starts": torch.tensor(
                noise_region_starts,
                device=device,
                dtype=torch.long,
            ),
            "noise_chunk_size": fm_gen_patch_size,
            "noise_inner_offset": hidden_patch_size,
            "valid_patch_counts": valid_patch_counts,
            "latent_lens": latent_lens,
            "latent_patch_size": latent_patch_size,
        }

    @staticmethod
    def replace_noise_latents_in_fm_seq(
        prefix_data: dict[str, Any],
        new_noise_latents: torch.Tensor,
        noisy_proj,
    ) -> torch.Tensor:
        projected = noisy_proj(new_noise_latents)
        fm_seq = prefix_data["fm_seq"].clone()
        starts = prefix_data["noise_region_starts"]
        chunk_size = int(prefix_data["noise_chunk_size"])
        inner_offset = int(prefix_data["noise_inner_offset"])
        latent_patch_size = int(prefix_data["latent_patch_size"])
        valid_patch_counts = prefix_data["valid_patch_counts"]

        for batch_idx in range(fm_seq.size(0)):
            patch_count = int(valid_patch_counts[batch_idx].item())
            if patch_count <= 0:
                continue
            base = int(starts[batch_idx].item())
            for patch_idx in range(patch_count):
                src_start = patch_idx * latent_patch_size
                dst_start = base + patch_idx * chunk_size + inner_offset
                fm_seq[
                    batch_idx,
                    dst_start : dst_start + latent_patch_size,
                    :,
                ] = projected[
                    batch_idx,
                    src_start : src_start + latent_patch_size,
                    :,
                ]
        return fm_seq

    @staticmethod
    def get_dit_outputs(
        pred_v,
        fm_prefix_lengths,
        fm_gen_lengths,
        fm_gen_patch_size,
        latent_patch_size,
    ):
        B, P = fm_prefix_lengths.shape
        fm_pred = []
        for b in range(B):
            p_offset = 0
            for p in range(P):
                latents_b = pred_v[
                    b,
                    p_offset + fm_prefix_lengths[b][p] : p_offset
                    + fm_prefix_lengths[b][p]
                    + fm_gen_lengths[b][p],
                ]
                latents_b = rearrange(
                    latents_b, "(n p) d -> n p d", p=fm_gen_patch_size
                )
                # extract only the latent parts
                latents_b = latents_b[:, -latent_patch_size:, :]
                fm_pred.append(latents_b)
                p_offset += fm_prefix_lengths[b][p] + fm_gen_lengths[b][p]
        return torch.cat(fm_pred, dim=0)
