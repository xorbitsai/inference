from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules.backbone.encoder import VAESemanticEncoder
from ...modules.backbone.inference_utils import (
    build_causal_update_mask,
    build_rotary_cos_sin,
    compile_module_forward,
    fuse_qkv_projection,
    project_attention,
)


@dataclass
class SemanticEncoderDecodeState:
    conv_tail: torch.Tensor
    layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    seq_len: int
    static: bool = True


class SemanticEncoderDecodeStep(nn.Module):
    @staticmethod
    def supports(encoder: nn.Module) -> bool:
        return (
            isinstance(encoder, VAESemanticEncoder) and len(encoder.encoder.layers) > 0
        )

    @staticmethod
    def downsample(
        encoder: nn.Module,
        latent_patch: torch.Tensor,
        *,
        conv_tail: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = latent_patch.transpose(1, 2)
        conv_input = torch.cat([conv_tail, raw], dim=-1)
        ds_proj = encoder.ds_proj
        projected = F.conv1d(
            conv_input,
            ds_proj.weight,
            ds_proj.bias,
            stride=ds_proj.stride[0],
            padding=0,
            dilation=ds_proj.dilation[0],
            groups=ds_proj.groups,
        ).transpose(1, 2)
        new_conv_tail = raw[..., -ds_proj.left_padding :]
        return encoder.in_proj(projected), new_conv_tail

    def __init__(self, encoder: nn.Module):
        super().__init__()
        if not self.supports(encoder):
            raise TypeError(
                "SemanticEncoderDecodeStep only supports VAESemanticEncoder-style modules."
            )
        self.encoder = encoder
        self.out_ds_rate = int(encoder.out_ds_rate)
        attn0 = encoder.encoder.layers[0].attn
        self.num_heads = int(attn0.num_heads)
        self.head_dim = int(attn0.head_dim)
        self.num_layers = len(encoder.encoder.layers)

    def forward(
        self,
        latent_patch: torch.Tensor,
        conv_tail: torch.Tensor,
        layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        attn_mask: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, new_conv_tail = self.downsample(
            self.encoder, latent_patch, conv_tail=conv_tail
        )
        batch_size = x.size(0)
        new_k = torch.empty(
            self.num_layers,
            batch_size,
            self.num_heads,
            self.out_ds_rate,
            self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        new_v = torch.empty_like(new_k)

        for layer_idx, layer in enumerate(self.encoder.encoder.layers):
            cache_k, cache_v = layer_caches[layer_idx]
            h = layer.attn_norm(x)
            h, k, v = project_attention(
                layer.attn,
                h,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                rotary_cos=rotary_cos if layer.attn.rotary_bias else None,
                rotary_sin=rotary_sin if layer.attn.rotary_bias else None,
                key_prefix=cache_k,
                value_prefix=cache_v,
                attn_mask=attn_mask,
                dropout_p=0.0,
            )
            new_k[layer_idx].copy_(k)
            new_v[layer_idx].copy_(v)
            x = x + h
            h = layer.ffn_norm(x)
            h = layer.ffn(h)
            x = x + h

        return self.encoder._project_embeddings(x), new_conv_tail, new_k, new_v


class SemanticEncoderDecodeProgram:
    def __init__(
        self,
        *,
        encoder: nn.Module,
        cache_capacity: int,
        block_len: int,
        device: torch.device,
        compile_step: bool,
    ):
        if not SemanticEncoderDecodeStep.supports(encoder):
            raise TypeError(
                "SemanticEncoderDecodeProgram only supports VAESemanticEncoder-style modules."
            )
        for layer in encoder.encoder.layers:
            fuse_qkv_projection(layer.attn)
        self.encoder = encoder
        self.cache_capacity = int(cache_capacity)
        self.block_len = int(block_len)
        self.device = device
        step_module = SemanticEncoderDecodeStep(encoder).eval()
        self.step = compile_module_forward(step_module) if compile_step else step_module
        self._mask_cache: dict[int, torch.Tensor] = {}
        self._rotary_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def mask_for(self, *, start_pos: int) -> torch.Tensor:
        if (cached := self._mask_cache.get(int(start_pos))) is not None:
            return cached
        mask = build_causal_update_mask(
            capacity_tokens=self.cache_capacity,
            valid_persistent_tokens=int(start_pos),
            prev_len=self.block_len,
            current_len=0,
            device=self.device,
        )
        self._mask_cache[int(start_pos)] = mask
        return mask

    def rotary_for(self, *, start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
        if (cached := self._rotary_cache.get(int(start_pos))) is not None:
            return cached
        attn0 = self.encoder.encoder.layers[0].attn
        if attn0.rotary_bias:
            rotary = build_rotary_cos_sin(
                attn0.rotary,
                start_pos=int(start_pos),
                seq_len=self.block_len,
                device=self.device,
            )
        else:
            empty = torch.empty((0,), device=self.device)
            rotary = (empty, empty)
        self._rotary_cache[int(start_pos)] = rotary
        return rotary

    def decode_patch(
        self,
        latent_patch: torch.Tensor,
        conv_tail: torch.Tensor,
        layer_caches: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        *,
        start_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_mask = self.mask_for(start_pos=int(start_pos))
        rotary_cos, rotary_sin = self.rotary_for(start_pos=int(start_pos))
        embedding, new_conv_tail, new_k, new_v = self.step(
            latent_patch,
            conv_tail,
            layer_caches,
            attn_mask,
            rotary_cos,
            rotary_sin,
        )
        dst = slice(int(start_pos), int(start_pos) + self.block_len)
        for layer_idx, (cache_k, cache_v) in enumerate(layer_caches):
            cache_k[:, :, dst, :].copy_(new_k[layer_idx])
            cache_v[:, :, dst, :].copy_(new_v[layer_idx])
        return embedding, new_conv_tail


class SemanticEncoderInference:
    def __init__(self, encoder: nn.Module):
        if not isinstance(encoder, VAESemanticEncoder):
            raise TypeError(
                f"SemanticEncoderInference only supports VAESemanticEncoder, got {type(encoder)!r}."
            )
        self.encoder = encoder
        self._programs: dict[tuple[Any, ...], SemanticEncoderDecodeProgram] = {}
        for layer in encoder.encoder.layers:
            fuse_qkv_projection(layer.attn)

    @staticmethod
    def _require_rank3(x: torch.Tensor, owner: str) -> None:
        if x.ndim != 3:
            raise ValueError(f"{owner} expects rank-3 input, got {tuple(x.shape)}.")

    @classmethod
    def _require_decode_patch(
        cls, encoder: nn.Module, x: torch.Tensor, owner: str
    ) -> None:
        cls._require_rank3(x, owner)
        if x.size(1) != int(encoder.patch_size):
            raise ValueError(
                f"decode_patch expects patch length {encoder.patch_size}, got {x.size(1)}."
            )

    def _semantic_embeddings(
        self,
        step_inputs: torch.Tensor,
        *,
        state: SemanticEncoderDecodeState,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if step_inputs.size(1) <= 0:
            raise ValueError("Semantic encoder decode expects a non-empty input.")
        semantic_encoder = self.encoder.encoder
        layer_caches = state.layer_caches
        if len(layer_caches) != len(semantic_encoder.layers):
            raise ValueError("Layer cache count does not match encoder depth.")
        if positions.ndim != 1 or positions.size(0) != step_inputs.size(1):
            raise ValueError(
                "positions must be rank-1 and match the decode block length."
            )

        is_static = bool(getattr(state, "static", True))
        cache_capacity = int(layer_caches[0][0].size(2))
        step_len = int(step_inputs.size(1))
        start_pos = int(positions[0].item())
        end_pos = start_pos + step_len
        expected_positions = torch.arange(
            start_pos,
            end_pos,
            device=positions.device,
            dtype=positions.dtype,
        )
        if not bool(torch.equal(positions, expected_positions)):
            raise ValueError("Semantic encoder decode positions must be contiguous.")
        if is_static and end_pos > cache_capacity:
            raise ValueError(
                f"Semantic encoder static cache capacity exceeded: required={end_pos} capacity={cache_capacity}."
            )
        total_key_len = end_pos
        key_positions = torch.arange(
            total_key_len,
            device=positions.device,
            dtype=torch.long,
        ).unsqueeze(0)
        attn_mask = (
            (key_positions <= positions.unsqueeze(1)) & (key_positions <= positions[-1])
        ).reshape(
            1,
            1,
            int(positions.numel()),
            total_key_len,
        )
        attn0 = semantic_encoder.layers[0].attn
        num_heads = int(attn0.num_heads)
        head_dim = int(attn0.head_dim)
        rotary_cos = rotary_sin = None
        if attn0.rotary_bias:
            rotary = attn0.rotary(positions)
            rotary_cos, rotary_sin = rotary.cos(), rotary.sin()
        x = step_inputs
        cache_updates = []
        for layer, cache in zip(semantic_encoder.layers, layer_caches, strict=True):
            h = layer.attn_norm(x)
            h, k, v = project_attention(
                layer.attn,
                h,
                num_heads=num_heads,
                head_dim=head_dim,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                key_prefix=cache[0][:, :, :start_pos, :],
                value_prefix=cache[1][:, :, :start_pos, :],
                attn_mask=attn_mask,
            )
            cache_updates.append((cache, k.detach(), v.detach()))
            x = x + h
            h = layer.ffn_norm(x)
            h = layer.ffn(h)
            x = x + h
        if is_static:
            for cache, k, v in cache_updates:
                cache[0][:, :, start_pos:end_pos, :].copy_(k)
                cache[1][:, :, start_pos:end_pos, :].copy_(v)
        if not is_static:
            state.layer_caches = tuple(
                (
                    torch.cat([cache[0], k], dim=2),
                    torch.cat([cache[1], v], dim=2),
                )
                for cache, k, v in cache_updates
            )
        return self.encoder._project_embeddings(x)

    def clear(self) -> None:
        self._programs.clear()

    def init_decode_state(
        self,
        *,
        max_audio_patch_count: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SemanticEncoderDecodeState:
        encoder = self.encoder
        layer_caches = []
        max_seq_len = int(max_audio_patch_count) * int(encoder.out_ds_rate)
        for layer in encoder.encoder.layers:
            cache_shape = (
                int(batch_size),
                int(layer.attn.num_heads),
                max_seq_len,
                int(layer.attn.head_dim),
            )
            layer_caches.append(
                (
                    torch.zeros(cache_shape, dtype=dtype, device=device),
                    torch.zeros(cache_shape, dtype=dtype, device=device),
                )
            )
        return SemanticEncoderDecodeState(
            conv_tail=torch.zeros(
                (
                    int(batch_size),
                    int(encoder.ds_proj.in_channels),
                    int(encoder.ds_proj.left_padding),
                ),
                dtype=dtype,
                device=device,
            ),
            layer_caches=tuple(layer_caches),
            seq_len=0,
            static=True,
        )

    def init_dynamic_decode_state(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SemanticEncoderDecodeState:
        encoder = self.encoder
        layer_caches = []
        for layer in encoder.encoder.layers:
            cache_shape = (
                int(batch_size),
                int(layer.attn.num_heads),
                0,
                int(layer.attn.head_dim),
            )
            layer_caches.append(
                (
                    torch.empty(cache_shape, dtype=dtype, device=device),
                    torch.empty(cache_shape, dtype=dtype, device=device),
                )
            )
        return SemanticEncoderDecodeState(
            conv_tail=torch.zeros(
                (
                    int(batch_size),
                    int(encoder.ds_proj.in_channels),
                    int(encoder.ds_proj.left_padding),
                ),
                dtype=dtype,
                device=device,
            ),
            layer_caches=tuple(layer_caches),
            seq_len=0,
            static=False,
        )

    def _ensure_dynamic_decode_state(
        self,
        state: Any | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SemanticEncoderDecodeState:
        if state is None:
            return self.init_dynamic_decode_state(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        if not getattr(state, "static", True):
            return state
        dynamic_state = self.init_dynamic_decode_state(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        self.copy_decode_state(state, dynamic_state)
        return dynamic_state

    def reset_decode_state(self, state: Any) -> None:
        state.conv_tail.zero_()
        if getattr(state, "static", True):
            for key_cache, value_cache in state.layer_caches:
                key_cache.zero_()
                value_cache.zero_()
        else:
            state.layer_caches = self.init_dynamic_decode_state(
                batch_size=state.conv_tail.size(0),
                device=state.conv_tail.device,
                dtype=state.conv_tail.dtype,
            ).layer_caches
        state.seq_len = 0

    def prefill(self, x: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        encoder = self.encoder
        self._require_rank3(x, "VAESemanticEncoder.prefill")
        if x.size(1) % int(encoder.patch_size) != 0:
            raise ValueError(
                f"Prompt latent length {x.size(1)} must be divisible by patch_size={encoder.patch_size}."
            )
        if x.size(1) == 0:
            return x.new_zeros((x.size(0), 0, encoder.out_proj.out_features)), state
        if state.conv_tail.size(0) != x.size(0):
            raise ValueError(
                "Patch encoder prefill batch size does not match state."
            )

        step_inputs = encoder.in_proj(encoder._downsample(x))
        expected_token_count = (x.size(1) // int(encoder.patch_size)) * int(
            encoder.out_ds_rate
        )
        if step_inputs.size(1) != expected_token_count:
            raise RuntimeError(
                f"Patch encoder prefill produced an unexpected token count: expected={expected_token_count} actual={step_inputs.size(1)}."
            )
        current_seq_len = int(state.seq_len)
        next_seq_len = current_seq_len + int(step_inputs.size(1))
        cache_capacity = int(state.layer_caches[0][0].size(2))
        if getattr(state, "static", True) and next_seq_len > cache_capacity:
            raise ValueError(
                f"Patch encoder prefill exceeds decode-state capacity: required={next_seq_len} capacity={cache_capacity}."
            )

        embedding = self._semantic_embeddings(
            step_inputs,
            state=state,
            positions=torch.arange(
                step_inputs.size(1), device=x.device, dtype=torch.long
            )
            + current_seq_len,
        )
        raw = x.transpose(1, 2)
        state.conv_tail.copy_(raw[..., -encoder.ds_proj.left_padding :])
        state.seq_len = next_seq_len
        return embedding, state

    @staticmethod
    def decode_state_capacity(state: Any) -> int:
        return int(state.layer_caches[0][0].size(2))

    @classmethod
    def copy_decode_state(cls, source: Any, target: Any) -> None:
        seq_len = int(source.seq_len)
        target_capacity = cls.decode_state_capacity(target)
        target_is_static = bool(getattr(target, "static", True))
        if target_is_static and seq_len > target_capacity:
            raise ValueError(
                f"Patch encoder state copy exceeds target capacity: seq_len={seq_len} capacity={target_capacity}."
            )

        target.conv_tail.copy_(source.conv_tail)
        target.seq_len = seq_len
        if not target_is_static:
            target_device = target.conv_tail.device
            target_dtype = target.conv_tail.dtype
            target.layer_caches = tuple(
                (
                    source_key[:, :, :seq_len, :]
                    .to(device=target_device, dtype=target_dtype)
                    .clone(),
                    source_value[:, :, :seq_len, :]
                    .to(device=target_device, dtype=target_dtype)
                    .clone(),
                )
                for source_key, source_value in source.layer_caches
            )
            return
        for (source_key, source_value), (target_key, target_value) in zip(
            source.layer_caches,
            target.layer_caches,
            strict=True,
        ):
            if seq_len > 0:
                target_key[:, :, :seq_len, :].copy_(source_key[:, :, :seq_len, :])
                target_value[:, :, :seq_len, :].copy_(source_value[:, :, :seq_len, :])

    def ensure_decode_state_capacity(
        self,
        state: Any | None,
        *,
        required_seq_len: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        optimize: bool,
        bucket_resolver: Callable[[int], int] | None = None,
    ) -> Any:
        if not optimize:
            return self._ensure_dynamic_decode_state(
                state,
                batch_size=int(batch_size),
                device=device,
                dtype=dtype,
            )

        if (
            state is not None
            and getattr(state, "static", True)
            and self.decode_state_capacity(state) >= int(required_seq_len)
        ):
            return state

        requested = int(required_seq_len)
        if requested <= 0:
            raise ValueError("required_seq_len must be positive.")
        out_ds_rate = int(self.encoder.out_ds_rate)
        target_audio_patch_count = (requested + out_ds_rate - 1) // out_ds_rate
        if optimize and bucket_resolver is not None:
            target_audio_patch_count = int(bucket_resolver(target_audio_patch_count))
        next_state = self.init_decode_state(
            max_audio_patch_count=target_audio_patch_count,
            batch_size=int(batch_size),
            device=device,
            dtype=dtype,
        )
        if state is not None:
            self.copy_decode_state(state, next_state)
        return next_state

    def prefill_with_state(
        self,
        x: torch.Tensor,
        state: Any | None,
        *,
        optimize: bool,
        bucket_resolver: Callable[[int], int] | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, Any]:
        token_count = (x.size(1) // int(self.encoder.patch_size)) * int(
            self.encoder.out_ds_rate
        )
        required_seq_len = int(token_count) + (
            0 if state is None else int(state.seq_len)
        )
        state = self.ensure_decode_state_capacity(
            state,
            required_seq_len=max(1, required_seq_len),
            batch_size=x.size(0),
            device=x.device,
            dtype=x.dtype if dtype is None else dtype,
            optimize=bool(optimize),
            bucket_resolver=bucket_resolver,
        )
        return self.prefill(x, state)

    def _get_decode_program(
        self,
        state: Any,
        *,
        compile_step: bool,
    ) -> SemanticEncoderDecodeProgram | None:
        if not SemanticEncoderDecodeStep.supports(self.encoder):
            return None
        key_cache, _ = state.layer_caches[0]
        key = (
            int(key_cache.size(2)),
            int(self.encoder.out_ds_rate),
            str(key_cache.device),
            key_cache.dtype,
            bool(compile_step),
        )
        program = self._programs.get(key)
        if program is None:
            program = SemanticEncoderDecodeProgram(
                encoder=self.encoder,
                cache_capacity=int(key_cache.size(2)),
                block_len=int(self.encoder.out_ds_rate),
                device=key_cache.device,
                compile_step=bool(compile_step),
            )
            self._programs[key] = program
        return program

    def decode_patch(
        self,
        latent_patch: torch.Tensor,
        state: Any,
        *,
        optimize: bool,
    ) -> torch.Tensor:
        if not optimize and getattr(state, "static", True):
            dynamic_state = self._ensure_dynamic_decode_state(
                state,
                batch_size=latent_patch.size(0),
                device=latent_patch.device,
                dtype=latent_patch.dtype,
            )
            state.conv_tail = dynamic_state.conv_tail
            state.layer_caches = dynamic_state.layer_caches
            state.seq_len = dynamic_state.seq_len
            state.static = False
        start_pos = int(state.seq_len)
        compile_step = bool(optimize and latent_patch.device.type == "cuda")
        program = (
            self._get_decode_program(state, compile_step=compile_step)
            if optimize
            else None
        )
        if program is not None:
            embedding, conv_tail = program.decode_patch(
                latent_patch,
                state.conv_tail,
                state.layer_caches,
                start_pos=start_pos,
            )
        else:
            encoder = self.encoder
            self._require_decode_patch(
                encoder, latent_patch, "VAESemanticEncoder.decode_patch"
            )
            positions = (
                torch.arange(
                    int(encoder.out_ds_rate),
                    device=latent_patch.device,
                    dtype=torch.long,
                )
                + start_pos
            )
            step_inputs, conv_tail = SemanticEncoderDecodeStep.downsample(
                encoder, latent_patch, conv_tail=state.conv_tail
            )
            if step_inputs.size(1) != int(encoder.out_ds_rate):
                raise RuntimeError(
                    f"Downsample step produced {step_inputs.size(1)} tokens, expected {encoder.out_ds_rate}."
                )
            embedding = self._semantic_embeddings(
                step_inputs,
                state=state,
                positions=positions,
            )
        state.conv_tail.copy_(conv_tail)
        state.seq_len = start_pos + int(self.encoder.out_ds_rate)
        return embedding

    def decode_patch_with_state(
        self,
        latent_patch: torch.Tensor,
        state: Any | None,
        *,
        optimize: bool,
        bucket_resolver: Callable[[int], int] | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, Any]:
        required_seq_len = (0 if state is None else int(state.seq_len)) + int(
            self.encoder.out_ds_rate
        )
        state = self.ensure_decode_state_capacity(
            state,
            required_seq_len=required_seq_len,
            batch_size=latent_patch.size(0),
            device=latent_patch.device,
            dtype=latent_patch.dtype if dtype is None else dtype,
            optimize=bool(optimize),
            bucket_resolver=bucket_resolver,
        )
        return self.decode_patch(latent_patch, state, optimize=optimize), state
