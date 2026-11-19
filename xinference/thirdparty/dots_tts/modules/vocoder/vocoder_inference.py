from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ...modules.backbone.inference_utils import compile_bound_method
from ...modules.vocoder.alias_free_act import Activation1d
from ...modules.vocoder.bigvgan import AMPBlock1, AMPBlock2, AudioVAE
from ...utils.logging import categorized_log as logc
from ...utils.profiling import measure_inference


@dataclass(slots=True)
class DecoderStreamState:
    window: torch.Tensor
    chunk_size: int
    total_frames: int = 0
    emitted_frames: int = 0


@dataclass(slots=True)
class VocoderStreamState:
    lstm_hidden: tuple[torch.Tensor, torch.Tensor]
    decoder: DecoderStreamState


class VocoderInference:
    def __init__(self, vocoder: nn.Module):
        self.vocoder = vocoder
        self._compiled_stream_steps: dict[int, Callable[..., Any]] = {}
        self._stream_window_sizes: dict[int, int] = {}
        self._lstm_num_layers = 0
        self._lstm_hidden_size = 0
        self._lstm_skip = False
        self._lstm_weight_ih: tuple[torch.Tensor, ...] = ()
        self._lstm_weight_hh: tuple[torch.Tensor, ...] = ()
        self._lstm_bias_ih: tuple[torch.Tensor, ...] = ()
        self._lstm_bias_hh: tuple[torch.Tensor, ...] = ()

    def clear(self) -> None:
        self._compiled_stream_steps.clear()

    @torch.no_grad()
    @torch.autocast(enabled=False, device_type="cuda")
    def extract_latents(
        self,
        x: torch.Tensor,
        *,
        do_sample: bool = False,
    ) -> torch.Tensor:
        if not isinstance(self.vocoder, AudioVAE):
            raise TypeError(
                f"Unsupported vocoder for latent extraction: {type(self.vocoder)!r}."
            )
        x = x.float()
        x = self.vocoder.audio_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.vocoder.enc_mi_layer(x)
        x = x.permute(0, 2, 1)
        x = self.vocoder.pre_proj(x)
        if do_sample:
            m_q, logs_q = torch.split(x, self.vocoder.h.latent_dim, dim=1)
            x = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        return x

    @torch.no_grad()
    @torch.autocast(enabled=False, device_type="cuda")
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        with measure_inference("latent_decoder", phase="batch_decode"):
            x = latents.transpose(1, 2).float()
            if not isinstance(self.vocoder, AudioVAE):
                raise TypeError(
                    f"Unsupported vocoder for latent decoding: {type(self.vocoder)!r}."
                )
            if x.size(1) != int(self.vocoder.h.latent_dim):
                raise ValueError(
                    f"AudioVAE latent channel count must be {self.vocoder.h.latent_dim}, got {x.size(1)}."
                )
            x = self.vocoder.post_proj(x)
            x = x.permute(0, 2, 1)
            x = self.vocoder.dec_mi_layer(x)
            x = x.permute(0, 2, 1)
            return self.vocoder.decoder(x)

    def init_stream_state(
        self,
        *,
        batch_size: int = 1,
        chunk_size: int,
    ) -> VocoderStreamState:
        if not isinstance(self.vocoder, AudioVAE) or not self.vocoder.h.causal:
            raise RuntimeError("Strict streaming requires a causal AudioVAE vocoder.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        self._prepare_lstm_stream_params()
        window_size = self._stream_window_size(int(chunk_size))
        conv_pre_weight = next(self.vocoder.decoder.conv_pre.parameters())
        return VocoderStreamState(
            lstm_hidden=self._init_lstm_stream_state(int(batch_size)),
            decoder=DecoderStreamState(
                window=conv_pre_weight.new_zeros(
                    (
                        int(batch_size),
                        int(self.vocoder.h.latent_dim),
                        max(0, window_size),
                    )
                ),
                chunk_size=int(chunk_size),
            ),
        )

    @torch.no_grad()
    @torch.autocast(enabled=False, device_type="cuda")
    def stream_step(
        self,
        latent_patch: torch.Tensor,
        stream_state: VocoderStreamState,
        *,
        optimize: bool,
        profile_step: int | str | None = None,
        use_compiled: bool = True,
    ) -> torch.Tensor:
        latents = latent_patch.transpose(1, 2)
        if not optimize or not use_compiled:
            with measure_inference("vocoder", phase="stream", step=profile_step):
                return self._stream_step_eager(latents, stream_state)

        valid_frames = min(
            stream_state.decoder.total_frames,
            stream_state.decoder.window.size(-1),
        )
        valid_frames_tensor = stream_state.decoder.window.new_tensor(
            valid_frames,
            dtype=torch.int64,
        )
        stream_step = self._get_compiled_stream_step(int(latents.size(-1)))
        with measure_inference("vocoder", phase="stream", step=profile_step):
            audio_window, hidden_h, hidden_c, new_window = stream_step(
                latents,
                stream_state.lstm_hidden[0],
                stream_state.lstm_hidden[1],
                stream_state.decoder.window,
                valid_frames_tensor,
            )
        stream_state.lstm_hidden = (hidden_h.clone(), hidden_c.clone())
        stream_state.decoder.window = new_window.clone()
        stream_state.decoder.total_frames += int(latents.size(-1))
        audio_chunk = self._slice_stream_audio_window(
            audio_window,
            stream_state,
            final=False,
        )
        return audio_chunk.clone()

    @torch.no_grad()
    @torch.autocast(enabled=False, device_type="cuda")
    def flush(self, stream_state: VocoderStreamState) -> torch.Tensor:
        with measure_inference("vocoder", phase="stream_flush"):
            audio_window = self._decode_stream_window(stream_state.decoder.window)
            return self._slice_stream_audio_window(
                audio_window, stream_state, final=True
            )

    def _get_compiled_stream_step(self, chunk_frames: int) -> Callable[..., Any]:
        chunk_frames = int(chunk_frames)
        if chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive, got {chunk_frames}.")
        stream_step = self._compiled_stream_steps.get(chunk_frames)
        if stream_step is None:
            stream_step = compile_bound_method(
                self,
                "_compiled_stream_step",
                unwrap=True,
            )
            self._compiled_stream_steps[chunk_frames] = stream_step
            logger.debug(
                logc("compile", "Compiled vocoder stream step: chunk_frames={}"),
                chunk_frames,
            )
        return stream_step

    @torch.no_grad()
    def _compiled_stream_step(
        self,
        latents: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        window: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder_input, (hidden_h, hidden_c) = self._decode_stream_latents(
            latents,
            (hidden_h, hidden_c),
        )
        new_window = self._append_stream_decoder_input_tensor(
            decoder_input,
            window,
            valid_frames,
        )
        audio_window = self._decode_stream_window(new_window)
        return audio_window, hidden_h, hidden_c, new_window

    def _stream_step_eager(
        self,
        latents: torch.Tensor,
        stream_state: VocoderStreamState,
    ) -> torch.Tensor:
        decoder_input, stream_state.lstm_hidden = self._decode_stream_latents(
            latents,
            stream_state.lstm_hidden,
        )
        decoder_state = stream_state.decoder
        decoder_state.window = self._append_stream_decoder_input_tensor(
            decoder_input,
            decoder_state.window,
            decoder_state.window.new_tensor(
                min(decoder_state.total_frames, decoder_state.window.size(-1)),
                dtype=torch.int64,
            ),
        )
        decoder_state.total_frames += int(latents.size(-1))
        audio_window = self._decode_stream_window(decoder_state.window)
        return self._slice_stream_audio_window(
            audio_window,
            stream_state,
            final=False,
        )

    def _validate_stream_latents(self, latents: torch.Tensor) -> None:
        latent_dim = int(self.vocoder.h.latent_dim)
        if latents.ndim != 3:
            raise ValueError(
                "Streaming latents must have shape [batch, latent_dim, frames], "
                f"got {tuple(latents.shape)}."
            )
        if latents.size(1) != latent_dim:
            raise ValueError(
                f"Streaming latent_dim must be {latent_dim}, got {latents.size(1)}."
            )

    def _decode_stream_latents(
        self,
        latents: torch.Tensor,
        lstm_hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        self._validate_stream_latents(latents)
        vocoder = self.vocoder
        x = vocoder.post_proj(latents.float())
        x = x.permute(0, 2, 1)
        x = vocoder.dec_mi_layer[0](x)
        x, lstm_hidden = self._lstm_stream_step(x, lstm_hidden)
        x = vocoder.dec_mi_layer[2](x)
        decoder_dtype = next(vocoder.decoder.conv_pre.parameters()).dtype
        return x.permute(0, 2, 1).to(dtype=decoder_dtype), lstm_hidden

    def _prepare_lstm_stream_params(self) -> None:
        lstm = self.vocoder.dec_mi_layer[1]
        if lstm.bidirectional:
            raise RuntimeError("Streaming only supports unidirectional SLSTM.")
        num_layers = int(lstm.lstm.num_layers)
        self._lstm_num_layers = num_layers
        self._lstm_hidden_size = int(lstm.lstm.hidden_size)
        self._lstm_skip = bool(lstm.skip)
        self._lstm_weight_ih = tuple(
            getattr(lstm.lstm, f"weight_ih_l{layer_idx}")
            for layer_idx in range(num_layers)
        )
        self._lstm_weight_hh = tuple(
            getattr(lstm.lstm, f"weight_hh_l{layer_idx}")
            for layer_idx in range(num_layers)
        )
        self._lstm_bias_ih = tuple(
            getattr(lstm.lstm, f"bias_ih_l{layer_idx}")
            for layer_idx in range(num_layers)
        )
        self._lstm_bias_hh = tuple(
            getattr(lstm.lstm, f"bias_hh_l{layer_idx}")
            for layer_idx in range(num_layers)
        )

    def _init_lstm_stream_state(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_shape = (
            self._lstm_num_layers,
            int(batch_size),
            self._lstm_hidden_size,
        )
        weight = self._lstm_weight_hh[0]
        return (
            weight.new_zeros(state_shape),
            weight.new_zeros(state_shape),
        )

    def _lstm_stream_step(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if hidden is None:
            hidden = self._init_lstm_stream_state(x.size(0))

        residual = x
        hidden_h, hidden_c = hidden
        next_hidden_h = []
        next_hidden_c = []
        for layer_idx in range(self._lstm_num_layers):
            layer_input = x
            hx = hidden_h[layer_idx]
            cx = hidden_c[layer_idx]
            outputs = []
            weight_ih = self._lstm_weight_ih[layer_idx]
            weight_hh = self._lstm_weight_hh[layer_idx]
            bias_ih = self._lstm_bias_ih[layer_idx]
            bias_hh = self._lstm_bias_hh[layer_idx]

            for frame_idx in range(x.size(1)):
                gates = F.linear(layer_input[:, frame_idx, :], weight_ih, bias_ih)
                gates = gates + F.linear(hx, weight_hh, bias_hh)
                input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=-1)
                input_gate = torch.sigmoid(input_gate)
                forget_gate = torch.sigmoid(forget_gate)
                cell_gate = torch.tanh(cell_gate)
                output_gate = torch.sigmoid(output_gate)
                cx = forget_gate * cx + input_gate * cell_gate
                hx = output_gate * torch.tanh(cx)
                outputs.append(hx)

            x = torch.stack(outputs, dim=1)
            next_hidden_h.append(hx)
            next_hidden_c.append(cx)

        if self._lstm_skip:
            x = x + residual
        return x, (torch.stack(next_hidden_h, dim=0), torch.stack(next_hidden_c, dim=0))

    def _append_stream_decoder_input_tensor(
        self,
        decoder_input: torch.Tensor,
        window: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> torch.Tensor:
        if window.dtype != decoder_input.dtype:
            window = window.to(dtype=decoder_input.dtype)
        chunk_size = int(decoder_input.size(-1))
        if chunk_size >= window.size(-1):
            raise ValueError(
                f"decoder window size {window.size(-1)} must be larger than chunk_size {chunk_size}."
            )
        positions = torch.arange(
            window.size(-1),
            device=window.device,
            dtype=valid_frames.dtype,
        )
        clipped_valid = valid_frames.clamp(min=0, max=window.size(-1))
        combined = torch.cat(
            [
                window,
                decoder_input.new_zeros(window.size(0), window.size(1), chunk_size),
            ],
            dim=-1,
        )
        insert_index = clipped_valid + torch.arange(
            chunk_size,
            device=window.device,
            dtype=valid_frames.dtype,
        )
        combined.scatter_(
            -1,
            insert_index.view(1, 1, -1).expand_as(decoder_input),
            decoder_input,
        )
        new_valid = (clipped_valid + chunk_size).clamp(max=window.size(-1))
        start = (clipped_valid + chunk_size - window.size(-1)).clamp(min=0)
        gather_index = (start + positions).clamp(max=combined.size(-1) - 1)
        gathered = combined.gather(
            -1,
            gather_index.view(1, 1, -1).expand_as(window),
        )
        mask = (positions < new_valid).to(dtype=window.dtype).view(1, 1, -1)
        return gathered * mask

    def _decode_stream_window(self, window: torch.Tensor) -> torch.Tensor:
        decoder_dtype = next(self.vocoder.decoder.conv_pre.parameters()).dtype
        if window.dtype != decoder_dtype:
            window = window.to(dtype=decoder_dtype)
        return self.vocoder.decoder(window)

    def _slice_stream_audio_window(
        self,
        audio_window: torch.Tensor,
        stream_state: VocoderStreamState,
        *,
        final: bool,
    ) -> torch.Tensor:
        decoder_state = stream_state.decoder
        stable_end = (
            decoder_state.total_frames
            if final
            else max(0, decoder_state.total_frames - self._decoder_stream_lookahead())
        )
        if stable_end <= decoder_state.emitted_frames:
            return audio_window.new_zeros((audio_window.size(0), 1, 0))

        window_size = decoder_state.window.size(-1)
        valid_frames = min(decoder_state.total_frames, window_size)
        window_start = decoder_state.total_frames - valid_frames
        if decoder_state.emitted_frames < window_start:
            raise RuntimeError(
                "Decoder stream window is too short for fixed-graph decoding."
            )

        local_start = decoder_state.emitted_frames - window_start
        local_end = stable_end - window_start
        sample_start = local_start * int(self.vocoder.hop_size)
        sample_end = local_end * int(self.vocoder.hop_size)
        decoder_state.emitted_frames = stable_end
        return audio_window[..., sample_start:sample_end]

    def _stream_window_size(self, chunk_size: int) -> int:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        cached = self._stream_window_sizes.get(chunk_size)
        if cached is not None:
            return cached

        window_size = (
            int(chunk_size)
            + self._decoder_stream_lookahead()
            + self._decoder_stream_left_context()
        )
        self._stream_window_sizes[chunk_size] = window_size
        return window_size

    def _decoder_stream_lookahead(self) -> int:
        conv_pre = self.vocoder.decoder.conv_pre
        return int((conv_pre.kernel_size[0] - 1) // 2)

    @staticmethod
    def _conv1d_left_context(layer: nn.Module) -> int:
        dilation = (
            layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation
        )
        kernel_size = (
            layer.kernel_size[0]
            if isinstance(layer.kernel_size, tuple)
            else layer.kernel_size
        )
        if layer.causal:
            return int(dilation) * (int(kernel_size) - 1)
        return (
            layer.padding[0] if isinstance(layer.padding, tuple) else int(layer.padding)
        )

    @staticmethod
    def _convtranspose1d_left_context(layer: nn.Module) -> int:
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
        kernel_size = (
            layer.kernel_size[0]
            if isinstance(layer.kernel_size, tuple)
            else layer.kernel_size
        )
        if not layer.causal:
            raise NotImplementedError("Streaming only supports causal ConvTranspose1d.")
        if int(kernel_size) != 2 * int(stride):
            raise ValueError(
                "Streaming ConvTranspose1d expects kernel_size == 2 * stride, got "
                f"kernel_size={kernel_size} stride={stride}."
            )
        return 1

    @classmethod
    def _activation_left_context(cls, activation: Activation1d) -> int:
        upsample = activation.upsample
        downsample = activation.downsample.lowpass
        if not upsample.causal or not downsample.padding or downsample.pad_right != 0:
            raise NotImplementedError(
                "Streaming only supports causal alias-free activations."
            )
        ratio = int(upsample.ratio)
        if ratio != int(downsample.stride):
            raise ValueError(
                "Alias-free activation expects matched up/down ratios, got "
                f"up_ratio={ratio} down_ratio={downsample.stride}."
            )
        total_left = (upsample.kernel_size - 1) + (downsample.kernel_size - 1)
        return (total_left + ratio - 1) // ratio

    @classmethod
    def _ampblock_left_context(cls, block: nn.Module) -> int:
        if isinstance(block, AMPBlock1):
            left_context = 0
            acts1 = block.activations[::2]
            acts2 = block.activations[1::2]
            for conv1, conv2, act1, act2 in zip(
                block.convs1,
                block.convs2,
                acts1,
                acts2,
                strict=True,
            ):
                left_context += (
                    cls._activation_left_context(act1)
                    + cls._conv1d_left_context(conv1)
                    + cls._activation_left_context(act2)
                    + cls._conv1d_left_context(conv2)
                )
            return left_context
        if isinstance(block, AMPBlock2):
            left_context = 0
            for conv, activation in zip(block.convs, block.activations, strict=True):
                left_context += cls._activation_left_context(
                    activation
                ) + cls._conv1d_left_context(conv)
            return left_context
        raise TypeError(f"Unsupported resblock type: {type(block).__name__}.")

    def _decoder_stream_left_context(self) -> int:
        decoder = self.vocoder.decoder
        left_context = Fraction(self._conv1d_left_context(decoder.conv_pre), 1)
        current_scale = Fraction(1, 1)
        for stage_idx, upsample_layers in enumerate(decoder.ups):
            for upsample in upsample_layers:
                left_context += current_scale * self._convtranspose1d_left_context(
                    upsample
                )
                stride = (
                    upsample.stride[0]
                    if isinstance(upsample.stride, tuple)
                    else upsample.stride
                )
                current_scale /= int(stride)

            stage_start = stage_idx * int(decoder.num_kernels)
            stage_end = stage_start + int(decoder.num_kernels)
            stage_context = max(
                self._ampblock_left_context(block)
                for block in decoder.resblocks[stage_start:stage_end]
            )
            left_context += current_scale * stage_context

        left_context += current_scale * self._activation_left_context(
            decoder.activation_post
        )
        left_context += current_scale * self._conv1d_left_context(decoder.conv_post)
        return int(left_context.__ceil__())
