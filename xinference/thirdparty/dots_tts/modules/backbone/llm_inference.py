from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from loguru import logger
from transformers import StaticCache

from ...modules.backbone.inference_utils import compile_bound_method
from ...utils.logging import categorized_log as logc


@dataclass
class LLMInferenceState:
    cache: Any | None = None
    seq_len: int = 0
    position_ids: torch.Tensor | None = None
    decode_position: torch.Tensor | None = None
    decode_inputs_embeds: torch.Tensor | None = None
    max_sequence_length: int | None = None


class LLMInference:
    def __init__(self, core: Any):
        self.core = core
        self._static_workspaces: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._compiled_static_steps: dict[tuple[Any, ...], Callable[..., Any]] = {}

    def clear(self) -> None:
        self._static_workspaces.clear()
        self._compiled_static_steps.clear()

    def init_state(
        self,
        *,
        optimize: bool,
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LLMInferenceState:
        if not optimize:
            return LLMInferenceState()

        workspace = self._get_static_workspace(
            max_sequence_length=int(max_sequence_length),
            device=device,
            dtype=dtype,
        )
        return LLMInferenceState(
            **workspace, max_sequence_length=int(max_sequence_length)
        )

    def _get_static_workspace(
        self,
        *,
        max_sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        workspace_key = (
            int(max_sequence_length),
            str(device),
            dtype,
            int(self.core.llm_hidden_size),
        )
        workspace = self._static_workspaces.get(workspace_key)
        if workspace is None:
            workspace = {
                "cache": StaticCache(
                    config=self.core.llm.config,
                    max_cache_len=int(max_sequence_length),
                ),
                "position_ids": torch.arange(
                    int(max_sequence_length),
                    dtype=torch.long,
                    device=device,
                ),
                "decode_position": torch.zeros(
                    (1,),
                    dtype=torch.long,
                    device=device,
                ),
                "decode_inputs_embeds": torch.zeros(
                    (1, 1, int(self.core.llm_hidden_size)),
                    dtype=dtype,
                    device=device,
                ),
            }
            self._static_workspaces[workspace_key] = workspace
            logger.debug(
                logc(
                    "cache",
                    "Prepared LLM StaticCache workspace: max_sequence_length={} device={} dtype={}",
                ),
                int(max_sequence_length),
                device,
                dtype,
            )
        else:
            workspace["cache"].reset()
            workspace["decode_position"].zero_()
            workspace["decode_inputs_embeds"].zero_()
        return workspace

    def _ensure_sequence_capacity(
        self,
        state: LLMInferenceState,
        *,
        input_len: int,
        max_sequence_length: int,
    ) -> None:
        required = int(state.seq_len) + int(input_len)
        if required > int(max_sequence_length):
            raise ValueError(
                f"LLM sequence length exceeds max_sequence_length: required={required} max_sequence_length={int(max_sequence_length)}."
            )

    def _run_step(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Any | None,
        *,
        request_logits: bool,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, Any | None]:
        outputs = self.core._llm_base_model()(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            cache_position=cache_position,
        )
        hidden = self.core._last_hidden_state(outputs)
        if hidden is None:
            raise RuntimeError("LLM base model did not return last_hidden_state.")
        logits = self.core._compute_lm_logits(hidden) if request_logits else None
        return hidden, logits, outputs.past_key_values

    @torch.no_grad()
    def _static_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Any,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        return self._run_step(
            inputs_embeds,
            past_key_values,
            request_logits=False,
            cache_position=cache_position,
        )[0]

    def _get_static_token_step(
        self,
        *,
        signature: tuple[Any, ...],
        compile_step: bool,
    ) -> Callable[..., Any]:
        if not compile_step:
            return self._static_inputs_embeds

        compiled = self._compiled_static_steps.get(signature)
        if compiled is None:
            compiled = compile_bound_method(self, "_static_inputs_embeds", unwrap=True)
            self._compiled_static_steps[signature] = compiled
            logger.debug(
                logc("compile", "Compiled LLM static decode step: signature={}"),
                signature,
            )
        return compiled

    def step(
        self,
        state: LLMInferenceState,
        *,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        request_logits: bool = False,
        compile_static: bool = False,
        optimize: bool,
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if (inputs_embeds is None) == (input_ids is None):
            raise ValueError(
                "Exactly one of inputs_embeds or input_ids must be provided to LLM step."
            )
        if inputs_embeds is None:
            inputs_embeds = self.core.llm.get_input_embeddings()(input_ids)
        input_len = int(inputs_embeds.size(1))
        if input_len <= 0:
            raise ValueError("LLM step input length must be positive.")

        if not (optimize and isinstance(state.cache, StaticCache)):
            hidden, logits, state.cache = self._run_step(
                inputs_embeds,
                state.cache,
                request_logits=request_logits,
            )
            state.seq_len += input_len
            return inputs_embeds, hidden, logits

        self._ensure_sequence_capacity(
            state,
            input_len=input_len,
            max_sequence_length=int(max_sequence_length),
        )
        if compile_static and not request_logits and state.cache.is_initialized:
            return self._step_static_tokens(
                state,
                inputs_embeds=inputs_embeds,
                max_sequence_length=int(max_sequence_length),
                optimize=bool(optimize),
            )

        if state.position_ids is None:
            raise RuntimeError("LLM StaticCache position workspace is not initialized.")
        start = int(state.seq_len)
        cache_position = state.position_ids[start : start + input_len]
        hidden, logits, state.cache = self._run_step(
            inputs_embeds,
            state.cache,
            request_logits=request_logits,
            cache_position=cache_position,
        )
        state.seq_len += input_len
        return inputs_embeds, hidden, logits

    def _step_static_tokens(
        self,
        state: LLMInferenceState,
        *,
        inputs_embeds: torch.Tensor,
        max_sequence_length: int,
        optimize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        token_count = int(inputs_embeds.size(1))
        if token_count <= 0:
            raise ValueError("LLM static token step input length must be positive.")
        if state.decode_position is None:
            raise RuntimeError("LLM StaticCache decode position is not initialized.")
        if state.decode_inputs_embeds is None:
            raise RuntimeError("LLM StaticCache decode embedding is not initialized.")
        if state.decode_inputs_embeds.dtype != inputs_embeds.dtype:
            raise RuntimeError(
                f"LLM StaticCache decode embedding dtype mismatch: workspace={state.decode_inputs_embeds.dtype} inputs={inputs_embeds.dtype}."
            )

        signature = (
            int(max_sequence_length),
            int(self.core.llm_hidden_size),
            inputs_embeds.dtype,
            str(inputs_embeds.device),
        )
        token_step = self._get_static_token_step(
            signature=signature,
            compile_step=bool(optimize and inputs_embeds.device.type == "cuda"),
        )
        self._ensure_sequence_capacity(
            state,
            input_len=token_count,
            max_sequence_length=int(max_sequence_length),
        )
        hidden_chunks: list[torch.Tensor] = []
        decode_inputs_embeds = state.decode_inputs_embeds
        for token_index in range(token_count):
            state.decode_position.fill_(int(state.seq_len))
            decode_inputs_embeds.copy_(
                inputs_embeds[:, token_index : token_index + 1, :]
            )
            hidden = token_step(
                decode_inputs_embeds,
                state.cache,
                state.decode_position,
            )
            hidden_chunks.append(hidden if token_count == 1 else hidden.clone())
            state.seq_len += 1

        return inputs_embeds, torch.cat(hidden_chunks, dim=1), None
