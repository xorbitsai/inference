# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import importlib.util
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote_to_bytes

import torch

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from xinference.thirdparty.audio8_tts.factory import create_audio8_engine, make_server_args
from xinference.thirdparty.audio8_tts.io import Audio8TTSState
from xinference.thirdparty.audio8_tts.pipeline.engine_io import (
    apply_tts_result,
    build_tts_request,
)
from xinference.thirdparty.audio8_tts.pipeline.state_io import load_state, store_state
from xinference.thirdparty.audio8_tts.pipeline.streaming_vocoder import (
    Audio8StreamingVocoderExecutor,
)
from xinference.thirdparty.audio8_tts.tokenizer import Audio8TokenizerAdapter, Reference
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


def _load_config(model_path: str) -> SimpleNamespace:
    with open(Path(model_path) / "config.json", encoding="utf-8") as handle:
        return SimpleNamespace(**json.load(handle))


def _load_codec(model_path: str, device: str) -> Any:
    model_dir = Path(model_path)
    module_path = model_dir / "modeling_arktts_codec.py"
    spec = importlib.util.spec_from_file_location("_audio8_arktts_codec", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Audio8 codec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    codec = module.ArkttsCodec(_load_config(model_path))
    state = torch.load(
        model_dir / "codec.pth",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator." in key for key in state):
        state = {
            key.replace("generator.", ""): value
            for key, value in state.items()
            if "generator." in key
        }
    state = {
        key: value
        for key, value in state.items()
        if not key.endswith(("freqs_cis", "causal_mask"))
    }
    codec.load_state_dict(state, strict=True, assign=True)
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return codec.eval().to(device=device, dtype=dtype)


def _load_audio(audio_path: str, sample_rate: int) -> torch.Tensor:
    import httpx

    from sglang_omni.preprocessing.audio import AudioMediaIO

    media_io = AudioMediaIO(target_sr=sample_rate)
    if audio_path.startswith("data:"):
        header, payload = audio_path.split(",", 1)
        raw = base64.b64decode(payload) if ";base64" in header else unquote_to_bytes(payload)
        audio, _ = media_io.load_bytes(raw)
    elif audio_path.startswith(("http://", "https://")):
        response = httpx.get(audio_path, follow_redirects=True, timeout=30)
        response.raise_for_status()
        audio, _ = media_io.load_bytes(response.content)
    else:
        audio, _ = media_io.load_file(Path(audio_path))
    value = torch.from_numpy(audio).float()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    elif value.ndim == 2 and value.shape[0] > 1:
        value = value.mean(dim=0, keepdim=True)
    return value


def _encode_reference(codec: Any, audio_path: str, device: str) -> torch.Tensor:
    audio = _load_audio(audio_path, codec.sample_rate).to(
        device=device,
        dtype=next(codec.parameters()).dtype,
    )
    lengths = torch.tensor([audio.shape[-1]], device=device, dtype=torch.long)
    with torch.inference_mode():
        codes, code_lengths = codec.encode(audio, lengths)
    return codes[0, :, : int(code_lengths[0].item())].cpu()


def _validate_codes(value: Any, num_codebooks: int, codebook_size: int) -> torch.Tensor:
    codes = torch.as_tensor(value, dtype=torch.long, device="cpu")
    if codes.ndim != 2 or codes.shape[0] != num_codebooks or codes.shape[1] == 0:
        raise ValueError(
            f"reference codes must have shape [{num_codebooks}, T>0], got {tuple(codes.shape)}"
        )
    if int(codes.min().item()) < 0 or int(codes.max().item()) >= codebook_size:
        raise ValueError(f"reference codes must be in [0, {codebook_size - 1}]")
    return codes


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
) -> PreprocessingExecutor:
    from transformers import PreTrainedTokenizerFast

    config = _load_config(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        fix_mistral_regex=False,
    )
    adapter = Audio8TokenizerAdapter(
        tokenizer,
        semantic_begin_id=config.semantic_begin_id,
        semantic_end_id=config.semantic_end_id,
        eos_token_id=config.eos_token_id,
    )
    codec: Any | None = None

    def get_codec() -> Any:
        nonlocal codec
        if codec is None:
            logger.info("Loading Audio8 reference codec on %s", device)
            codec = _load_codec(model_path, device)
        return codec

    def preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        metadata = payload.request.metadata or {}
        tts_params = metadata.get("tts_params") or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}
        references: list[Reference] | None = None
        raw_references = inputs.get("references")
        if raw_references:
            references = []
            for raw in raw_references:
                codes = raw.get("vq_codes")
                if codes is None:
                    audio_path = raw.get("audio_path")
                    if not audio_path:
                        raise ValueError("reference requires audio_path or vq_codes")
                    codes = _encode_reference(get_codec(), audio_path, device)
                codes = _validate_codes(codes, config.num_codebooks, config.codebook_size)
                references.append(Reference(text=raw.get("text", ""), vq_codes=codes))
        prompt = adapter.build_prompt(
            inputs.get("text", ""),
            references,
            num_codebooks=config.num_codebooks,
        )
        state = Audio8TTSState(
            input_ids=prompt["input_ids"],
            vq_mask_tokens=prompt["vq_mask_tokens"],
            vq_parts=prompt["vq_parts"],
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
            max_new_tokens=int(params.get("max_new_tokens", 1024)),
            temperature=float(params.get("temperature", 0.8)),
            top_p=float(params.get("top_p", 0.95)),
            top_k=int(params.get("top_k", 50)),
            do_sample=float(params.get("temperature", 0.8)) > 0,
            sample_rate=config.codec_sample_rate,
            response_format=str(tts_params.get("response_format", "wav")).lower(),
        )
        return store_state(payload, state)

    return PreprocessingExecutor(preprocess)


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int = 1024,
    stream_enabled: bool | None = None,
) -> EngineExecutor:
    from transformers import PreTrainedTokenizerFast

    config = _load_config(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_path,
        fix_mistral_regex=False,
    )
    server_args = make_server_args(model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    enqueue_fn_holder: dict[str, Any] = {}
    codes_only_ids: set[str] = set()
    if stream_enabled is None:
        stream_enabled = os.getenv("AUDIO8_TTS_STREAM_ENABLED", "0") == "1"

    def enqueue_stream(request_id: str, codes: torch.Tensor) -> None:
        if not stream_enabled:
            return
        if request_id in codes_only_ids:
            return
        enqueue_fn = enqueue_fn_holder.get("fn")
        if enqueue_fn is not None:
            # The model overwrites its persistent output buffer on the next
            # decode step, so the vocoder must receive its own copy.
            enqueue_fn(request_id, codes.clone(), target_stage="vocoder")

    engine = create_audio8_engine(
        server_args,
        gpu_id=gpu_id,
        eos_token_id=config.eos_token_id,
        max_new_tokens=max_new_tokens,
        stream_fn=enqueue_stream,
    )

    def request_builder(payload: StagePayload):
        state = load_state(payload)
        if state.response_format in {"codes", "codec", "npy"}:
            codes_only_ids.add(payload.request_id)
        return build_tts_request(state, tokenizer, payload.request_id)

    def result_builder(payload: StagePayload, result: Any) -> StagePayload:
        state = load_state(payload)
        apply_tts_result(state, result)
        codes_only_ids.discard(payload.request_id)
        return store_state(payload, state)

    executor_kwargs: dict[str, Any] = {}
    if "stream_enabled" in inspect.signature(EngineExecutor.__init__).parameters:
        executor_kwargs["stream_enabled"] = stream_enabled
    executor = EngineExecutor(
        engine=engine,
        request_builder=request_builder,
        result_builder=result_builder,
        **executor_kwargs,
    )

    def set_stream_fn(stream_fn: Any) -> None:
        enqueue_fn_holder["fn"] = stream_fn

    executor.set_stream_fn = set_stream_fn
    return executor


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
) -> Audio8StreamingVocoderExecutor:
    codec = _load_codec(model_path, device)
    config = _load_config(model_path)
    warmup_codes = torch.zeros(
        1,
        config.num_codebooks,
        4,
        device=device,
        dtype=torch.long,
    )
    with torch.inference_mode():
        codec.decode(warmup_codes)
    return Audio8StreamingVocoderExecutor(
        codec,
        device=device,
        eos_token_id=config.eos_token_id,
        num_codebooks=config.num_codebooks,
        chunk_frames=int(os.getenv("AUDIO8_TTS_STREAM_CHUNK_FRAMES", "12")),
        context_frames=int(os.getenv("AUDIO8_TTS_STREAM_CONTEXT_FRAMES", "128")),
        guard_frames=int(os.getenv("AUDIO8_TTS_STREAM_GUARD_FRAMES", "1")),
        hop_length=config.codec_frame_size,
    )
