# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Qwen3-TTS CustomVoice with native asynchronous, batched audio generation."""

from __future__ import annotations

import asyncio
import copy
import io
import json
from contextlib import aclosing
from pathlib import Path

import numpy as np

from .native_async import NativeAsyncAudioModel


class VLLMOmniTTSModel(NativeAsyncAudioModel):
    async def _load(self):
        from transformers import AutoTokenizer
        from vllm_omni import AsyncOmni
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import (
            Qwen3TTSConfig,
        )
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
            Qwen3TTSPromptEmbedsBuilder,
        )

        options = dict(self._kwargs)
        if options.get("async_chunk") is False:
            raise ValueError("Streaming Qwen3-TTS requires async_chunk=True")
        options.setdefault("async_chunk", True)
        options.setdefault("max_num_seqs", 16)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        self._config = Qwen3TTSConfig.from_pretrained(self._model_path).talker_config
        codec = json.loads(
            (Path(self._model_path) / "speech_tokenizer/config.json").read_text()
        )
        self._samples_per_token = codec["decode_upsample_rate"]
        if type(self._samples_per_token) is not int or self._samples_per_token <= 0:
            raise ValueError("Invalid speech tokenizer decode_upsample_rate")
        self._estimate_prompt = (
            Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information
        )
        self._engine = AsyncOmni(model=self._model_path, **options)

    def get_speech_capabilities(self):
        if self._engine is None or self._stopping:
            return {}
        return {
            "streaming": True,
            "formats": ["pcm"],
            "file_formats": ["pcm", "wav", "mp3", "flac", "ogg"],
            "sample_rates": [24000],
            "voices": [
                {"voice_id": name, "name": name} for name in self._config.spk_id
            ],
        }

    def speech(self, *args, **kwargs):
        raise RuntimeError("Use the asynchronous speech entrypoint")

    def _prompt(self, text, voice, language, instruct, max_tokens):
        if (voice is not None and not isinstance(voice, str)) or (
            language is not None and not isinstance(language, str)
        ):
            raise ValueError("voice and language must be strings")
        speaker = voice or "ryan"
        speakers = self._config.spk_id
        if speaker.lower() not in {name.lower() for name in speakers}:
            raise ValueError(f"Unknown speaker: {speaker}")
        language = language or "English"
        if language.lower() != "auto" and language.lower() not in {
            name.lower() for name in self._config.codec_language_id
        }:
            raise ValueError(f"Unsupported language: {language}")
        info = {
            "task_type": ["CustomVoice"],
            "text": [text],
            "speaker": [speaker],
            "language": [language],
            "instruct": [instruct],
            "max_new_tokens": [max_tokens],
        }
        length = self._estimate_prompt(
            additional_information=info,
            task_type="CustomVoice",
            tokenize_prompt=lambda value: self._tokenizer(value, padding=False)[
                "input_ids"
            ],
            codec_language_id=self._config.codec_language_id,
            spk_is_dialect=self._config.spk_is_dialect,
            estimate_ref_code_len=lambda _: None,
        )
        return {"prompt_token_ids": [1] * length, "additional_information": info}

    async def async_speech(
        self,
        input: str,
        voice: str = "",
        response_format="mp3",
        speed=1.0,
        stream=False,
        **kwargs,
    ):
        if not isinstance(input, str) or not input.strip() or len(input) > 5000:
            raise ValueError("Speech input must contain 1..5000 characters")
        if speed != 1.0:
            raise ValueError("Native Qwen3-TTS currently requires speed=1")
        if response_format not in {"pcm", "wav", "mp3", "flac", "ogg"}:
            raise ValueError("Unsupported speech format")
        if stream and response_format != "pcm":
            raise ValueError("Streamed Qwen3-TTS requires response_format=pcm")
        language = kwargs.pop("language", None)
        instruct = kwargs.pop("instruct", "")
        seed = kwargs.pop("seed", None)
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32)")
        max_tokens = kwargs.pop("max_new_tokens", 2048)
        if kwargs:
            raise ValueError(f"Unsupported speech options: {sorted(kwargs)}")
        if not isinstance(instruct, str) or len(instruct) > 500:
            raise ValueError("instruct must be a string of at most 500 characters")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
            raise ValueError("max_new_tokens must be 1..4096")
        prompt = await asyncio.to_thread(
            self._prompt, input, voice, language, instruct, max_tokens
        )
        params = copy.deepcopy(self._engine.default_sampling_params_list)
        from vllm.sampling_params import RequestOutputKind

        for item in params:
            item.output_kind = RequestOutputKind.DELTA
        params[0].max_tokens = max_tokens
        params[0].min_tokens = min(getattr(params[0], "min_tokens", 0), max_tokens)
        if seed is not None:
            params[0].seed = seed
        chunks = self._audio_chunks(prompt, params)
        if stream:
            return chunks
        async with aclosing(chunks):
            audio = [chunk async for chunk in chunks]
        pcm = b"".join(audio)
        if response_format == "pcm":
            return pcm

        def encode():
            import soundfile as sf

            output = io.BytesIO()
            sf.write(
                output, np.frombuffer(pcm, dtype="<i2"), 24000, format=response_format
            )
            return output.getvalue()

        return await asyncio.to_thread(encode)

    async def _audio_chunks(self, prompt, params):
        any_audio = False
        samples = 0
        async with aclosing(
            self._generate(
                prompt, sampling_params_list=params, output_modalities=["audio"]
            )
        ) as outputs:
            async for result in outputs:
                output = getattr(result, "request_output", result)
                if any(
                    getattr(item, "finish_reason", None) in {"length", "abort"}
                    for item in getattr(output, "outputs", [])
                ):
                    raise RuntimeError(
                        "Speech generation did not complete; output is incomplete"
                    )
                audio = getattr(output, "multimodal_output", None)
                if not audio:
                    for item in getattr(output, "outputs", []):
                        audio = getattr(item, "multimodal_output", None)
                        if audio:
                            break
                if not audio:
                    continue
                key = "audio" if "audio" in audio else "model_outputs"
                if key not in audio:
                    continue
                rate = audio.get("sr", 24000)
                if isinstance(rate, list):
                    rate = rate[-1]
                if int(rate) != 24000:
                    raise RuntimeError("Qwen3-TTS returned an unexpected sample rate")
                values = audio[key]
                # vllm-omni 0.32 drains modality data after each DELTA emission
                # (outputs/output_processor.py); lists contain only new chunks.
                chunks = values if isinstance(values, list) else [values]
                for chunk in chunks:
                    if hasattr(chunk, "detach"):
                        chunk = chunk.detach().float().cpu().numpy()
                    chunk = np.asarray(chunk)
                    if (
                        chunk.ndim > 2
                        or (chunk.ndim == 2 and 1 not in chunk.shape)
                        or not np.isfinite(chunk).all()
                    ):
                        raise RuntimeError("Qwen3-TTS returned invalid mono audio")
                    chunk = chunk.reshape(-1)
                    if chunk.size:
                        samples += chunk.size
                        # AsyncOmni 0.32 may report decoder 'stop' when the
                        # talker hit its budget. Each codec frame consumes one
                        # talker token; normal EOS consumes a token without audio.
                        if samples >= params[0].max_tokens * self._samples_per_token:
                            raise RuntimeError(
                                "Speech token budget exhausted; output is incomplete"
                            )
                        any_audio = True
                        yield (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()
        if not any_audio:
            raise RuntimeError("Qwen3-TTS returned no audio")
