# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from typing import Any, ClassVar

from sglang_omni.config import (
    ExecutorConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni.config.schema import StreamTargetConfig

_PKG = "xinference.thirdparty.audio8_tts.pipeline"


class Audio8TTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "ArkttsModel"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_preprocessing_executor",
                args={"device": "cuda:0"},
            ),
            get_next=f"{_PKG}.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name="tts_engine",
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
                args={"device": "cuda:0", "max_new_tokens": 1024},
            ),
            get_next=f"{_PKG}.next_stage.tts_engine_next",
            relay=RelayConfig(device="cuda"),
            stream_to=[StreamTargetConfig(to_stage="vocoder")],
        ),
        StageConfig(
            name="vocoder",
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_vocoder_executor",
                args={"device": "cuda:0"},
            ),
            get_next=f"{_PKG}.next_stage.vocoder_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        if os.getenv("AUDIO8_TTS_STREAM_ENABLED", "0") != "1":
            # Default to non-streaming: do not wire per-step vocoder queues.
            for stage in self.stages:
                stage.stream_to = []


EntryClass = Audio8TTSPipelineConfig
