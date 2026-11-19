from __future__ import annotations

import soundfile as sf
import torch

from ...utils.profiling import ensure_data_profiler
from ...data.pipelines.base import BaseSamplePipeline
from ...data.pipelines.preprocessing import (
    compute_num_audio_tokens,
    normalize_edge_silence_duration,
    pad_waveform_align_only,
)
from ...data.pipelines.tokenizing import build_tokenized_example
from ...modules.speaker.fbank import extract_speaker_fbank
from ...utils.audio import high_quality_resample

TTS_TEXT_PREFIX = "[文本]"
TTS_AUDIO_PREFIX = "[文本对应语音]"
TTS_INSTRUCTION_TEXT_PREFIX = "[带指令文本]"
TTA_TEXT_PREFIX = "[声音描述]"
TTA_AUDIO_PREFIX = "[描述对应声音]"
TTS_INTERLEAVE_PREFIX = "[流式语音合成]"
DEFAULT_TRAIN_TEMPLATE = f"{TTS_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
DEFAULT_INSTRUCTION_TTS_TEMPLATE = (
    f"{TTS_INSTRUCTION_TEXT_PREFIX}{{text}}{TTS_AUDIO_PREFIX}{{audio}}"
)
DEFAULT_TEXT_TO_AUDIO_TEMPLATE = f"{TTA_TEXT_PREFIX}{{text}}{TTA_AUDIO_PREFIX}{{audio}}"
DEFAULT_INTERLEAVE_TRAIN_TEMPLATE = f"{TTS_INTERLEAVE_PREFIX}{{interleave}}"


class BasicTtsPipeline(BaseSamplePipeline):
    """Fixed internal training pipeline for adapter-emitted samples."""

    template = DEFAULT_TRAIN_TEMPLATE

    def __init__(self, tokenizer, data_cfg, *, profiler=None):
        self.tokenizer = tokenizer
        self.train_audio_sample_rate = int(data_cfg.train_audio_sample_rate)
        self.audio_samples_per_llm_token = int(data_cfg.audio_samples_per_llm_token)
        self.profiler = ensure_data_profiler(profiler)

    @staticmethod
    def _load_waveform(audio_path: str) -> tuple[torch.Tensor, int]:
        if not isinstance(audio_path, str):
            raise TypeError(
                f"Training audio must be a filesystem path, got {type(audio_path)}."
            )
        audio_data, sample_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(audio_data.T)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.contiguous(), int(sample_rate)

    @staticmethod
    def _validate_source_sample(sample: dict) -> None:
        missing = [field for field in ("fid", "text", "audio") if field not in sample]
        if missing:
            raise ValueError(
                "Source adapter must emit fid/text/audio. "
                f"Missing fields: {missing}. Sample keys: {sorted(sample.keys())}"
            )

    def process_sample(self, raw_sample: dict) -> dict:
        sample = dict(raw_sample)
        self._validate_source_sample(sample)
        sample["fid"] = str(sample["fid"])

        with self.profiler.measure("worker.process_sample_total"):
            return self._process_sample_impl(sample)

    def _process_sample_impl(self, sample: dict) -> dict:
        profiler = self.profiler
        with profiler.measure("worker.load_audio"):
            waveform, sample_rate = self._load_waveform(sample["audio"])
        with profiler.measure("worker.resample_audio"):
            waveform = high_quality_resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=self.train_audio_sample_rate,
            )
        with profiler.measure("worker.normalize_edge_silence"):
            waveform = normalize_edge_silence_duration(
                waveform,
                sample_rate=self.train_audio_sample_rate,
            )
        sample["sample"] = waveform
        sample["sample_rate"] = self.train_audio_sample_rate
        sample["unpadded_sample_length"] = int(waveform.size(-1))

        with profiler.measure("worker.pad_audio"):
            waveform = pad_waveform_align_only(
                waveform,
                multiple_of=self.audio_samples_per_llm_token,
            )
        sample["sample"] = waveform
        sample["sample_length"] = int(waveform.size(-1))

        num_audio_tokens = compute_num_audio_tokens(
            sample["sample_length"],
            audio_samples_per_llm_token=self.audio_samples_per_llm_token,
        )
        with profiler.measure("worker.tokenize"):
            tokenized = build_tokenized_example(
                text=sample["text"],
                tokenizer=self.tokenizer,
                template=self.template,
                num_audio_tokens=num_audio_tokens,
            )
        sample["input_ids"] = tokenized["input_ids"]
        sample["labels"] = tokenized["labels"]
        sample["loss_mask"] = tokenized["loss_mask"]
        sample["input_ids_length"] = len(tokenized["input_ids"])
        sample["num_text_tokens"] = tokenized["text_token_count"]
        sample["num_audio_tokens"] = num_audio_tokens
        sample["num_total_tokens"] = sample["input_ids_length"]

        with profiler.measure("worker.extract_fbank"):
            fbank = extract_speaker_fbank(
                sample["sample"],
                sample_rate=sample["sample_rate"],
            )
        sample["fbank"] = fbank
        sample["fbank_length"] = int(fbank.size(0))
        return sample


class InterleaveTtsPipeline(BasicTtsPipeline):
    template = DEFAULT_INTERLEAVE_TRAIN_TEMPLATE
