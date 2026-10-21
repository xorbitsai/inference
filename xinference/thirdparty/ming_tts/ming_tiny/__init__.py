"""Ming-3.0-tts-tiny (7.9B total, 1.3B active) inference.

Kept separate from the 0.5b model on purpose: the two share no tokenizer, no
prompt layout, and no weight-tying convention, so they are two independent
stacks that happen to live in one repository.
"""

from typing import Any

__all__ = [
    "MingTinyTTS",
    "MingTinyEngine",
    "LingV3TTSConfig",
    "LingV3TTSForConditionalGeneration",
    "build_control_instruction",
]


def __getattr__(name: str) -> Any:
    # Imported lazily so `import ming_tiny` stays cheap and does not
    # initialize CUDA in a CPU-only shell.
    if name == "MingTinyTTS":
        from .api import MingTinyTTS

        return MingTinyTTS
    if name == "MingTinyEngine":
        from .inference import MingTinyEngine

        return MingTinyEngine
    if name == "build_control_instruction":
        from .task_protocol import build_control_instruction

        return build_control_instruction
    if name == "LingV3TTSConfig":
        from .configuration_lingv3_tts import LingV3TTSConfig

        return LingV3TTSConfig
    if name == "LingV3TTSForConditionalGeneration":
        from .modeling_lingv3_tts import LingV3TTSForConditionalGeneration

        return LingV3TTSForConditionalGeneration
    raise AttributeError(name)
