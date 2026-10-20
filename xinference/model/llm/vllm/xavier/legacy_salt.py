# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""A serializable request field for V0, which predates native cache salts."""

from typing import Optional

from vllm.sampling_params import SamplingParams


class XavierSamplingParams(SamplingParams):
    # A declared msgspec field survives clone(), pickle and actor boundaries.
    cache_salt: Optional[str] = None

    def __repr__(self) -> str:
        return "XavierSamplingParams(cache_salt=<redacted>)"
