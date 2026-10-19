# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import sys
from types import SimpleNamespace

import pytest
from packaging.version import Version

from .. import core


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt",
    [
        "text",
        {"prompt_token_ids": [1, 2]},
        {"prompt": "text", "multi_modal_data": {"image": "image"}},
    ],
)
async def test_native_v1_salt_is_prompt_metadata_not_sampling_params(
    monkeypatch, prompt
):
    class SamplingParams:
        def __init__(self, n=1, max_tokens=2):
            self.max_tokens = max_tokens

    class ReachedEngine(Exception):
        pass

    def generate(actual_prompt, params, *args, **kwargs):
        expected = {"prompt": prompt} if isinstance(prompt, str) else dict(prompt)
        assert actual_prompt == {**expected, "cache_salt": "scope"}
        assert not hasattr(params, "cache_salt")
        raise ReachedEngine

    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(SamplingParams=SamplingParams),
    )
    monkeypatch.setattr(core, "VLLM_INSTALLED", False)
    monkeypatch.setattr(core, "VLLM_VERSION", Version("0.28.0"))
    model = object.__new__(core.VLLMModel)
    model._engine = SimpleNamespace(generate=generate)
    model._nixl_config = model._xavier_config = None
    model.reasoning_parser = None
    model._sanitize_generate_config = lambda config: dict(
        n=1, lora_name=None, max_tokens=2, stream=False
    )
    with pytest.raises(ReachedEngine):
        await core.VLLMModel.async_generate.__wrapped__(
            model, prompt, {"cache_salt": "scope"}
        )
