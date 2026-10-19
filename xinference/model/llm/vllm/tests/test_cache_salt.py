# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
# CI: linux
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from packaging.version import Version

from .. import core


def salt_model(monkeypatch, vllm_version, *, v1=True, native_salt=True):
    class SamplingParams:
        def __init__(self, n=1, max_tokens=2):
            self.max_tokens = max_tokens

    class LegacySamplingParams(SamplingParams):
        def __init__(self, n=1, max_tokens=2, cache_salt=None):
            super().__init__(n=n, max_tokens=max_tokens)
            self.cache_salt = cache_salt

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            envs=SimpleNamespace(
                is_set=lambda name: name == "VLLM_USE_V1", VLLM_USE_V1=v1
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.sampling_params",
        SimpleNamespace(SamplingParams=SamplingParams),
    )
    prompt_type = SimpleNamespace(
        __annotations__={"cache_salt": str} if native_salt else {}
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.inputs.data",
        SimpleNamespace(TextPrompt=prompt_type, TokensPrompt=prompt_type),
    )
    monkeypatch.setitem(
        sys.modules,
        "xinference.model.llm.vllm.xavier.legacy_salt",
        SimpleNamespace(XavierSamplingParams=LegacySamplingParams),
    )
    monkeypatch.setattr(core, "VLLM_INSTALLED", False)
    monkeypatch.setattr(core, "VLLM_VERSION", Version(vllm_version))
    model = object.__new__(core.VLLMModel)
    model._engine = SimpleNamespace(generate=Mock())
    model._nixl_config = model._xavier_config = None
    model.reasoning_parser = None
    model._sanitize_generate_config = lambda config: dict(
        n=1, lora_name=None, max_tokens=2, stream=config.get("stream", False)
    )
    return model


@pytest.mark.asyncio
@pytest.mark.parametrize("vllm_version", ["0.10.2", "0.11.0", "0.28.0"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "prompt",
    [
        "text",
        {"prompt_token_ids": [1, 2]},
        {"prompt": "text", "multi_modal_data": {"image": "image"}},
    ],
)
async def test_native_v1_salt_is_prompt_metadata_not_sampling_params(
    monkeypatch, prompt, stream, vllm_version
):
    class ReachedEngine(Exception):
        pass

    def generate(actual_prompt, params, *args, **kwargs):
        expected = {"prompt": prompt} if isinstance(prompt, str) else dict(prompt)
        assert actual_prompt == {**expected, "cache_salt": "scope"}
        assert not hasattr(params, "cache_salt")
        raise ReachedEngine

    model = salt_model(monkeypatch, vllm_version)
    model._engine.generate = generate
    with pytest.raises(ReachedEngine):
        await core.VLLMModel.async_generate.__wrapped__(
            model, prompt, {"cache_salt": "scope", "stream": stream}
        )


@pytest.mark.asyncio
async def test_v0_xavier_salt_stays_in_sampling_params(monkeypatch):
    model = salt_model(monkeypatch, "0.10.2", v1=False)
    model._xavier_config = {"enabled": True}
    reached_engine = RuntimeError("Reached engine")

    def generate(prompt, params, *args, **kwargs):
        assert prompt == "text"
        assert params.cache_salt == "scope"
        raise reached_engine

    model._engine.generate = generate
    with pytest.raises(RuntimeError) as caught:
        await core.VLLMModel.async_generate.__wrapped__(
            model, "text", {"cache_salt": "scope"}
        )
    assert caught.value is reached_engine


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vllm_version,v1,native_salt", [("0.10.2", False, True), ("0.8.5", True, False)]
)
async def test_non_xavier_engines_without_native_salt_reject_before_generation(
    monkeypatch, vllm_version, v1, native_salt
):
    model = salt_model(monkeypatch, vllm_version, v1=v1, native_salt=native_salt)
    with pytest.raises(ValueError, match="cache_salt"):
        await core.VLLMModel.async_generate.__wrapped__(
            model, "text", {"cache_salt": "scope"}
        )
    model._engine.generate.assert_not_called()


@pytest.mark.asyncio
async def test_unsalted_request_does_not_probe_salt_capabilities(monkeypatch):
    model = salt_model(monkeypatch, "0.10.2", v1=False)
    model._is_vllm_v1 = Mock(side_effect=AssertionError("Unexpected capability probe"))
    reached_engine = RuntimeError("Reached engine")
    model._engine.generate.side_effect = reached_engine
    with pytest.raises(RuntimeError) as caught:
        await core.VLLMModel.async_generate.__wrapped__(model, "text", {})
    assert caught.value is reached_engine
    assert model._engine.generate.call_args.args[0] == "text"
    model._is_vllm_v1.assert_not_called()
