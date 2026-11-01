# CI: linux
import copy
import json

import pytest
from packaging import version

from ...llm_family import match_llm
from ...utils import ChatModelMixin
from .. import core
from .test_speculative_config import _model_for_config


@pytest.fixture
def family():
    return copy.deepcopy(
        match_llm("MiniMax-M3.1-Flash", "fp8", 428, "mxfp8", "huggingface")
    )


@pytest.mark.parametrize("vllm_version,expected", [("0.32.0", False), ("0.33.0", True)])
def test_version_gate(monkeypatch, family, vllm_version, expected):
    monkeypatch.setattr(
        core,
        "_get_effective_vllm_version_for_family",
        lambda _: version.parse(vllm_version),
    )
    monkeypatch.setattr(core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(core.VLLMMultiModel, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(core.VLLMMultiModel, "_is_linux", lambda: True)
    result = core.VLLMMultiModel.match_json(family, family.model_specs[0], "mxfp8")
    assert (result if isinstance(result, bool) else result[0]) is expected
    if not expected:
        assert "MiniMax-M3.1-Flash requires vLLM >= 0.33.0" in result[1]


def test_supported_engine_registration(monkeypatch, family):
    import xinference.model.llm as llm

    from ...sglang import core as sglang
    from ...transformers.core import PytorchChatModel

    monkeypatch.setattr(core, "VLLM_INSTALLED", True)
    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.33.0"))
    monkeypatch.setattr(core.VLLMMultiModel, "check_lib", lambda: True)
    monkeypatch.setattr(core.VLLMMultiModel, "_has_cuda_device", lambda: True)
    monkeypatch.setattr(core.VLLMMultiModel, "_is_linux", lambda: True)
    monkeypatch.setattr(sglang.SGLANGVisionModel, "check_lib", lambda: True)
    monkeypatch.setattr(PytorchChatModel, "check_lib", lambda: True)
    monkeypatch.setattr(
        llm,
        "SUPPORTED_ENGINES",
        {
            "vLLM": [core.VLLMMultiModel],
            "SGLang": [sglang.SGLANGVisionModel],
            "Transformers": [PytorchChatModel],
        },
    )
    monkeypatch.setattr(llm, "LLM_ENGINES", {})
    llm.generate_engine_config_by_model_family(family)
    assert set(llm.LLM_ENGINES[family.model_name]) == {"vLLM"}
    assert family.model_specs[0].model_id == "MiniMaxAI/MiniMax-M3.1-Flash"
    assert core._get_virtualenv_vllm_version(family) == version.parse("0.33.0")


def test_sparse_cache_and_checkpoint_quantization(monkeypatch, family, tmp_path):
    monkeypatch.setattr(core, "VLLM_VERSION", version.parse("0.33.0"))
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "mxfp8"}})
    )
    model = _model_for_config(model_path=tmp_path)
    model.model_family = family
    model.model_spec = family.model_specs[0]
    config = model._sanitize_model_config({})
    assert config["block_size"] == 128
    assert config["disable_hybrid_kv_cache_manager"] is True
    assert config["enable_thinking"] is True
    # Preserve the checkpoint's mixed MXFP8/NVFP4 packing for vLLM to detect.
    assert config["quantization"] is None
    explicit = {"block_size": 256, "disable_hybrid_kv_cache_manager": False}
    assert model._sanitize_model_config(explicit)["block_size"] == 256
    assert explicit["disable_hybrid_kv_cache_manager"] is False


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_reasoning_effort_and_tool_prompt(family, effort):
    model = ChatModelMixin()
    model.model_family = family
    prompt = model.get_full_context(
        [{"role": "user", "content": "What is the weather in Hangzhou?"}],
        family.chat_template,
        reasoning_effort=effort,
        tools=[
            {
                "type": "function",
                "function": {"name": "weather", "parameters": {"type": "object"}},
            }
        ],
    )
    assert f"<effort>{effort}</effort>" in prompt
    assert '<tool>{"name": "weather"' in prompt
    assert prompt.endswith("]~b]ai\n")
