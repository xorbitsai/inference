# CI: linux
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ..core import PytorchChatModel


@pytest.mark.parametrize(
    "architecture,model_format,expected",
    [
        ("DeepseekV41ForCausalLM", "fp8", True),
        ("UnrelatedForCausalLM", "fp8", False),
        ("DeepseekV41ForCausalLM", "ggufv2", False),
        ("DeepseekV41ForCausalLM", "pytorch", True),
    ],
)
def test_deepseek_v41_fp8_match(architecture, model_format, expected):
    family = Mock()
    family.architectures = [architecture]
    family.model_ability = ["chat", "vision", "reasoning", "tools"]
    family.has_architecture.side_effect = lambda *names: architecture in names
    family.matches_supported_architectures.return_value = False
    result = PytorchChatModel.match_json(
        family, SimpleNamespace(model_format=model_format), "fp8"
    )
    assert (result is True) is expected


def test_unsloth_gguf_registry():
    from ....utils import generate_model_file_names_with_quantization_parts
    from ...llm_family import get_llm_family_by_name

    family = get_llm_family_by_name("DeepSeek-V4.1-Flash-1017")
    specs = [s for s in family.model_specs if s.model_format == "ggufv2"]
    assert specs
    for spec in specs:
        assert spec.model_id == "unsloth/DeepSeek-V4.1-Flash-1017-GGUF"
        files, first, merge = generate_model_file_names_with_quantization_parts(spec)
        assert merge
        assert len(files) == int(files[-1].split("-of-")[-1].split(".")[0])
        assert spec.multimodal_projectors == ["mmproj-BF16.gguf", "mmproj-F16.gguf"]
