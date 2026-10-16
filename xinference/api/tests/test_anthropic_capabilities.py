from types import SimpleNamespace

from xinference.api.protocols import parse_anthropic_request
from xinference.api.protocols.anthropic_capabilities import (
    AnthropicBackendCapabilities,
    SupportLevel,
    plan_anthropic_compatibility,
)
from xinference.model.llm.core import LLM


class StubLLM(LLM):
    def load(self):
        pass

    @classmethod
    def check_lib(cls):
        return True

    @classmethod
    def match_json(cls, llm_family, llm_spec, quantization):
        return True


def test_canonical_request_tracks_requested_features():
    request = parse_anthropic_request(
        {
            "model": "test-model",
            "max_tokens": 128,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/image.png",
                            },
                        }
                    ],
                }
            ],
            "thinking": {"type": "adaptive"},
            "tools": [
                {
                    "name": "weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object"},
                }
            ],
        }
    )

    assert {"vision", "reasoning", "adaptive_thinking", "tools"} <= set(
        request.requested_features
    )
    body = request.to_openai_body()
    assert body["_anthropic_content_metrics"]["total_images"] == 1
    assert "vision" in body["_anthropic_requested_features"]


def test_planner_rejects_images_over_backend_limit():
    plan = plan_anthropic_compatibility(
        frozenset({"vision"}),
        {"total_images": 3},
        AnthropicBackendCapabilities(
            engine="vllm",
            vision=SupportLevel.NATIVE,
            max_images_per_prompt=2,
        ),
    )

    assert plan.allowed is False
    assert plan.effective_max_images == 2
    assert plan.decisions[0].feature == "vision.max_images_per_prompt"


def test_planner_records_translated_and_degraded_features():
    plan = plan_anthropic_compatibility(
        frozenset({"adaptive_thinking", "reasoning_effort"}),
        {"total_images": 0},
        AnthropicBackendCapabilities(
            engine="vllm",
            adaptive_thinking=SupportLevel.TRANSLATED,
            reasoning_effort=SupportLevel.DEGRADED,
        ),
    )

    assert plan.allowed is True
    assert [(decision.feature, decision.action) for decision in plan.decisions] == [
        ("adaptive_thinking", "translated"),
        ("reasoning_effort", "degraded"),
    ]


def test_loaded_model_capabilities_include_deployment_image_limit():
    model = StubLLM.__new__(StubLLM)
    model.model_family = SimpleNamespace(
        model_ability=["chat", "vision", "tools", "reasoning"]
    )
    model.reasoning_parser = None
    model.tool_parser = None
    model._model_config = {"limit_mm_per_prompt": {"image": 4}}

    capabilities = model.get_anthropic_capabilities()

    assert capabilities.vision == SupportLevel.NATIVE
    assert capabilities.tools == SupportLevel.NATIVE
    assert capabilities.reasoning == SupportLevel.NATIVE
    assert capabilities.max_images_per_prompt == 4


def test_capabilities_allow_partially_initialized_model_instances():
    model = StubLLM.__new__(StubLLM)
    model.model_family = SimpleNamespace(model_ability=["chat"])

    capabilities = model.get_anthropic_capabilities()

    assert capabilities.tools == SupportLevel.UNSUPPORTED
    assert capabilities.reasoning == SupportLevel.UNSUPPORTED


def test_planner_rejects_unsupported_explicit_constraint():
    plan = plan_anthropic_compatibility(
        frozenset({"parallel_tools"}),
        {"total_images": 0},
        AnthropicBackendCapabilities(engine="transformers"),
    )

    assert plan.allowed is False
    assert plan.decisions[0].feature == "parallel_tools"
    assert plan.decisions[0].action == "rejected"
