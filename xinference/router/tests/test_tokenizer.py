import json
from copy import deepcopy
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from xinference.router.tokenizer import DeepSeekV4TokenEstimator, TokenizationError


def make_assets(tmp_path: Path) -> Path:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    encoding = tmp_path / "encoding"
    encoding.mkdir()
    (encoding / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, thinking_mode, reasoning_effort=None):\n"
        "    prefix = '<think> ' if thinking_mode == 'thinking' else ''\n"
        "    return prefix + ' '.join(str(m.get('content', '')) for m in messages)\n"
    )
    return tmp_path


def test_estimates_final_budget(tmp_path: Path) -> None:
    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=64, default_output_tokens=512
    )
    result = estimator.estimate(
        {
            "messages": [{"role": "user", "content": "hello hello"}],
            "max_tokens": 8,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    )
    assert result.prompt_tokens == 2
    assert result.output_tokens == 8
    assert result.total_tokens == 74
    assert result.enable_thinking is False


def test_thinking_and_text_content_list(tmp_path: Path) -> None:
    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=0, default_output_tokens=1
    )
    result = estimator.estimate(
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]}
            ],
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )
    assert result.enable_thinking is True
    assert result.prompt_tokens == 4
    assert result.total_tokens == 5


def test_null_text_content_part_is_treated_as_empty(tmp_path: Path) -> None:
    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=0, default_output_tokens=1
    )

    result = estimator.estimate(
        {"messages": [{"role": "user", "content": [{"type": "text", "text": None}]}]}
    )

    assert result.prompt_tokens == 0
    assert result.total_tokens == 1


def test_chat_template_kwargs_cannot_override_messages(tmp_path: Path) -> None:
    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=0, default_output_tokens=1
    )

    result = estimator.estimate(
        {
            "messages": [{"role": "user", "content": "hello hello"}],
            "chat_template_kwargs": {
                "messages": [],
                "enable_thinking": True,
                "reasoning_effort": "high",
            },
        }
    )

    assert result.prompt_tokens == 5
    assert result.enable_thinking is True


def test_normalization_does_not_mutate_input_messages(tmp_path: Path) -> None:
    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=0, default_output_tokens=1
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "tool_calls": [{"id": "call-1"}],
        }
    ]

    estimator.estimate({"messages": messages})

    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "tool_calls": [{"id": "call-1"}],
        }
    ]


def test_rejects_multimodal_content(tmp_path: Path) -> None:
    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=0, default_output_tokens=1
    )
    with pytest.raises(TokenizationError):
        estimator.estimate(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "x"}}],
                    }
                ]
            }
        )


def test_top_level_thinking_overrides_extra_body_and_template_kwargs(
    tmp_path: Path,
) -> None:
    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=0, default_output_tokens=1
    )

    result = estimator.estimate(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": False,
            "extra_body": {"enable_thinking": True},
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )

    assert result.enable_thinking is False
    assert result.prompt_tokens == 1


def test_extra_body_thinking_is_normalized(tmp_path: Path) -> None:
    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=0, default_output_tokens=1
    )

    result = estimator.estimate(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "extra_body": {"enable_thinking": True},
        }
    )

    assert result.enable_thinking is True
    assert result.prompt_tokens == 4


@pytest.mark.parametrize(
    "template_kwargs,overrides,expected",
    [
        ({"thinking": True}, {}, True),
        (json.dumps({"thinking": True}), {}, True),
        ({"thinking": False}, {}, False),
        ({"thinking": True, "enable_thinking": False}, {}, False),
        ({"thinking": True, "enable_thinking": None}, {}, False),
        ({"thinking": True}, {"enable_thinking": False}, False),
        ({"thinking": True}, {"extra_body": {"enable_thinking": False}}, False),
        (
            {"thinking": True},
            {"enable_thinking": True, "extra_body": {"enable_thinking": False}},
            True,
        ),
        ({"thinking": True}, {"enable_thinking": None}, True),
        ({"thinking": True}, {"enable_thinking": "invalid"}, True),
        ({"thinking": "true"}, {}, False),
        ({"thinking": 1}, {}, False),
        ({"thinking": None}, {}, False),
        ({"enable_thinking": True}, {}, True),
        (json.dumps({"enable_thinking": True}), {}, True),
        ({}, {}, False),
        (None, {}, False),
    ],
)
def test_thinking_alias_normalization(
    tmp_path: Path, template_kwargs, overrides, expected: bool
) -> None:
    from xinference.router.app import _payload_thinking

    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=0, default_output_tokens=1
    )
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": template_kwargs,
        **overrides,
    }
    original = deepcopy(payload)

    result = estimator.estimate(payload)

    assert result.enable_thinking is expected
    assert result.prompt_tokens == (4 if expected else 1)
    assert _payload_thinking(payload) is expected
    assert payload == original


@pytest.mark.parametrize("template_kwargs", ["{", "null", "[]", [1], True])
def test_thinking_alias_invalid_kwargs(tmp_path: Path, template_kwargs) -> None:
    from xinference.router.app import _payload_thinking

    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=0, default_output_tokens=1
    )
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": template_kwargs,
        "enable_thinking": True,
    }
    with pytest.raises(TokenizationError):
        estimator.estimate(payload)
    assert _payload_thinking(payload) is False


def test_max_completion_tokens_overrides_max_tokens(tmp_path: Path) -> None:
    estimator = DeepSeekV4TokenEstimator(
        make_assets(tmp_path), reserve_tokens=0, default_output_tokens=1
    )

    result = estimator.estimate(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "max_completion_tokens": 16,
        }
    )

    assert result.output_tokens == 16
    assert result.total_tokens == result.prompt_tokens + 16
