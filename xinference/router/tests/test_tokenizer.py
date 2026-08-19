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
