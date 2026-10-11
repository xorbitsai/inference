import copy

import pytest

from ..core import chat_context_var
from ..llm_family import match_llm
from ..reasoning_parser import ReasoningParser
from ..transformers.core import PytorchChatModel
from ..utils import ChatModelMixin


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_thinking": False, "reasoning_effort": "low", "clear_thinking": True},
        '{"thinking": false, "reasoning_effort": "high"}',
        {},
    ],
)
def test_transformers_glm53_always_thinks(custom, kwargs):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    model = object.__new__(PytorchChatModel)
    model.model_family = match_llm(
        "glm-5.3",
        model_format="pytorch",
        model_size_in_billions=753,
        quantization="none",
        download_hub="huggingface",
    ).copy(deep=True)
    if custom:
        model.model_family.model_name = "custom-glm53"
        model.model_family.model_family = "glm-5.3"
    model._tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    )
    model.prepare_parse_reasoning_content(True, True)
    original = copy.deepcopy(kwargs)
    token = chat_context_var.set({})
    try:
        prompt = model._get_full_prompt(
            [{"role": "user", "content": "hello"}],
            None,
            {"chat_template_kwargs": kwargs},
        )
        assert prompt.endswith("<|assistant|><think>")
        assert chat_context_var.get()["enable_thinking"] is True
        assert model.reasoning_parser.check_content_parser() is True
        # A batch may switch request contexts before processing this result.
        chat_context_var.set({"enable_thinking": False})
        assert model.reasoning_parser.check_content_parser() is True
        response = model._to_chat_completion(
            {
                "id": "test",
                "created": 1,
                "model": "glm53",
                "usage": None,
                "choices": [
                    {
                        "index": 0,
                        "text": "analysis</think>answer",
                        "finish_reason": "stop",
                    }
                ],
            },
            model.reasoning_parser,
        )
        message = response["choices"][0]["message"]
        assert message["content"] == "answer"
        assert message["reasoning_content"].strip() == "analysis"
        delta = model.reasoning_parser.extract_reasoning_content_streaming(
            "analysis", "analysis</think>answer", "</think>answer"
        )
        assert delta["content"] == "answer"
        assert kwargs == original
    finally:
        chat_context_var.reset(token)


@pytest.mark.parametrize("allow_disable", [True, False])
def test_thinking_constraint_preserves_output_preference(allow_disable):
    parser = ReasoningParser(
        reasoning_content=False,
        enable_thinking=False,
        allow_disable_thinking=allow_disable,
    )
    token = chat_context_var.set({"enable_thinking": False})
    try:
        kwargs = ChatModelMixin._get_chat_template_kwargs_from_generate_config(
            {"chat_template_kwargs": {"enable_thinking": False}}, parser
        )
        assert kwargs["enable_thinking"] is (not allow_disable)
        assert parser.is_enable_thinking() is (not allow_disable)
        assert parser.check_content_parser() is False
        defaults = ChatModelMixin._get_chat_template_kwargs_from_generate_config(
            None, parser
        )
        assert defaults["enable_thinking"] is (not allow_disable)
    finally:
        chat_context_var.reset(token)
