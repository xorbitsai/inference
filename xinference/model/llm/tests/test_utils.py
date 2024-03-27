# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pytest

from ....types import ChatCompletionMessage
from ..llm_family import PromptStyleV1
from ..utils import ChatModelMixin, ModelHubUtil


def test_prompt_style_add_colon_single():
    prompt_style = PromptStyleV1(
        style_name="ADD_COLON_SINGLE",
        system_prompt=(
            "A chat between a curious human and an artificial intelligence assistant. The "
            "assistant gives helpful, detailed, and polite answers to the human's questions."
        ),
        roles=["user", "assistant"],
        intra_message_sep="\n### ",
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "A chat between a curious human and an artificial intelligence assistant. The assistant"
        " gives helpful, detailed, and polite answers to the human's questions."
        "\n### user: Hi there."
        "\n### assistant: Hello, how may I help you?"
        "\n### user: Write a poem."
        "\n### assistant:"
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_add_colon_two():
    prompt_style = PromptStyleV1(
        style_name="ADD_COLON_TWO",
        system_prompt=(
            "A chat between a curious user and an artificial intelligence assistant. The "
            "assistant gives helpful, detailed, and polite answers to the user's questions."
        ),
        roles=["USER", "ASSISTANT"],
        intra_message_sep=" ",
        inter_message_sep="</s>",
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "A chat between a curious user and an artificial intelligence assistant. The "
        "assistant gives helpful, detailed, and polite answers to the user's questions. "
        "USER: Hi there. "
        "ASSISTANT: Hello, how may I help you?</s>"
        "USER: Write a poem. "
        "ASSISTANT:"
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_no_colon_two():
    prompt_style = PromptStyleV1(
        style_name="NO_COLON_TWO",
        system_prompt="",
        roles=[" <reserved_102> ", " <reserved_103> "],
        intra_message_sep="",
        inter_message_sep="</s>",
        stop_token_ids=[2, 195],
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        " <reserved_102> Hi there."
        " <reserved_103> Hello, how may I help you?</s>"
        " <reserved_102> Write a poem."
        " <reserved_103> "
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_llama2():
    prompt_style = PromptStyleV1(
        style_name="LLAMA2",
        system_prompt=(
            "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer"
            " as helpfully as possible, while being safe. Your answers should not include any"
            " harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please"
            " ensure that your responses are socially unbiased and positive in nature.\n\nIf a"
            " question does not make any sense, or is not factually coherent, explain why instead"
            " of answering something not correct. If you don't know the answer to a question,"
            " please don't share false information.\n<</SYS>>\n\n"
        ),
        roles=["[INST]", "[/INST]"],
        intra_message_sep=" ",
        inter_message_sep=" </s><s>",
        stop_token_ids=[2],
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer"
        " as helpfully as possible, while being safe. Your answers should not include any"
        " harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please"
        " ensure that your responses are socially unbiased and positive in nature.\n\nIf a"
        " question does not make any sense, or is not factually coherent, explain why instead"
        " of answering something not correct. If you don't know the answer to a question,"
        " please don't share false information.\n<</SYS>>\n\nHi there.[/INST] Hello, how may I help"
        " you? </s><s>[INST] Write a poem. [/INST]"
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_falcon():
    prompt_style = PromptStyleV1(
        style_name="FALCON",
        system_prompt="",
        roles=["User", "Assistant"],
        intra_message_sep="\n",
        inter_message_sep="<|endoftext|>",
        stop=["\nUser"],
        stop_token_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "User: Hi there.\n\n"
        "Assistant: Hello, how may I help you?\n\n"
        "User: Write a poem.\n\n"
        "Assistant:"
    )

    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_chatglm_v1():
    prompt_style = PromptStyleV1(
        style_name="CHATGLM",
        system_prompt="",
        roles=["问", "答"],
        intra_message_sep="\n",
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "[Round 0]\n问：Hi there.\n"
        "答：Hello, how may I help you?\n"
        "[Round 1]\n问：Write a poem.\n"
        "答："
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_chatglm_v2():
    prompt_style = PromptStyleV1(
        style_name="CHATGLM",
        system_prompt="",
        roles=["问", "答"],
        intra_message_sep="\n\n",
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "[Round 1]\n\n问：Hi there.\n\n"
        "答：Hello, how may I help you?\n\n"
        "[Round 2]\n\n问：Write a poem.\n\n"
        "答："
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_chatglm_v3():
    prompt_style = PromptStyleV1(
        style_name="CHATGLM3",
        system_prompt="",
        roles=["user", "assistant"],
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "<|user|>\n Hi there.\n"
        "<|assistant|>\n Hello, how may I help you?\n"
        "<|user|>\n Write a poem.\n"
        "<|assistant|>"
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_xverse():
    prompt_style = PromptStyleV1(
        style_name="XVERSE",
        system_prompt="",
        roles=["user", "assistant"],
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "<|user|> \n Hi there."
        "<|assistant|> \n Hello, how may I help you?"
        "<|user|> \n Write a poem."
        "<|assistant|>"
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_qwen():
    prompt_style = PromptStyleV1(
        style_name="QWEN",
        system_prompt="You are a helpful assistant.",
        roles=["user", "assistant"],
        intra_message_sep="\n",
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi there."
        "<|im_end|>\n<|im_start|>assistant\nHello, how may I help you?<|im_end|>\n<|im_start|>"
        "user\nWrite a poem.<|im_end|>\n<|im_start|>assistant\n"
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_chatml():
    prompt_style = PromptStyleV1(
        style_name="CHATML",
        system_prompt="<system>You are a wonderful code assistant\n",
        roles=["<|user|>", "<|assistant|>"],
        intra_message_sep="<|end|>",
    )

    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]

    expected = (
        "<system>You are a wonderful code assistant\n"
        "<|end|>\n"
        "<|user|>\n"
        "Hi there.<|end|>\n"
        "<|assistant|>\n"
        "Hello, how may I help you?<|end|>\n"
        "<|user|>\n"
        "Write me a HelloWorld Function<|end|>\n"
        "<|assistant|>\n"
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write me a HelloWorld Function", chat_history, prompt_style
    )


def test_prompt_style_internlm():
    prompt_style = PromptStyleV1(
        style_name="INTERNLM",
        system_prompt="",
        roles=["<|User|>", "<|Bot|>"],
        intra_message_sep="<eoh>\n",
        inter_message_sep="<eoa>\n",
    )

    expected = "<s><|User|>:Write a poem.<eoh>\n<|Bot|>:"
    actual = ChatModelMixin.get_prompt("Write a poem.", [], prompt_style)
    assert expected == actual

    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "<s><|User|>:Hi there.<eoh>\n<|Bot|>:Hello, how may I help you?<eoa>\n"
        "<|User|>:Write a poem.<eoh>\n<|Bot|>:"
    )
    actual = ChatModelMixin.get_prompt("Write a poem.", chat_history, prompt_style)
    assert expected == actual


def test_prompt_style_add_colon_single_cot():
    prompt_style = PromptStyleV1(
        style_name="ADD_COLON_SINGLE_COT",
        system_prompt=(
            "Below is an instruction that describes a task. Write a response that appropriately "
            "completes the request."
        ),
        roles=["Instruction", "Response"],
        intra_message_sep="\n\n### ",
    )

    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request."
        "\n\n### Instruction: Hi there."
        "\n\n### Response: Hello, how may I help you?"
        "\n\n### Instruction: Write a poem."
        "\n\n### Response: Let's think step by step."
    )
    assert expected == ChatModelMixin.get_prompt(
        "Write a poem.", chat_history, prompt_style
    )


def test_prompt_style_zephyr():
    prompt_style = PromptStyleV1(
        style_name="NO_COLON_TWO",
        system_prompt=(
            "<|system|>\nYou are a friendly chatbot who always responds in the style of a pirate.</s>\n"
        ),
        roles=["<|user|>\n", "<|assistant|>\n"],
        intra_message_sep="</s>\n",
        inter_message_sep="</s>\n",
        stop_token_ids=[2, 195],
        stop=["</s>"],
    )

    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "<|system|>\n"
        "You are a friendly chatbot who always responds in the style of a pirate.</s>\n"
        "<|user|>\n"
        "Hi there.</s>\n"
        "<|assistant|>\n"
        "Hello, how may I help you?</s>\n"
        "<|user|>\n"
        "Write a poem.</s>\n"
        "<|assistant|>\n"
    )
    actual = ChatModelMixin.get_prompt("Write a poem.", chat_history, prompt_style)
    assert expected == actual


def test_is_valid_model_name():
    from ...utils import is_valid_model_name

    assert is_valid_model_name("foo")
    assert is_valid_model_name("foo-bar")
    assert is_valid_model_name("foo_bar")
    assert is_valid_model_name("123")
    assert is_valid_model_name("foo@bar")
    assert is_valid_model_name("_foo")
    assert is_valid_model_name("-foo")
    assert not is_valid_model_name("foo bar")
    assert not is_valid_model_name("foo/bar")
    assert not is_valid_model_name("   ")
    assert not is_valid_model_name("")


@pytest.fixture
def model_hub_util():
    return ModelHubUtil()


def test__hf_api(model_hub_util):
    assert model_hub_util._hf_api is not None


def test__ms_api(model_hub_util):
    assert model_hub_util._ms_api is not None


def test_repo_exists(model_hub_util):
    assert model_hub_util.repo_exists(
        "TheBloke/KafkaLM-70B-German-V0.1-GGUF", "huggingface"
    )
    assert not model_hub_util.repo_exists("Nobody/No_This_Repo", "huggingface")
    with pytest.raises(ValueError, match="Unsupported model hub"):
        model_hub_util.repo_exists("Nobody/No_This_Repo", "unknown_hub")

    assert model_hub_util.repo_exists("qwen/Qwen1.5-72B-Chat-GGUF", "modelscope")
    assert not model_hub_util.repo_exists("Nobody/No_This_Repo", "modelscope")
    with pytest.raises(ValueError, match="Unsupported model hub"):
        model_hub_util.repo_exists("Nobody/No_This_Repo", "unknown_hub")


@pytest.mark.asyncio
async def test_a_repo_exists(model_hub_util):
    assert await model_hub_util.a_repo_exists(
        "TheBloke/KafkaLM-70B-German-V0.1-GGUF", "huggingface"
    )
    assert not await model_hub_util.a_repo_exists("Nobody/No_This_Repo", "huggingface")
    with pytest.raises(ValueError, match="Unsupported model hub"):
        model_hub_util.repo_exists("Nobody/No_This_Repo", "unknown_hub")

    assert await model_hub_util.a_repo_exists(
        "qwen/Qwen1.5-72B-Chat-GGUF", "modelscope"
    )
    assert not await model_hub_util.a_repo_exists("Nobody/No_This_Repo", "modelscope")
    with pytest.raises(ValueError, match="Unsupported model hub"):
        await model_hub_util.a_repo_exists("Nobody/No_This_Repo", "unknown_hub")


def test_get_config_path(model_hub_util):
    p = model_hub_util.get_config_path(
        "TheBloke/KafkaLM-70B-German-V0.1-GGUF", "huggingface"
    )
    assert p is not None
    assert os.path.isfile(p)

    assert model_hub_util.get_config_path("Nobody/No_This_Repo", "huggingface") is None

    p = model_hub_util.get_config_path("qwen/Qwen1.5-72B-Chat-GGUF", "modelscope")
    assert p is None

    p = model_hub_util.get_config_path("deepseek-ai/deepseek-vl-7b-chat", "modelscope")
    assert p is not None
    assert os.path.isfile(p)

    assert model_hub_util.get_config_path("Nobody/No_This_Repo", "modelscope") is None


@pytest.mark.asyncio
async def test_a_get_config_path_async(model_hub_util):
    p = await model_hub_util.a_get_config_path(
        "TheBloke/KafkaLM-70B-German-V0.1-GGUF", "huggingface"
    )
    assert p is not None
    assert os.path.isfile(p)

    assert (
        await model_hub_util.a_get_config_path("Nobody/No_This_Repo", "huggingface")
        is None
    )

    p = await model_hub_util.a_get_config_path(
        "qwen/Qwen1.5-72B-Chat-GGUF", "modelscope"
    )
    assert p is None

    p = await model_hub_util.a_get_config_path(
        "deepseek-ai/deepseek-vl-7b-chat", "modelscope"
    )
    assert p is not None
    assert os.path.isfile(p)

    assert (
        await model_hub_util.a_get_config_path("Nobody/No_This_Repo", "modelscope")
        is None
    )


def test_list_repo_files(model_hub_util):
    files = model_hub_util.list_repo_files(
        "TheBloke/KafkaLM-70B-German-V0.1-GGUF", "huggingface"
    )
    assert len(files) == 20

    files = model_hub_util.list_repo_files(
        "deepseek-ai/deepseek-vl-7b-chat", "modelscope"
    )
    assert len(files) == 12  # the `.gitattributes` file is not included

    with pytest.raises(ValueError, match="Repository Nobody/No_This_Repo not found."):
        model_hub_util.list_repo_files("Nobody/No_This_Repo", "huggingface")

    with pytest.raises(ValueError, match="Repository Nobody/No_This_Repo not found."):
        model_hub_util.list_repo_files("Nobody/No_This_Repo", "modelscope")


@pytest.mark.asyncio
async def test_a_list_repo_files(model_hub_util):
    files = await model_hub_util.a_list_repo_files(
        "TheBloke/KafkaLM-70B-German-V0.1-GGUF", "huggingface"
    )
    assert len(files) == 20

    files = await model_hub_util.a_list_repo_files(
        "deepseek-ai/deepseek-vl-7b-chat", "modelscope"
    )
    assert len(files) == 12  # the `.gitattributes` file is not included

    with pytest.raises(ValueError, match="Repository Nobody/No_This_Repo not found."):
        await model_hub_util.a_list_repo_files("Nobody/No_This_Repo", "huggingface")

    with pytest.raises(ValueError, match="Repository Nobody/No_This_Repo not found."):
        await model_hub_util.a_list_repo_files("Nobody/No_This_Repo", "modelscope")
