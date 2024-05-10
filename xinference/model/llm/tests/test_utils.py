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
import pytest

from ....types import ChatCompletionMessage
from ..llm_family import CodePromptStyleV1, FIMSpecV1, PromptStyleV1
from ..llm_family import RepoLevelCodeCompletionSpecV1 as RepoLevelSpecV1
from ..utils import ChatModelMixin, CodeModelMixin


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


def test_prompt_style_llama3():
    prompt_style = PromptStyleV1(
        style_name="LLAMA3",
        system_prompt=(
            "You are a helpful, respectful and honest assistant. Always answer"
            " as helpfully as possible, while being safe. Your answers should not include any"
            " harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please"
            " ensure that your responses are socially unbiased and positive in nature.\n\nIf a"
            " question does not make any sense, or is not factually coherent, explain why instead"
            " of answering something not correct. If you don't know the answer to a question,"
            " please don't share false information"
        ),
        roles=["user", "assistant"],
        intra_message_sep="\n\n",
        inter_message_sep="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
    chat_history = [
        ChatCompletionMessage(role=prompt_style.roles[0], content="Hi there."),
        ChatCompletionMessage(
            role=prompt_style.roles[1], content="Hello, how may I help you?"
        ),
    ]
    expected = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful, respectful and honest assistant. Always answer"
        " as helpfully as possible, while being safe. Your answers should not include any"
        " harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please"
        " ensure that your responses are socially unbiased and positive in nature.\n\nIf a"
        " question does not make any sense, or is not factually coherent, explain why instead"
        " of answering something not correct. If you don't know the answer to a question,"
        " please don't share false information<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi there.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nHello, how may I help you?<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nWrite a poem.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
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


def test_path_to_name():
    path = "/home/test/works/project/main.py"
    assert "main.py" == CodeModelMixin._path_to_name(path)

    path = "/main.py"
    assert "main.py" == CodeModelMixin._path_to_name(path)

    path = "main.py"
    assert "main.py" == CodeModelMixin._path_to_name(path)

    path = ".main.py"
    assert ".main.py" == CodeModelMixin._path_to_name(path)

    path = r"\main.py"
    assert "main.py" == CodeModelMixin._path_to_name(path)

    path = r"C:\works\main.py"
    assert "main.py" == CodeModelMixin._path_to_name(path)


def test_code_prompt_style_starcoder():
    code_prompt_style = CodePromptStyleV1(
        style_name="STARCODER",
        fim_spec=FIMSpecV1(
            style="PSM",
            prefix="<fim_prefix>",
            middle="<fim_middle>",
            suffix="<fim_suffix>",
        ),
    )
    prompt = "def print_hello_world():"
    expected = prompt
    assert expected == CodeModelMixin.get_code_prompt(
        "completion", prompt, code_prompt_style
    )

    prompt = "def print_hello_world():\n    "
    suffix = "\n    print('Hello world!')"
    expected = "<fim_prefix>def print_hello_world():\n    <fim_suffix>\n    print('Hello world!')<fim_middle>"
    assert expected == CodeModelMixin.get_code_prompt(
        "infill", prompt, code_prompt_style, suffix
    )

    suffix = None
    with pytest.raises(ValueError) as exc_info:
        CodeModelMixin.get_code_prompt("infill", prompt, code_prompt_style, suffix)
        assert exc_info.value == ValueError("suffix is required in infill mode")

    with pytest.raises(ValueError) as exc_info:
        CodeModelMixin.get_code_prompt("test", prompt, code_prompt_style)
        assert exc_info.value == ValueError(
            "Unsupported generate mode: test, only 'PSM' and 'PMS' are supported now"
        )


def test_code_prompt_style_deepseek_coder():
    code_prompt_style = CodePromptStyleV1(
        style_name="DEEPSEEK_CODER",
        fim_spec=FIMSpecV1(
            style="PMS",
            prefix="<｜fim▁begin｜>",
            middle="<｜fim▁hole｜>",
            suffix="<｜fim▁end｜>",
        ),
        repo_level_spec=RepoLevelSpecV1(file_type="filename", file_separator="#"),
    )

    prompt = "#write a quick sort algorithm"
    expected = prompt

    assert expected == CodeModelMixin.get_code_prompt(
        "completion", prompt, code_prompt_style
    )

    prompt = """def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
"""
    suffix = """
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)"""

    expected = """<｜fim▁begin｜>def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    left = []
    right = []
<｜fim▁hole｜>
        if arr[i] < pivot:
            left.append(arr[i])
        else:
            right.append(arr[i])
    return quick_sort(left) + [pivot] + quick_sort(right)<｜fim▁end｜>"""

    assert expected == CodeModelMixin.get_code_prompt(
        "infill", prompt, code_prompt_style, suffix
    )

    files = {
        "utils.py": """import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert numpy data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    return X_train, X_test, y_train, y_test

def evaluate_predictions(y_test, y_pred):
    return accuracy_score(y_test, y_pred)""",
        "model.py": """import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.fc(x)

    def train_model(self, X_train, y_train, epochs, lr, batch_size):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Create DataLoader for batches
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X_test):
        with torch.no_grad():
            outputs = self(X_test)
            _, predicted = outputs.max(1)
        return predicted.numpy()""",
        "main.py": """from utils import load_data, evaluate_predictions
from model import IrisClassifier as Classifier

def main():
    # Model training and evaluation
""",
    }

    prompt = ""

    expected = """#utils.py
import torch
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert numpy data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    return X_train, X_test, y_train, y_test

def evaluate_predictions(y_test, y_pred):
    return accuracy_score(y_test, y_pred)
#model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.fc(x)

    def train_model(self, X_train, y_train, epochs, lr, batch_size):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Create DataLoader for batches
        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, X_test):
        with torch.no_grad():
            outputs = self(X_test)
            _, predicted = outputs.max(1)
        return predicted.numpy()
#main.py
from utils import load_data, evaluate_predictions
from model import IrisClassifier as Classifier

def main():
    # Model training and evaluation
"""
    assert expected == CodeModelMixin.get_code_prompt(
        "completion", prompt, code_prompt_style, None, None, files
    )


def test_code_prompt_style_without_fim():
    code_prompt_style = CodePromptStyleV1(
        style_name="NO_FIM_CODER",
    )
    prompt = "def print_hello_world():\n    "
    suffix = "\n    print('Hello world!')"
    with pytest.raises(ValueError) as exc_info:
        CodeModelMixin.get_code_prompt("infill", prompt, code_prompt_style, suffix)
        assert exc_info.value == ValueError(
            "This model is not support infill mode generate"
        )
