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

import abc
import contextlib
import logging
import textwrap
from abc import abstractmethod
from time import time
from typing import Dict, List, Optional, Tuple, Type, TypedDict

logger = logging.getLogger(__name__)


class Completion(TypedDict):
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: Optional[str]
    elapsed_time: Optional[int]


class LlamaCppGenerateConfig(TypedDict):
    # TODO
    pass


class LlamaCppModelConfig(TypedDict):
    # TODO
    pass


class ChatHistory:
    _inputs: List[str]
    _outputs: List[str]

    def __init__(self):
        self._inputs = []
        self._outputs = []

    def to_prompt(
        self,
        system_prompt: str,
        sep: str,
        user_name: str,
        assistant_name: str,
        input: str,
    ) -> str:
        ret = system_prompt
        for i, o in zip(self._inputs, self._outputs):
            ret += f"{sep}{user_name}: {i}"
            ret += f"{sep}{assistant_name}: {o}"
        ret += f"{sep}{user_name}: {input}"
        ret += f"{sep}{assistant_name}:"
        return ret

    def append(self, i: str, o: str):
        self._inputs.append(i)
        self._outputs.append(o)

    def clear(self):
        self._inputs = []
        self._outputs = []


class Model(abc.ABC):
    name: str

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def load(self):
        pass


MODEL_TO_CLS: Dict[str, Tuple[Type[Model], Dict]] = dict()


def register_model(config: Dict):
    def wrap_cls(cls: Type[Model]):
        MODEL_TO_CLS[cls.name] = (cls, config)
        return cls

    return wrap_cls


class LlamaCppModel(Model):
    def __init__(
        self,
        model_path: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__()
        self._model_path = model_path
        self._llamacpp_model_config = llamacpp_model_config or {}
        self._llm = None

    def load(self):
        from llama_cpp import Llama

        with contextlib.redirect_stdout(None):
            self._llm = Llama(
                model_path=self._model_path,
                verbose=False,
                **self._llamacpp_model_config,
            )

    def generate(
        self, prompt: str, generate_config: Optional[LlamaCppGenerateConfig] = None
    ) -> Completion:
        logger.debug("prompt:\n%s", "\n".join(textwrap.wrap(prompt, width=80)))
        generate_config = generate_config or {}

        start = time()
        with contextlib.redirect_stdout(None):
            assert self._llm is not None
            completion = self._llm(prompt=prompt, **generate_config)
        elapsed = time() - start

        completion = Completion(
            text=self._format_completion(completion["choices"][0]["text"]),
            prompt_tokens=completion["usage"]["prompt_tokens"],
            completion_tokens=completion["usage"]["completion_tokens"],
            finish_reason=completion["choices"][0]["finish_reason"],
            elapsed_time=int(elapsed),
        )

        logger.debug("completion:\n%s", completion)
        return completion

    @staticmethod
    def _format_completion(text: str):
        return text.strip()


class LlamaCppChatModel(LlamaCppModel):
    _history = ChatHistory()

    def __init__(
        self,
        model_path: str,
        system_prompt: str,
        sep: str,
        user_name: str,
        assistant_name: str,
        llamacpp_model_config: Optional[LlamaCppModelConfig] = None,
    ):
        super().__init__(model_path, llamacpp_model_config)
        self._system_prompt: str = system_prompt
        self._sep: str = sep
        self._user_name: str = user_name
        self._assistant_name: str = assistant_name

    def chat(
        self, prompt: str, generate_config: Optional[LlamaCppGenerateConfig] = None
    ) -> Completion:
        full_prompt = self._history.to_prompt(
            self._system_prompt,
            self._sep,
            self._user_name,
            self._assistant_name,
            prompt,
        )

        completion = self.generate(full_prompt, generate_config)
        self._history.append(prompt, completion["text"])
        return completion

    def clear(self):
        self._history.clear()
