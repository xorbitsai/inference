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

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

import requests
import xoscar as xo

from .core.model import ModelActor
from .core.service import SupervisorActor
from .isolation import Isolation

if TYPE_CHECKING:
    from .model import ModelSpec
    from .model.llm.chatglm import ChatglmCppGenerateConfig
    from .model.llm.core import LlamaCppGenerateConfig
    from .model.llm.types import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessage,
        Completion,
        CompletionChunk,
    )


class Client:
    def __init__(self, endpoint: str):
        restful_client = RESTfulClient(endpoint)
        self._supervisor_address = restful_client._get_supervisor_internal_address()
        self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
        self._isolation.start()
        self._supervisor_ref: xo.ActorRefType["SupervisorActor"] = self._isolation.call(
            xo.actor_ref(address=self._supervisor_address, uid=SupervisorActor.uid())
        )

    @classmethod
    def gen_model_uid(cls) -> str:
        # generate a time-based uuid.
        return str(uuid.uuid1())

    def launch_model(
        self,
        model_name: str,
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        model_uid = self.gen_model_uid()

        coro = self._supervisor_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
            model_size_in_billions=model_size_in_billions,
            model_format=model_format,
            quantization=quantization,
            **kwargs,
        )
        self._isolation.call(coro)

        return model_uid

    def terminate_model(self, model_uid: str):
        coro = self._supervisor_ref.terminate_model(model_uid)
        return self._isolation.call(coro)

    def list_models(self) -> List[Tuple[str, "ModelSpec"]]:
        coro = self._supervisor_ref.list_models()
        return self._isolation.call(coro)

    def get_model(self, model_uid: str) -> "ModelHandle":
        model_spec: "ModelSpec" = self._isolation.call(
            self._supervisor_ref.describe_model(model_uid)
        )
        model_ref = self._isolation.call(self._supervisor_ref.get_model(model_uid))

        if model_spec.model_name == "chatglm" or model_spec.model_name == "chatglm2":
            return ChatglmCppChatModelHandle(model_ref, self._isolation)
        elif model_spec.model_name == "baichuan":
            return LlamaCppModelHandle(model_ref, self._isolation)
        else:
            return LlamaCppChatModelHandle(model_ref, self._isolation)


class ModelHandle:
    """
    A sync model interface which provides type hints that makes it much easier to use xinference
    programmatically.
    """

    def __init__(self, model_ref: xo.ActorRefType["ModelActor"], isolation: Isolation):
        self._model_ref = model_ref
        self._isolation = isolation


class LlamaCppModelHandle(ModelHandle):
    def generate(
        self, prompt: str, generate_config: Optional["LlamaCppGenerateConfig"] = None
    ) -> Union["Completion", Iterator["CompletionChunk"]]:
        coro = self._model_ref.generate(prompt, generate_config)
        return self._isolation.call(coro)


class LlamaCppChatModelHandle(LlamaCppModelHandle):
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List["ChatCompletionMessage"]] = None,
        generate_config: Optional["LlamaCppGenerateConfig"] = None,
    ) -> Union["ChatCompletion", Iterator["ChatCompletionChunk"]]:
        coro = self._model_ref.chat(
            prompt, system_prompt, chat_history, generate_config
        )
        return self._isolation.call(coro)


class ChatglmCppChatModelHandle(ModelHandle):
    def chat(
        self,
        prompt: str,
        chat_history: Optional[List["ChatCompletionMessage"]] = None,
        generate_config: Optional["ChatglmCppGenerateConfig"] = None,
    ) -> Union["ChatCompletion", Iterator["ChatCompletionChunk"]]:
        coro = self._model_ref.chat(prompt, chat_history, generate_config)
        return self._isolation.call(coro)


class RESTfulClient:
    def __init__(self, base_url):
        self.base_url = base_url

    @classmethod
    def gen_model_uid(cls) -> str:
        # generate a time-based uuid.
        return str(uuid.uuid1())

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        url = f"{self.base_url}/v1/models"

        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to launch model, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data

    def launch_model(
        self,
        model_name: str,
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs,
    ) -> str:
        url = f"{self.base_url}/v1/models"

        model_uid = self.gen_model_uid()
        payload = {
            "model_uid": model_uid,
            "model_name": model_name,
            "model_size_in_billions": model_size_in_billions,
            "model_format": model_format,
            "quantization": quantization,
            "kwargs": kwargs,
        }
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to launch model, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        model_uid = response_data["model_uid"]
        return model_uid

    def terminate_model(self, model_uid: str):
        url = f"{self.base_url}/v1/models/{model_uid}"

        response = requests.delete(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to terminate model, detail: {response.json()['detail']}"
            )

    def _get_supervisor_internal_address(self):
        url = f"{self.base_url}/v1/address"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to get supervisor internal address")
        response_data = response.json()
        return response_data
