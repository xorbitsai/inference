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
from typing import List, Optional, Tuple

import requests
import xoscar as xo

from .core.model import ModelActor
from .core.service import SupervisorActor
from .isolation import Isolation
from .model import ModelSpec

from .model.llm.core import LlamaCppGenerateConfig
from .model.llm.types import ChatCompletionMessage

class Client:
    def __init__(self, supervisor_address: str):
        self._supervisor_address = supervisor_address
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

    def list_models(self) -> List[Tuple[str, ModelSpec]]:
        coro = self._supervisor_ref.list_models()
        return self._isolation.call(coro)

    def get_model(self, model_uid: str) -> xo.ActorRefType["ModelActor"]:
        coro = self._supervisor_ref.get_model(model_uid)
        return self._isolation.call(coro)


class RESTfulClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def list_models(self):
        url = f"{self.base_url}/v1/models"

        response = requests.get(url)
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

        payload = {
            "model_name": model_name,
            "model_size_in_billions": model_size_in_billions,
            "model_format": model_format,
            "quantization": quantization,
            "kwargs": kwargs,
        }
        response = requests.post(url, json=payload)
        response_data = response.json()
        model_uid = response_data["model_uid"]
        return model_uid

    def terminate_model(self, model_uid):
        url = f"{self.base_url}/v1/models/{model_uid}"

        response = requests.delete(url)
        response_data = response.json()
        return response_data

    def generate(
        self, model_uid: str, prompt: str, **kwargs
    ):
        url = f"{self.base_url}/v1/completions"

        request_body = {"model": model_uid, "prompt": prompt, **kwargs}
        response = requests.post(url, json=request_body)
        response_data = response.json()
        return response_data
    
    def chat(
        self, 
        model_uid: str, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None, 
        **kwargs
    ):
        url = f"{self.base_url}/v1/chat/completions"

        if chat_history is None:
            chat_history = []

        if chat_history and chat_history[0]["role"] == "system":
            if system_prompt is not None:
                chat_history[0]["content"] = system_prompt
        else:
            if system_prompt is not None:
                chat_history.insert(0, ChatCompletionMessage(role="system", content=system_prompt))

        chat_history.append(ChatCompletionMessage(role="user", content=prompt))
        request_body = {"model": model_uid, "messages": chat_history, **kwargs}
        response = requests.post(url, json=request_body)
        response_data = response.json()
        return response_data