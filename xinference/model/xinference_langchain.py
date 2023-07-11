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

from typing import Any, List, Optional

from langchain.llms.base import LLM

from xinference.client import RESTfulClient


class Xinference(LLM):
    client: Any
    model_name: str
    model_size_in_billions: Optional[int]
    model_format: Optional[str]
    quantization: Optional[str]

    def __init__(
        self,
        server_url: str,
        model_name: str,
        model_size_in_billions: Optional[int],
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            **{
                "model_name": model_name,
                "server_url": server_url,
                "llm_kwargs": kwargs,
            }
        )
        self.client = RESTfulClient(server_url)
        self.model_name = model_name
        self.model_size_in_billions = model_size_in_billions or None
        self.model_format = model_format or None
        self.quantization = quantization or None

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "xinference"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        model_uid = self.client.launch_model(
            self.model_name,
            self.model_size_in_billions,
            self.model_format,
            self.quantization,
            **kwargs,
        )
        completion = self.client.generate(
            model_uid=model_uid, prompt=prompt, kwargs=kwargs
        )
        return completion["choices"][0]["text"]


if __name__ == "__main__":
    llm = Xinference(
        server_url="http://0.0.0.0:9997",
        model_name="orca",
        model_size_in_billions=3,
        quantization="q4_0",
    )
    answer = llm("Q: what is the capital of France? A:")
    print(answer)
