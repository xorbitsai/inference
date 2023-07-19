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

from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

if TYPE_CHECKING:
    from xinference.client import RESTfulChatModelHandle, RESTfulGenerateModelHandle
    from xinference.model.llm.core import LlamaCppGenerateConfig
else:
    RESTfulGenerateModelHandle = Any
    RESTfulChatModelHandle = Any
    LlamaCppGenerateConfig = Any


class Xinference(LLM):
    """Wrapper for accessing Xinference's large-scale model inference service.

    To use, you should have the xinference library installed:

    .. code-block:: bash

        pip install xinference

    Check out: https://github.com/xorbitsai/inference

    To run, you need to start a Xinference supervisor on one server and Xinference workers on the other servers

    Example:
        Starting the supervisor:
        .. code-block:: bash

            $ xinference-supervisor

        Starting the worker:
        .. code-block:: bash
            $ xinference-worker

    Then, you can accessing Xinference's model inference service.

    Example:
        .. code-block:: python
        llm = Xinference(
            server_url="http://0.0.0.0:9997",
            model_name="orca",
            model_size_in_billions=3,
            quantization="q4_0",
        )

        llm("Q: what is the capital of France? A:")

    To view all the supported builtin models, run:

    .. code-block:: bash

        $ xinference list --all

    """

    client: Any
    server_url: str
    """Server URL to run the xinference server on"""
    model_name: str
    """Model name to use. See 'xinference list --all' for all builtin models."""
    model_size_in_billions: Optional[int] = None
    """model size in billions"""
    model_format: Optional[str] = None
    """format of the model"""
    quantization: Optional[str] = None
    """quantization of the model"""
    model_kwargs: Optional[dict] = None

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_name: Optional[str] = None,
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        **kwargs: Any,
    ):
        try:
            from xinference.client import RESTfulClient
        except ImportError as e:
            raise ImportError(
                "Could not import xinference. Make sure to install it with "
                "'pip install xinference'"
            ) from e

        super().__init__(
            **{
                "model_name": model_name,
                "server_url": server_url,
                "model_kwargs": kwargs,
            }
        )

        if self.model_name is None:
            raise ValueError(ValueError(f"Please provide the model name"))

        if self.server_url is None:
            raise ValueError(ValueError(f"Please provide server URL"))

        self.client = RESTfulClient(server_url)

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "xinference"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"server_url": self.server_url},
            **{"model_kwargs": _model_kwargs},
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        model_uid = self.client.launch_model(
            self.model_name,
            self.model_size_in_billions,
            self.model_format,
            self.quantization,
            **(self.model_kwargs),
        )

        model = self.client.get_model(model_uid)

        generate_config = kwargs.get("generate_config", {})

        if stop:
            generate_config["stop"] = stop

        if generate_config and generate_config.get("stream") == True:
            combined_text_output = ""
            for token in self._stream(
                model=model,
                prompt=prompt,
                run_manager=run_manager,
                generate_config=generate_config,
            ):
                combined_text_output += token
            return combined_text_output

        else:
            completion = model.generate(prompt=prompt, generate_config=generate_config)
            return completion["choices"][0]["text"]

    def _stream(
        self,
        model: Union[RESTfulGenerateModelHandle, RESTfulChatModelHandle],
        prompt: str,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        generate_config: Optional[LlamaCppGenerateConfig] = None,
    ):
        streaming_response = model.generate(
            prompt=prompt, generate_config=generate_config
        )
        for chunk in streaming_response:
            if isinstance(chunk, dict):
                choices = chunk.get("choices", [])
                if choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        token = choice.get("text")
                        log_probs = choice.get("logprobs")
                        if run_manager:
                            run_manager.on_llm_new_token(
                                token=token, verbose=self.verbose, log_probs=log_probs
                            )
                        yield token


if __name__ == "__main__":
    llm = Xinference(
        server_url="http://0.0.0.0:9997",
        model_name="orca",
        model_size_in_billions=3,
        quantization="q4_0",
        n_ctx=100,
    )
    answer = llm(
        prompt="Q: where we can visit in the capital of France? A:",
        generate_config={"max_tokens": 1024, "stream": True},
    )
    print(answer)
