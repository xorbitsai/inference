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
import json
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import requests
import xoscar as xo

from .core.model import ModelActor
from .core.supervisor import SupervisorActor
from .isolation import Isolation

if TYPE_CHECKING:
    from .model.llm.ggml.chatglm import ChatglmCppGenerateConfig
    from .model.llm.ggml.llamacpp import LlamaCppGenerateConfig
    from .model.llm.pytorch.core import PytorchGenerateConfig
    from .types import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessage,
        Completion,
        CompletionChunk,
        Embedding,
    )


class ModelHandle:
    """
    A sync model interface (for rpc client) which provides type hints that makes it much easier to use xinference
    programmatically.
    """

    def __init__(self, model_ref: xo.ActorRefType["ModelActor"], isolation: Isolation):
        self._model_ref = model_ref
        self._isolation = isolation


class EmbeddingModelHandle(ModelHandle):
    def create_embedding(self, input: Union[str, List[str]]) -> "Embedding":
        """
        Creates an embedding vector representing the input text.

        Parameters
        ----------
        input: Union[str, List[str]]
            Input text to embed, encoded as a string or array of tokens.
            To embed multiple inputs in a single request, pass an array of strings or array of token arrays.

        Returns
        -------
        Embedding
            The resulted Embedding vector that can be easily consumed by machine learning models and algorithms.

        """

        coro = self._model_ref.create_embedding(input)
        return self._isolation.call(coro)


class GenerateModelHandle(EmbeddingModelHandle):
    def generate(
        self,
        prompt: str,
        generate_config: Optional[
            Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]
        ] = None,
    ) -> Union["Completion", Iterator["CompletionChunk"]]:
        """
        Creates a completion for the provided prompt and parameters.

        Parameters
        ----------
        prompt: str
            The user's input.
        generate_config: Optional[Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]]
            Additional configurations for completion.
            "LlamaCppGenerateConfig" -> Configuration for ggml model.
            "PytorchGenerateConfig" -> Configuration for pytorch model.

        Returns
        -------
        Union["Completion", Iterator["CompletionChunk"]]
            Stream is a parameter in generate_config.
            When stream is set to True, the function will return Iterator["CompletionChunk"].
            When stream is set to False, the function will return "Completion".

        """

        coro = self._model_ref.generate(prompt, generate_config)
        return self._isolation.call(coro)


class ChatModelHandle(GenerateModelHandle):
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List["ChatCompletionMessage"]] = None,
        generate_config: Optional[
            Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]
        ] = None,
    ) -> Union["ChatCompletion", Iterator["ChatCompletionChunk"]]:
        """
        Given a list of messages comprising a conversation, the model will return a response.

        Parameters
        ----------
        prompt : str
            The user's input.
        Parameters
        ----------
        prompt: str
            The user's input.
        system_prompt: Optional[str]
            The system context provide to Model prior to any chats.
        chat_history: Optional[List["ChatCompletionMessage"]]
            A list of messages comprising the conversation so far.
        generate_config: Optional[Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]]
            Additional configuration for the chat generation.
            "LlamaCppGenerateConfig" -> configuration for ggml model
            "PytorchGenerateConfig" -> configuration for pytorch model

        Returns
        -------
        Union["ChatCompletion", Iterator["ChatCompletionChunk"]]
            Stream is a parameter in generate_config.
            When stream is set to True, the function will return Iterator["ChatCompletionChunk"].
            When stream is set to False, the function will return "ChatCompletion".

        """

        coro = self._model_ref.chat(
            prompt, system_prompt, chat_history, generate_config
        )
        return self._isolation.call(coro)


class ChatglmCppChatModelHandle(EmbeddingModelHandle):
    def chat(
        self,
        prompt: str,
        chat_history: Optional[List["ChatCompletionMessage"]] = None,
        generate_config: Optional["ChatglmCppGenerateConfig"] = None,
    ) -> Union["ChatCompletion", Iterator["ChatCompletionChunk"]]:
        """
        Given a list of messages comprising a conversation, the ChatGLM model will return a response.

        Parameters
        ----------
        prompt: str
            The user's input
        chat_history: Optional[List["ChatCompletionMessage"]]
            A list of messages comprising the conversation so far.
        generate_config: Optional["ChatglmCppGenerateConfig"]
            Additional Configuration for the ChatGLM Model generation.

        Returns
        -------
        Union["ChatCompletion", Iterator["ChatCompletionChunk"]]
            Stream is a parameter in generate_config.
            When stream is set to True, the function will return Iterator["ChatCompletionChunk"].
            When stream is set to False, the function will return "ChatCompletion".

        """

        coro = self._model_ref.chat(prompt, chat_history, generate_config)
        return self._isolation.call(coro)


def streaming_response_iterator(
    response_lines: Iterator[bytes],
) -> Iterator["CompletionChunk"]:
    """
    Create an Iterator to handle the streaming type of generation.

    Parameters
    ----------
    response_lines: Iterator[bytes]
        Generated lines by the Model Generator.

    Returns
    -------
    Iterator["CompletionChunk"]
        Iterator of CompletionChunks generated by models.

    """

    for line in response_lines:
        line = line.strip()
        if line.startswith(b"data:"):
            data = json.loads(line.decode("utf-8").replace("data: ", "", 1))
            yield data


# Duplicate code due to type hint issues
def chat_streaming_response_iterator(
    response_lines: Iterator[bytes],
) -> Iterator["ChatCompletionChunk"]:
    """
    Create an Iterator to handle the streaming type of generation.

    Parameters
    ----------
    response_lines: Iterator[bytes]
        Generated lines by the Model Generator.

    Returns
    -------
    Iterator["ChatCompletionChunk"]
        Iterator of ChatCompletionChunks generated by models.

    """

    for line in response_lines:
        line = line.strip()
        if line.startswith(b"data:"):
            data = json.loads(line.decode("utf-8").replace("data: ", "", 1))
            yield data


class RESTfulModelHandle:
    """
    A sync model interface (for RESTful client) which provides type hints that makes it much easier to use xinference
    programmatically.
    """

    def __init__(self, model_uid: str, base_url: str):
        self._model_uid = model_uid
        self._base_url = base_url


class RESTfulEmbeddingModelHandle(RESTfulModelHandle):
    def create_embedding(self, input: Union[str, List[str]]) -> "Embedding":
        """
        Create an Embedding from user input via RESTful APIs.

        Parameters
        ----------
        input: Union[str, List[str]]
            Input text to embed, encoded as a string or array of tokens.
            To embed multiple inputs in a single request, pass an array of strings or array of token arrays.

        Returns
        -------
        Embedding
           The resulted Embedding vector that can be easily consumed by machine learning models and algorithms.

        Raises
        ------
        RuntimeError
            Report the failure of embeddings and provide the error message.

        """
        url = f"{self._base_url}/v1/embeddings"
        request_body = {"model": self._model_uid, "input": input}
        response = requests.post(url, json=request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to create the embeddings, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data


class RESTfulGenerateModelHandle(RESTfulEmbeddingModelHandle):
    def generate(
        self,
        prompt: str,
        generate_config: Optional[
            Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]
        ] = None,
    ) -> Union["Completion", Iterator["CompletionChunk"]]:
        """
        Creates a completion for the provided prompt and parameters via RESTful APIs.

        Parameters
        ----------
        prompt: str
            The user's message or user's input.
        generate_config: Optional[Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]]
            Additional configuration for the chat generation.
            "LlamaCppGenerateConfig" -> Configuration for ggml model
            "PytorchGenerateConfig" -> Configuration for pytorch model

        Returns
        -------
        Union["Completion", Iterator["CompletionChunk"]]
            Stream is a parameter in generate_config.
            When stream is set to True, the function will return Iterator["CompletionChunk"].
            When stream is set to False, the function will return "Completion".

        Raises
        ------
        RuntimeError
            Fail to generate the completion from the server. Detailed information provided in error message.

        """

        url = f"{self._base_url}/v1/completions"

        request_body: Dict[str, Any] = {"model": self._model_uid, "prompt": prompt}
        if generate_config is not None:
            for key, value in generate_config.items():
                request_body[key] = value

        stream = bool(generate_config and generate_config.get("stream"))

        response = requests.post(url, json=request_body, stream=stream)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to generate completion, detail: {response.json()['detail']}"
            )

        if stream:
            return streaming_response_iterator(response.iter_lines())

        response_data = response.json()
        return response_data


class RESTfulChatModelHandle(RESTfulGenerateModelHandle):
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List["ChatCompletionMessage"]] = None,
        generate_config: Optional[
            Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]
        ] = None,
    ) -> Union["ChatCompletion", Iterator["ChatCompletionChunk"]]:
        """
        Given a list of messages comprising a conversation, the model will return a response via RESTful APIs.

        Parameters
        ----------
        prompt: str
            The user's input.
        system_prompt: Optional[str]
            The system context provide to Model prior to any chats.
        chat_history: Optional[List["ChatCompletionMessage"]]
            A list of messages comprising the conversation so far.
        generate_config: Optional[Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]]
            Additional configuration for the chat generation.
            "LlamaCppGenerateConfig" -> configuration for ggml model
            "PytorchGenerateConfig" -> configuration for pytorch model

        Returns
        -------
        Union["ChatCompletion", Iterator["ChatCompletionChunk"]]
            Stream is a parameter in generate_config.
            When stream is set to True, the function will return Iterator["ChatCompletionChunk"].
            When stream is set to False, the function will return "ChatCompletion".

        Raises
        ------
        RuntimeError
            Report the failure to generate the chat from the server. Detailed information provided in error message.

        """

        url = f"{self._base_url}/v1/chat/completions"

        if chat_history is None:
            chat_history = []

        if chat_history and chat_history[0]["role"] == "system":
            if system_prompt is not None:
                chat_history[0]["content"] = system_prompt

        else:
            if system_prompt is not None:
                chat_history.insert(0, {"role": "system", "content": system_prompt})

        chat_history.append({"role": "user", "content": prompt})

        request_body: Dict[str, Any] = {
            "model": self._model_uid,
            "messages": chat_history,
        }
        if generate_config is not None:
            for key, value in generate_config.items():
                request_body[key] = value

        stream = bool(generate_config and generate_config.get("stream"))
        response = requests.post(url, json=request_body, stream=stream)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to generate chat completion, detail: {response.json()['detail']}"
            )

        if stream:
            return chat_streaming_response_iterator(response.iter_lines())

        response_data = response.json()
        return response_data


class RESTfulChatglmCppChatModelHandle(RESTfulEmbeddingModelHandle):
    def chat(
        self,
        prompt: str,
        chat_history: Optional[List["ChatCompletionMessage"]] = None,
        generate_config: Optional["ChatglmCppGenerateConfig"] = None,
    ) -> Union["ChatCompletion", Iterator["ChatCompletionChunk"]]:
        """
        Given a list of messages comprising a conversation, the ChatGLM model will return a response via RESTful APIs.

        Parameters
        ----------
        prompt: str
            The user's input.
        chat_history: Optional[List["ChatCompletionMessage"]]
            A list of messages comprising the conversation so far.
        generate_config: Optional["ChatglmCppGenerateConfig"]
            Additional configuration for ChatGLM chat generation.

        Returns
        -------
        Union["ChatCompletion", Iterator["ChatCompletionChunk"]]
            Stream is a parameter in generate_config.
            When stream is set to True, the function will return Iterator["ChatCompletionChunk"].
            When stream is set to False, the function will return "ChatCompletion".

        Raises
        ------
        RuntimeError
            Report the failure to generate the chat from the server. Detailed information provided in error message.

        """

        url = f"{self._base_url}/v1/chat/completions"

        if chat_history is None:
            chat_history = []

        chat_history.append({"role": "user", "content": prompt})

        request_body: Dict[str, Any] = {
            "model": self._model_uid,
            "messages": chat_history,
        }

        if generate_config is not None:
            for key, value in generate_config.items():
                request_body[key] = value

        stream = bool(generate_config and generate_config.get("stream"))
        response = requests.post(url, json=request_body, stream=stream)

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to generate chat completion, detail: {response.json()['detail']}"
            )

        if stream:
            return chat_streaming_response_iterator(response.iter_lines())

        response_data = response.json()
        return response_data


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
    def _gen_model_uid(cls) -> str:
        # generate a time-based uuid.
        return str(uuid.uuid1())

    def register_model(self, model_type: str, model: str, persist: bool):
        """
        Register a custom model.

        Parameters
        ----------
        model_type: str
            The type of model.
        model: str
            The model definition. (refer to: https://inference.readthedocs.io/en/latest/models/custom.html)
        persist: bool
        """
        coro = self._supervisor_ref.register_model(model_type, model, persist)
        self._isolation.call(coro)

    def unregister_model(self, model_type: str, model_name: str):
        """
        Unregister a custom model.

        Parameters
        ----------
        model_type: str
            The type of model.
        model_name: str
            The name of the model
        """
        coro = self._supervisor_ref.unregister_model(model_type, model_name)
        self._isolation.call(coro)

    def list_model_registrations(self, model_type: str) -> List[Dict[str, Any]]:
        """
        List models registered on the server.

        Parameters
        ----------
        model_type: str
            The type of the model.

        Returns
        -------
        List[Dict[str, Any]]
            The collection of registered models on the server.
        """
        coro = self._supervisor_ref.list_model_registrations(model_type)
        return self._isolation.call(coro)

    def get_model_registration(
        self, model_type: str, model_name: str
    ) -> Dict[str, Any]:
        """
        Get the model with the model type and model name registered on the server.

        Parameters
        ----------
        model_type: str
            The type of the model.

        model_name: str
            The name of the model.
        Returns
        -------
        List[Dict[str, Any]]
            The collection of registered models on the server.
        """
        coro = self._supervisor_ref.get_model_registration(model_type, model_name)
        return self._isolation.call(coro)

    def launch_model(
        self,
        model_name: str,
        model_type: str = "LLM",
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        replica: int = 1,
        n_gpu: Optional[Union[int, str]] = "auto",
        **kwargs,
    ) -> str:
        """
        Launch the Model based on the parameters on the server.

        Parameters
        ----------
        model_name: str
            The name of model.
        model_type: str
            Type of model.
        model_size_in_billions: Optional[int]
            The size (in billions) of the model.
        model_format: Optional[str]
            The format of the model.
        quantization: Optional[str]
            The quantization of model.
        replica: Optional[int]
            The replica of model, default is 1.
        n_gpu: Optional[Union[int, str]],
            The number of GPUs used by the model, default is "auto".
            ``n_gpu=None`` means cpu only, ``n_gpu=auto`` lets the system automatically determine the best number of GPUs to use.
        **kwargs:
            Any other parameters been specified.

        Returns
        -------
        str
            The unique model_uid for the launched model.

        """
        model_uid = self._gen_model_uid()

        coro = self._supervisor_ref.launch_builtin_model(
            model_uid=model_uid,
            model_name=model_name,
            model_type=model_type,
            model_size_in_billions=model_size_in_billions,
            model_format=model_format,
            quantization=quantization,
            replica=replica,
            n_gpu=n_gpu,
            **kwargs,
        )

        return self._isolation.call(coro)

    def terminate_model(self, model_uid: str):
        """
        Terminate the specific model running on the server.

        Parameters
        ----------
        model_uid: str
            The unique id that identify the model we want.
        """

        coro = self._supervisor_ref.terminate_model(model_uid)
        self._isolation.call(coro)

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve the model specifications from the Server.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            The collection of model specifications with their names on the server.

        """

        coro = self._supervisor_ref.list_models()
        return self._isolation.call(coro)

    def get_model(self, model_uid: str) -> "ModelHandle":
        """
        Launch the Model based on the parameters on the server.

        Parameters
        ----------
        model_uid: str
            The unique id that identify the model.

        Returns
        -------
        ModelHandle
            The corresponding Model Handler based on the Model specified in the uid:
            "ChatglmCppChatModelHandle" -> handler for ChatGLM chat model
            "GenerateModelHandle" -> handle for generate model. e.g. Baichuan.
            "ChatModelHandle" -> handle for chat model. e.g. Baichuan-chat.

        """

        desc: Dict[str, Any] = self._isolation.call(
            self._supervisor_ref.describe_model(model_uid)
        )
        model_ref = self._isolation.call(self._supervisor_ref.get_model(model_uid))
        if desc["model_type"] == "LLM":
            if desc["model_format"] == "ggmlv3" and "chatglm" in desc["model_name"]:
                return ChatglmCppChatModelHandle(model_ref, self._isolation)
            elif "chat" in desc["model_ability"]:
                return ChatModelHandle(model_ref, self._isolation)
            elif "generate" in desc["model_ability"]:
                return GenerateModelHandle(model_ref, self._isolation)
            else:
                raise ValueError(f"Unrecognized model ability: {desc['model_ability']}")
        elif desc["model_type"] == "embedding":
            return EmbeddingModelHandle(model_ref, self._isolation)
        else:
            raise ValueError(f"Unknown model type:{desc['model_type']}")

    def describe_model(self, model_uid: str) -> Dict:
        """
        Get model information.

        Parameters
        ----------
        model_uid: str
            The unique id that identify the model.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - "model_type": str
               the type of the model determined by its function, e.g. "LLM" (Large Language Model)
            - "model_name": str
               the name of the specific LLM model family
            - "model_lang": List[str]
               the languages supported by the LLM model
            - "model_ability": List[str]
               the ability or capabilities of the LLM model
            - "model_description": str
               a detailed description of the LLM model
            - "model_format": str
               the format specification of the LLM model
            - "model_size_in_billions": int
               the size of the LLM model in billions
            - "quantization": str
               the quantization applied to the model
            - "revision": str
               the revision number of the LLM model specification
            - "context_length": int
               the maximum text length the LLM model can accommodate (include all input & output)
        """

        return self._isolation.call(self._supervisor_ref.describe_model(model_uid))


class RESTfulClient:
    def __init__(self, base_url):
        self.base_url = base_url

    @classmethod
    def _gen_model_uid(cls) -> str:
        # generate a time-based uuid.
        return str(uuid.uuid1())

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve the model specifications from the Server.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            The collection of model specifications with their names on the server.

        """

        url = f"{self.base_url}/v1/models"

        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list model, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data

    def launch_model(
        self,
        model_name: str,
        model_type: str = "LLM",
        model_size_in_billions: Optional[int] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        replica: int = 1,
        n_gpu: Optional[Union[int, str]] = "auto",
        **kwargs,
    ) -> str:
        """
        Launch the model based on the parameters on the server via RESTful APIs.

        Parameters
        ----------
        model_name: str
            The name of model.
        model_type: str
            type of model.
        model_size_in_billions: Optional[int]
            The size (in billions) of the model.
        model_format: Optional[str]
            The format of the model.
        quantization: Optional[str]
            The quantization of model.
        replica: Optional[int]
            The replica of model, default is 1.
        n_gpu: Optional[Union[int, str]],
            The number of GPUs used by the model, default is "auto".
            ``n_gpu=None`` means cpu only, ``n_gpu=auto`` lets the system automatically determine the best number of GPUs to use.
        **kwargs:
            Any other parameters been specified.

        Returns
        -------
        str
            The unique model_uid for the launched model.

        """

        url = f"{self.base_url}/v1/models"

        model_uid = self._gen_model_uid()

        payload = {
            "model_uid": model_uid,
            "model_name": model_name,
            "model_type": model_type,
            "model_size_in_billions": model_size_in_billions,
            "model_format": model_format,
            "quantization": quantization,
            "replica": replica,
            "n_gpu": n_gpu,
        }

        for key, value in kwargs.items():
            payload[str(key)] = value

        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to launch model, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        model_uid = response_data["model_uid"]
        return model_uid

    def terminate_model(self, model_uid: str):
        """
        Terminate the specific model running on the server.

        Parameters
        ----------
        model_uid: str
            The unique id that identify the model we want.

        Raises
        ------
        RuntimeError
            Report failure to get the wanted model with given model_uid. Provide details of failure through error message.

        """

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

    def get_model(self, model_uid: str) -> RESTfulModelHandle:
        """
        Launch the model based on the parameters on the server via RESTful APIs.

        Parameters
        ----------
        model_uid: str
            The unique id that identify the model.

        Returns
        -------
        ModelHandle
            The corresponding Model Handler based on the Model specified in the uid:
            "RESTfulChatglmCppChatModelHandle" -> provide handle to ChatGLM Model
            "RESTfulGenerateModelHandle" -> provide handle to basic generate Model. e.g. Baichuan.
            "RESTfulChatModelHandle" -> provide handle to chat Model. e.g. Baichuan-chat.

        Raises
        ------
        RuntimeError
            Report failure to get the wanted model with given model_uid. Provide details of failure through error message.

        """

        url = f"{self.base_url}/v1/models/{model_uid}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get the model description, detail: {response.json()['detail']}"
            )
        desc = response.json()

        if desc["model_type"] == "LLM":
            if desc["model_format"] == "ggmlv3" and "chatglm" in desc["model_name"]:
                return RESTfulChatglmCppChatModelHandle(model_uid, self.base_url)
            elif "chat" in desc["model_ability"]:
                return RESTfulChatModelHandle(model_uid, self.base_url)
            elif "generate" in desc["model_ability"]:
                return RESTfulGenerateModelHandle(model_uid, self.base_url)
            else:
                raise ValueError(f"Unrecognized model ability: {desc['model_ability']}")
        elif desc["model_type"] == "embedding":
            return RESTfulEmbeddingModelHandle(model_uid, self.base_url)
        else:
            raise ValueError(f"Unknown model type:{desc['model_type']}")

    def describe_model(self, model_uid: str):
        """
        Get model information via RESTful APIs.

        Parameters
        ----------
        model_uid: str
            The unique id that identify the model.

        Returns
        -------
        dict
            A dictionary containing the following keys:
            - "model_type": str
               the type of the model determined by its function, e.g. "LLM" (Large Language Model)
            - "model_name": str
               the name of the specific LLM model family
            - "model_lang": List[str]
               the languages supported by the LLM model
            - "model_ability": List[str]
               the ability or capabilities of the LLM model
            - "model_description": str
               a detailed description of the LLM model
            - "model_format": str
               the format specification of the LLM model
            - "model_size_in_billions": int
               the size of the LLM model in billions
            - "quantization": str
               the quantization applied to the model
            - "revision": str
               the revision number of the LLM model specification
            - "context_length": int
               the maximum text length the LLM model can accommodate (include all input & output)

        Raises
        ------
        RuntimeError
            Report failure to get the wanted model with given model_uid. Provide details of failure through error message.

        """

        url = f"{self.base_url}/v1/models/{model_uid}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get the model description, detail: {response.json()['detail']}"
            )
        return response.json()

    def register_model(self, model_type: str, model: str, persist: bool):
        """
        Register a custom model.

        Parameters
        ----------
        model_type: str
            The type of model.
        model: str
            The model definition. (refer to: https://inference.readthedocs.io/en/latest/models/custom.html)
        persist: bool


        Raises
        ------
        RuntimeError
            Report failure to register the custom model. Provide details of failure through error message.
        """
        url = f"{self.base_url}/v1/model_registrations/{model_type}"
        request_body = {"model": model, "persist": persist}
        response = requests.post(url, json=request_body)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to register model, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data

    def unregister_model(self, model_type: str, model_name: str):
        """
        Unregister a custom model.

        Parameters
        ----------
        model_type: str
            The type of model.
        model_name: str
            The name of the model

        Raises
        ------
        RuntimeError
            Report failure to unregister the custom model. Provide details of failure through error message.
        """
        url = f"{self.base_url}/v1/model_registrations/{model_type}/{model_name}"
        response = requests.delete(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to register model, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data

    def list_model_registrations(self, model_type: str) -> List[Dict[str, Any]]:
        """
        List models registered on the server.

        Parameters
        ----------
        model_type: str
            The type of the model.

        Returns
        -------
        List[Dict[str, Any]]
            The collection of registered models on the server.

        Raises
        ------
        RuntimeError
            Report failure to list model registration. Provide details of failure through error message.

        """
        url = f"{self.base_url}/v1/model_registrations/{model_type}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list model registration, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data

    def get_model_registration(
        self, model_type: str, model_name: str
    ) -> Dict[str, Any]:
        """
        Get the model with the model type and model name registered on the server.

        Parameters
        ----------
        model_type: str
            The type of the model.

        model_name: str
            The name of the model.
        Returns
        -------
        List[Dict[str, Any]]
            The collection of registered models on the server.
        """
        url = f"{self.base_url}/v1/model_registrations/{model_type}/{model_name}"
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list model registration, detail: {response.json()['detail']}"
            )

        response_data = response.json()
        return response_data
