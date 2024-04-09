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
import re
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Union

import orjson
import xoscar as xo

from ...core.model import ModelActor
from ...core.supervisor import SupervisorActor
from ...isolation import Isolation
from ..restful.restful_client import Client

if TYPE_CHECKING:
    import PIL

    from ...types import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessage,
        ChatglmCppGenerateConfig,
        Completion,
        CompletionChunk,
        ImageList,
        LlamaCppGenerateConfig,
        PytorchGenerateConfig,
    )


class SSEEvent(object):
    # https://github.com/btubbs/sseclient/blob/master/sseclient.py
    sse_line_pattern = re.compile("(?P<name>[^:]*):?( ?(?P<value>.*))?")

    def __init__(self, data="", event="message", id=None, retry=None):
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry

    @classmethod
    def parse(cls, raw):
        """
        Given a possibly-multiline string representing an SSE message, parse it
        and return a Event object.
        """
        msg = cls()
        for line in raw.splitlines():
            m = cls.sse_line_pattern.match(line)
            if m is None:
                # Malformed line.  Discard but warn.
                continue

            name = m.group("name")
            if name == "":
                # line began with a ":", so is a comment.  Ignore
                continue
            value = m.group("value")

            if name == "data":
                # If we already have some data, then join to it with a newline.
                # Else this is it.
                if msg.data:
                    msg.data = "%s\n%s" % (msg.data, value)
                else:
                    msg.data = value
            elif name == "event":
                msg.event = value
            elif name == "id":
                msg.id = value
            elif name == "retry":
                msg.retry = int(value)

        return msg


class ModelHandle:
    """
    A sync model interface (for rpc client) which provides type hints that makes it much easier to use xinference
    programmatically.
    """

    def __init__(self, model_ref: xo.ActorRefType["ModelActor"], isolation: Isolation):
        self._model_ref = model_ref
        self._isolation = isolation


class ClientIteratorWrapper(AsyncIterator):
    def __init__(self, iterator_wrapper):
        self._iw = iterator_wrapper

    def __aiter__(self):
        return self

    async def __anext__(self):
        r = await self._iw.__anext__()
        text = r.decode("utf-8")
        return orjson.loads(SSEEvent.parse(text).data)


class EmbeddingModelHandle(ModelHandle):
    def create_embedding(self, input: Union[str, List[str]], **kwargs) -> bytes:
        """
        Creates an embedding vector representing the input text.

        Parameters
        ----------
        input: Union[str, List[str]]
            Input text to embed, encoded as a string or array of tokens.
            To embed multiple inputs in a single request, pass an array of strings or array of token arrays.

        Returns
        -------
        bytes
            A json bytes of Embedding. The resulted Embedding vector that can be easily consumed by
            machine learning models and algorithms.
        """

        coro = self._model_ref.create_embedding(input, **kwargs)
        return orjson.loads(self._isolation.call(coro))


class RerankModelHandle(ModelHandle):
    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        **kwargs,
    ):
        """
        Returns an ordered list of documents ordered by their relevance to the provided query.

        Parameters
        ----------
        query: str
            The search query
        documents: List[str]
            The documents to rerank
        top_n: int
            The number of results to return, defaults to returning all results
        max_chunks_per_doc: int
            The maximum number of chunks derived from a document
        return_documents: bool
            if return documents
        Returns
        -------
        Scores
           The scores of documents ordered by their relevance to the provided query

        """
        coro = self._model_ref.rerank(
            documents, query, top_n, max_chunks_per_doc, return_documents, **kwargs
        )
        results = orjson.loads(self._isolation.call(coro))
        for r in results["results"]:
            r["document"] = documents[r["index"]]
        return results


class GenerateModelHandle(ModelHandle):
    def generate(
        self,
        prompt: str,
        generate_config: Optional[
            Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]
        ] = None,
    ) -> Union["Completion", AsyncIterator["CompletionChunk"]]:
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
        r = self._isolation.call(coro)
        if isinstance(r, bytes):
            return orjson.loads(r)
        return ClientIteratorWrapper(r)


class ChatModelHandle(GenerateModelHandle):
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List["ChatCompletionMessage"]] = None,
        generate_config: Optional[
            Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]
        ] = None,
    ) -> Union["ChatCompletion", AsyncIterator["ChatCompletionChunk"]]:
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
        r = self._isolation.call(coro)
        if isinstance(r, bytes):
            return orjson.loads(r)
        return ClientIteratorWrapper(r)


class ChatglmCppChatModelHandle(ModelHandle):
    def chat(
        self,
        prompt: str,
        chat_history: Optional[List["ChatCompletionMessage"]] = None,
        generate_config: Optional["ChatglmCppGenerateConfig"] = None,
    ) -> Union["ChatCompletion", AsyncIterator["ChatCompletionChunk"]]:
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
        r = self._isolation.call(coro)
        if isinstance(r, bytes):
            return orjson.loads(r)
        return ClientIteratorWrapper(r)


class ImageModelHandle(ModelHandle):
    def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ) -> "ImageList":
        """
        Creates an image by the input text.

        Parameters
        ----------
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        n (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt. Must be between 1 and 10.
        size (`str`, *optional*, defaults to `1024*1024`):
            The width*height in pixels of the generated image. Must be one of 256x256, 512x512, or 1024x1024.
        response_format (`str`, *optional*, defaults to `url`):
            The format in which the generated images are returned. Must be one of url or b64_json.
        Returns
        -------
        ImageList
            A list of image objects.
        """

        coro = self._model_ref.text_to_image(prompt, n, size, response_format, **kwargs)
        return orjson.loads(self._isolation.call(coro))

    def image_to_image(
        self,
        image: "PIL.Image",
        prompt: str,
        negative_prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ) -> "ImageList":
        """
        Creates an image by the input text.

        Parameters
        ----------
        image (`PIL.Image`):
            The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
            specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
            accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
            and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
            `init`, images must be passed as a list such that each element of the list can be correctly batched for
            input to a single ControlNet.
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        n (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt. Must be between 1 and 10.
        size (`str`, *optional*, defaults to `1024*1024`):
            The width*height in pixels of the generated image. Must be one of 256x256, 512x512, or 1024x1024.
        response_format (`str`, *optional*, defaults to `url`):
            The format in which the generated images are returned. Must be one of url or b64_json.
        Returns
        -------
        ImageList
            A list of image objects.
        """

        coro = self._model_ref.image_to_image(
            image, prompt, negative_prompt, n, size, response_format, **kwargs
        )
        return orjson.loads(self._isolation.call(coro))


class ActorClient:
    def __init__(self, endpoint: str):
        restful_client = Client(endpoint)
        self._supervisor_address = restful_client._get_supervisor_internal_address()
        self._isolation = Isolation(asyncio.new_event_loop(), threaded=True)
        self._isolation.start()
        self._supervisor_ref: xo.ActorRefType["SupervisorActor"] = self._isolation.call(
            xo.actor_ref(address=self._supervisor_address, uid=SupervisorActor.uid())
        )

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
        request_limits: Optional[int] = None,
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
        request_limits: Optional[int]
            The number of request limits for this model， default is None.
            ``request_limits=None`` means no limits for this model.
        **kwargs:
            Any other parameters been specified.

        Returns
        -------
        str
            The unique model_uid for the launched model.

        """

        coro = self._supervisor_ref.launch_builtin_model(
            model_uid=None,
            model_name=model_name,
            model_type=model_type,
            model_size_in_billions=model_size_in_billions,
            model_format=model_format,
            quantization=quantization,
            replica=replica,
            n_gpu=n_gpu,
            request_limits=request_limits,
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
        elif desc["model_type"] == "image":
            return ImageModelHandle(model_ref, self._isolation)
        elif desc["model_type"] == "rerank":
            return RerankModelHandle(model_ref, self._isolation)
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
