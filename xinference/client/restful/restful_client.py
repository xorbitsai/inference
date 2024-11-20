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
import json
import typing
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import requests

from ..common import streaming_response_iterator

if TYPE_CHECKING:
    from ...types import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionMessage,
        Completion,
        CompletionChunk,
        Embedding,
        ImageList,
        LlamaCppGenerateConfig,
        PytorchGenerateConfig,
        VideoList,
    )


def convert_float_to_int_or_str(model_size: float) -> Union[int, str]:
    """convert float to int or string

    if float can be presented as int, convert it to int, otherwise convert it to string
    """
    if int(model_size) == model_size:
        return int(model_size)
    else:
        return str(model_size)


def _get_error_string(response: requests.Response) -> str:
    try:
        if response.content:
            return response.json()["detail"]
    except Exception:
        pass
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        return str(e)
    return "Unknown error"


@typing.no_type_check
def handle_system_prompts(
    chat_history: List["ChatCompletionMessage"], system_prompt: Optional[str]
) -> List["ChatCompletionMessage"]:
    history_system_prompts = [
        ch["content"] for ch in chat_history if ch["role"] == "system"
    ]
    if system_prompt is not None:
        history_system_prompts.append(system_prompt)

    # remove all the system prompt in the chat_history
    chat_history = list(filter(lambda x: x["role"] != "system", chat_history))
    # insert all system prompts at the beginning
    chat_history.insert(
        0, {"role": "system", "content": ". ".join(history_system_prompts)}
    )
    return chat_history


class RESTfulModelHandle:
    """
    A sync model interface (for RESTful client) which provides type hints that makes it much easier to use xinference
    programmatically.
    """

    def __init__(self, model_uid: str, base_url: str, auth_headers: Dict):
        self._model_uid = model_uid
        self._base_url = base_url
        self.auth_headers = auth_headers


class RESTfulEmbeddingModelHandle(RESTfulModelHandle):
    def create_embedding(self, input: Union[str, List[str]], **kwargs) -> "Embedding":
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
        request_body = {
            "model": self._model_uid,
            "input": input,
        }
        request_body.update(kwargs)
        response = requests.post(url, json=request_body, headers=self.auth_headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to create the embeddings, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data


class RESTfulRerankModelHandle(RESTfulModelHandle):
    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        return_documents: Optional[bool] = None,
        return_len: Optional[bool] = None,
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
        return_len: bool
            if return tokens len
        Returns
        -------
        Scores
           The scores of documents ordered by their relevance to the provided query

        Raises
        ------
        RuntimeError
            Report the failure of rerank and provide the error message.
        """
        url = f"{self._base_url}/v1/rerank"
        request_body = {
            "model": self._model_uid,
            "documents": documents,
            "query": query,
            "top_n": top_n,
            "max_chunks_per_doc": max_chunks_per_doc,
            "return_documents": return_documents,
            "return_len": return_len,
            "kwargs": json.dumps(kwargs),
        }
        request_body.update(kwargs)
        response = requests.post(url, json=request_body, headers=self.auth_headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to rerank documents, detail: {response.json()['detail']}"
            )
        response_data = response.json()
        return response_data


class RESTfulImageModelHandle(RESTfulModelHandle):
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
        prompt: `str` or `List[str]`
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        n: `int`, defaults to 1
            The number of images to generate per prompt. Must be between 1 and 10.
        size: `str`, defaults to `1024*1024`
            The width*height in pixels of the generated image. Must be one of 256x256, 512x512, or 1024x1024.
        response_format: `str`, defaults to `url`
            The format in which the generated images are returned. Must be one of url or b64_json.
        Returns
        -------
        ImageList
            A list of image objects.
        """
        url = f"{self._base_url}/v1/images/generations"
        request_body = {
            "model": self._model_uid,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
            "kwargs": json.dumps(kwargs),
        }
        response = requests.post(url, json=request_body, headers=self.auth_headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to create the images, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def image_to_image(
        self,
        image: Union[str, bytes],
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs,
    ) -> "ImageList":
        """
        Creates an image by the input text.

        Parameters
        ----------
        image: `Union[str, bytes]`
            The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
            specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
            accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
            and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
            `init`, images must be passed as a list such that each element of the list can be correctly batched for
            input to a single ControlNet.
        prompt: `str` or `List[str]`
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        n: `int`, defaults to 1
            The number of images to generate per prompt. Must be between 1 and 10.
        size: `str`, defaults to `1024*1024`
            The width*height in pixels of the generated image. Must be one of 256x256, 512x512, or 1024x1024.
        response_format: `str`, defaults to `url`
            The format in which the generated images are returned. Must be one of url or b64_json.
        Returns
        -------
        ImageList
            A list of image objects.
            :param prompt:
            :param image:
        """
        url = f"{self._base_url}/v1/images/variations"
        params = {
            "model": self._model_uid,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
            "kwargs": json.dumps(kwargs),
        }
        files: List[Any] = []
        for key, value in params.items():
            files.append((key, (None, value)))
        files.append(("image", ("image", image, "application/octet-stream")))
        response = requests.post(url, files=files, headers=self.auth_headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to variants the images, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def inpainting(
        self,
        image: Union[str, bytes],
        mask_image: Union[str, bytes],
        prompt: str,
        negative_prompt: Optional[str] = None,
        n: int = 1,
        size: Optional[str] = None,
        response_format: str = "url",
        **kwargs,
    ) -> "ImageList":
        """
        Inpaint an image by the input text.

        Parameters
        ----------
        image: `Union[str, bytes]`
            an image batch to be inpainted (which parts of the image to
            be masked out with `mask_image` and repainted according to `prompt`). For both numpy array and pytorch
            tensor, the expected value range is between `[0, 1]` If it's a tensor or a list or tensors, the
            expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a list of arrays, the
            expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image latents as `image`, but
            if passing latents directly it is not encoded again.
        mask_image: `Union[str, bytes]`
            representing an image batch to mask `image`. White pixels in the mask
            are repainted while black pixels are preserved. If `mask_image` is a PIL image, it is converted to a
            single channel (luminance) before use. If it's a numpy array or pytorch tensor, it should contain one
            color channel (L) instead of 3, so the expected shape for pytorch tensor would be `(B, 1, H, W)`, `(B,
            H, W)`, `(1, H, W)`, `(H, W)`. And for numpy array would be for `(B, H, W, 1)`, `(B, H, W)`, `(H, W,
            1)`, or `(H, W)`.
        prompt: `str` or `List[str]`
            The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        n: `int`, defaults to 1
            The number of images to generate per prompt. Must be between 1 and 10.
        size: `str`, defaults to None
            The width*height in pixels of the generated image.
        response_format: `str`, defaults to `url`
            The format in which the generated images are returned. Must be one of url or b64_json.
        Returns
        -------
        ImageList
            A list of image objects.
            :param prompt:
            :param image:
        """
        url = f"{self._base_url}/v1/images/inpainting"
        params = {
            "model": self._model_uid,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
            "kwargs": json.dumps(kwargs),
        }
        files: List[Any] = []
        for key, value in params.items():
            files.append((key, (None, value)))
        files.append(("image", ("image", image, "application/octet-stream")))
        files.append(
            ("mask_image", ("mask_image", mask_image, "application/octet-stream"))
        )
        response = requests.post(url, files=files, headers=self.auth_headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to inpaint the images, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def ocr(self, image: Union[str, bytes], **kwargs):
        url = f"{self._base_url}/v1/images/ocr"
        params = {
            "model": self._model_uid,
            "kwargs": json.dumps(kwargs),
        }
        files: List[Any] = []
        for key, value in params.items():
            files.append((key, (None, value)))
        files.append(("image", ("image", image, "application/octet-stream")))
        response = requests.post(url, files=files, headers=self.auth_headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to ocr the images, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data


class RESTfulVideoModelHandle(RESTfulModelHandle):
    def text_to_video(
        self,
        prompt: str,
        n: int = 1,
        **kwargs,
    ) -> "VideoList":
        """
        Creates a video by the input text.

        Parameters
        ----------
        prompt: `str` or `List[str]`
            The prompt or prompts to guide video generation. If not defined, you need to pass `prompt_embeds`.
        n: `int`, defaults to 1
            The number of videos to generate per prompt. Must be between 1 and 10.
        Returns
        -------
        VideoList
            A list of video objects.
        """
        url = f"{self._base_url}/v1/video/generations"
        request_body = {
            "model": self._model_uid,
            "prompt": prompt,
            "n": n,
            "kwargs": json.dumps(kwargs),
        }
        response = requests.post(url, json=request_body, headers=self.auth_headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to create the video, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data


class RESTfulGenerateModelHandle(RESTfulModelHandle):
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
            "LlamaCppGenerateConfig" -> Configuration for llama-cpp-python model
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

        response = requests.post(
            url, json=request_body, stream=stream, headers=self.auth_headers
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to generate completion, detail: {_get_error_string(response)}"
            )

        if stream:
            return streaming_response_iterator(response.iter_lines())

        response_data = response.json()
        return response_data


class RESTfulChatModelHandle(RESTfulGenerateModelHandle):
    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        generate_config: Optional[
            Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]
        ] = None,
    ) -> Union["ChatCompletion", Iterator["ChatCompletionChunk"]]:
        """
        Given a list of messages comprising a conversation, the model will return a response via RESTful APIs.

        Parameters
        ----------
        messages: List[Dict]
            A list of messages comprising the conversation so far.
        tools: Optional[List[Dict]]
            A tool list.
        generate_config: Optional[Union["LlamaCppGenerateConfig", "PytorchGenerateConfig"]]
            Additional configuration for the chat generation.
            "LlamaCppGenerateConfig" -> configuration for llama-cpp-python model
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

        request_body: Dict[str, Any] = {
            "model": self._model_uid,
            "messages": messages,
        }
        if tools is not None:
            request_body["tools"] = tools
        if generate_config is not None:
            for key, value in generate_config.items():
                request_body[key] = value

        stream = bool(generate_config and generate_config.get("stream"))
        response = requests.post(
            url, json=request_body, stream=stream, headers=self.auth_headers
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to generate chat completion, detail: {_get_error_string(response)}"
            )

        if stream:
            return streaming_response_iterator(response.iter_lines())

        response_data = response.json()
        return response_data


class RESTfulAudioModelHandle(RESTfulModelHandle):
    def transcriptions(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = "json",
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        """
        Transcribes audio into the input language.

        Parameters
        ----------

        audio: bytes
            The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg,
            mpga, m4a, ogg, wav, or webm.
        language: Optional[str]
            The language of the input audio. Supplying the input language in ISO-639-1
            (https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) format will improve accuracy and latency.
        prompt: Optional[str]
            An optional text to guide the model's style or continue a previous audio segment.
            The prompt should match the audio language.
        response_format: Optional[str], defaults to json
            The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
        temperature: Optional[float], defaults to 0
            The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic.
            If set to 0, the model will use log probability to automatically increase the temperature
            until certain thresholds are hit.
        timestamp_granularities: Optional[List[str]], default is None.
            The timestamp granularities to populate for this transcription. response_format must be set verbose_json
            to use timestamp granularities. Either or both of these options are supported: word, or segment.
            Note: There is no additional latency for segment timestamps, but generating word timestamps incurs
            additional latency.

        Returns
        -------
            The transcribed text.
        """
        url = f"{self._base_url}/v1/audio/transcriptions"
        params = {
            "model": self._model_uid,
            "language": language,
            "prompt": prompt,
            "response_format": response_format,
            "temperature": temperature,
            "timestamp_granularities[]": timestamp_granularities,
        }
        files: List[Any] = []
        files.append(("file", ("file", audio, "application/octet-stream")))
        response = requests.post(
            url, data=params, files=files, headers=self.auth_headers
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to transcribe the audio, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def translations(
        self,
        audio: bytes,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Optional[str] = "json",
        temperature: Optional[float] = 0,
        timestamp_granularities: Optional[List[str]] = None,
    ):
        """
        Translates audio into English.

        Parameters
        ----------

        audio: bytes
            The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg,
            mpga, m4a, ogg, wav, or webm.
        language: Optional[str]
            The language of the input audio. Supplying the input language in ISO-639-1
            (https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes) format will improve accuracy and latency.
        prompt: Optional[str]
            An optional text to guide the model's style or continue a previous audio segment.
            The prompt should match the audio language.
        response_format: Optional[str], defaults to json
            The format of the transcript output, in one of these options: json, text, srt, verbose_json, or vtt.
        temperature: Optional[float], defaults to 0
            The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random,
            while lower values like 0.2 will make it more focused and deterministic.
            If set to 0, the model will use log probability to automatically increase the temperature
            until certain thresholds are hit.
        timestamp_granularities: Optional[List[str]], default is None.
            The timestamp granularities to populate for this transcription. response_format must be set verbose_json
            to use timestamp granularities. Either or both of these options are supported: word, or segment.
            Note: There is no additional latency for segment timestamps, but generating word timestamps incurs
            additional latency.

        Returns
        -------
            The translated text.
        """
        url = f"{self._base_url}/v1/audio/translations"
        params = {
            "model": self._model_uid,
            "language": language,
            "prompt": prompt,
            "response_format": response_format,
            "temperature": temperature,
            "timestamp_granularities[]": timestamp_granularities,
        }
        files: List[Any] = []
        files.append(("file", ("file", audio, "application/octet-stream")))
        response = requests.post(
            url, data=params, files=files, headers=self.auth_headers
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to translate the audio, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def speech(
        self,
        input: str,
        voice: str = "",
        response_format: str = "mp3",
        speed: float = 1.0,
        stream: bool = False,
        prompt_speech: Optional[bytes] = None,
        **kwargs,
    ):
        """
        Generates audio from the input text.

        Parameters
        ----------

        input: str
            The text to generate audio for. The maximum length is 4096 characters.
        voice: str
            The voice to use when generating the audio.
        response_format: str
            The format to audio in.
        speed: str
            The speed of the generated audio.
        stream: bool
            Use stream or not.
        prompt_speech: bytes
            The audio bytes to be provided to the model.

        Returns
        -------
        bytes
            The generated audio binary.
        """
        url = f"{self._base_url}/v1/audio/speech"
        params = {
            "model": self._model_uid,
            "input": input,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
            "stream": stream,
            "kwargs": json.dumps(kwargs),
        }
        if prompt_speech:
            files: List[Any] = []
            files.append(
                (
                    "prompt_speech",
                    ("prompt_speech", prompt_speech, "application/octet-stream"),
                )
            )
            response = requests.post(
                url, data=params, files=files, headers=self.auth_headers, stream=stream
            )
        else:
            response = requests.post(
                url, json=params, headers=self.auth_headers, stream=stream
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to speech the text, detail: {_get_error_string(response)}"
            )

        if stream:
            return response.iter_content(chunk_size=1024)

        return response.content


class RESTfulFlexibleModelHandle(RESTfulModelHandle):
    def infer(
        self,
        **kwargs,
    ):
        """
        Call flexible model.

        Parameters
        ----------

        kwargs: dict
            The inference arguments.


        Returns
        -------
        bytes
            The inference result.
        """
        url = f"{self._base_url}/v1/flexible/infers"
        params = {
            "model": self._model_uid,
        }
        params.update(kwargs)

        response = requests.post(url, json=params, headers=self.auth_headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to predict, detail: {_get_error_string(response)}"
            )

        return response.content


class Client:
    def __init__(self, base_url, api_key: Optional[str] = None):
        self.base_url = base_url
        self._headers: Dict[str, str] = {}
        self._cluster_authed = False
        self._check_cluster_authenticated()
        if api_key is not None and self._cluster_authed:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def _set_token(self, token: Optional[str]):
        if not self._cluster_authed or token is None:
            return
        self._headers["Authorization"] = f"Bearer {token}"

    def _get_token(self) -> Optional[str]:
        return (
            str(self._headers["Authorization"]).replace("Bearer ", "")
            if "Authorization" in self._headers
            else None
        )

    def _check_cluster_authenticated(self):
        url = f"{self.base_url}/v1/cluster/auth"
        response = requests.get(url)
        # compatible with old version of xinference
        if response.status_code == 404:
            self._cluster_authed = False
        else:
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to get cluster information, detail: {response.json()['detail']}"
                )
            response_data = response.json()
            self._cluster_authed = bool(response_data["auth"])

    def vllm_models(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/models/vllm-supported"
        response = requests.get(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch VLLM models. detail: {response.json()['detail']}"
            )

        try:
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Error parsing JSON response: {e}")

    def login(self, username: str, password: str):
        if not self._cluster_authed:
            return
        url = f"{self.base_url}/token"

        payload = {"username": username, "password": password}

        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to login, detail: {response.json()['detail']}")

        response_data = response.json()
        # Only bearer token for now
        access_token = response_data["access_token"]
        self._headers["Authorization"] = f"Bearer {access_token}"

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve the model specifications from the Server.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            The collection of model specifications with their names on the server.

        """

        url = f"{self.base_url}/v1/models"

        response = requests.get(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list model, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        model_list = response_data["data"]
        return {item["id"]: item for item in model_list}

    def launch_model(
        self,
        model_name: str,
        model_type: str = "LLM",
        model_engine: Optional[str] = None,
        model_uid: Optional[str] = None,
        model_size_in_billions: Optional[Union[int, str, float]] = None,
        model_format: Optional[str] = None,
        quantization: Optional[str] = None,
        replica: int = 1,
        n_gpu: Optional[Union[int, str]] = "auto",
        peft_model_config: Optional[Dict] = None,
        request_limits: Optional[int] = None,
        worker_ip: Optional[str] = None,
        gpu_idx: Optional[Union[int, List[int]]] = None,
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
        model_engine: Optional[str]
            Specify the inference engine of the model when launching LLM.
        model_uid: str
            UID of model, auto generate a UUID if is None.
        model_size_in_billions: Optional[Union[int, str, float]]
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
        peft_model_config: Optional[Dict]
            - "lora_list": A List of PEFT (Parameter-Efficient Fine-Tuning) model and path.
            - "image_lora_load_kwargs": A Dict of lora load parameters for image model
            - "image_lora_fuse_kwargs": A Dict of lora fuse parameters for image model
        request_limits: Optional[int]
            The number of request limits for this model, default is None.
            ``request_limits=None`` means no limits for this model.
        worker_ip: Optional[str]
            Specify the worker ip where the model is located in a distributed scenario.
        gpu_idx: Optional[Union[int, List[int]]]
            Specify the GPU index where the model is located.
        **kwargs:
            Any other parameters been specified.

        Returns
        -------
        str
            The unique model_uid for the launched model.

        """

        url = f"{self.base_url}/v1/models"

        # convert float to int or string since the RESTful API does not accept float.
        if isinstance(model_size_in_billions, float):
            model_size_in_billions = convert_float_to_int_or_str(model_size_in_billions)

        payload = {
            "model_uid": model_uid,
            "model_name": model_name,
            "model_engine": model_engine,
            "peft_model_config": peft_model_config,
            "model_type": model_type,
            "model_size_in_billions": model_size_in_billions,
            "model_format": model_format,
            "quantization": quantization,
            "replica": replica,
            "n_gpu": n_gpu,
            "request_limits": request_limits,
            "worker_ip": worker_ip,
            "gpu_idx": gpu_idx,
        }

        for key, value in kwargs.items():
            payload[str(key)] = value

        response = requests.post(url, json=payload, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to launch model, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data["model_uid"]

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

        response = requests.delete(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to terminate model, detail: {_get_error_string(response)}"
            )

    def _get_supervisor_internal_address(self):
        url = f"{self.base_url}/v1/address"
        response = requests.get(url, headers=self._headers)
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
              - :obj:`xinference.client.handlers.GenerateModelHandle` -> provide handle to basic generate Model. e.g. Baichuan.
              - :obj:`xinference.client.handlers.ChatModelHandle` -> provide handle to chat Model. e.g. Baichuan-chat.


        Raises
        ------
        RuntimeError
            Report failure to get the wanted model with given model_uid. Provide details of failure through error message.

        """

        url = f"{self.base_url}/v1/models/{model_uid}"
        response = requests.get(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get the model description, detail: {_get_error_string(response)}"
            )
        desc = response.json()

        if desc["model_type"] == "LLM":
            if "chat" in desc["model_ability"]:
                return RESTfulChatModelHandle(
                    model_uid, self.base_url, auth_headers=self._headers
                )
            elif "generate" in desc["model_ability"]:
                return RESTfulGenerateModelHandle(
                    model_uid, self.base_url, auth_headers=self._headers
                )
            else:
                raise ValueError(f"Unrecognized model ability: {desc['model_ability']}")
        elif desc["model_type"] == "embedding":
            return RESTfulEmbeddingModelHandle(
                model_uid, self.base_url, auth_headers=self._headers
            )
        elif desc["model_type"] == "image":
            return RESTfulImageModelHandle(
                model_uid, self.base_url, auth_headers=self._headers
            )
        elif desc["model_type"] == "rerank":
            return RESTfulRerankModelHandle(
                model_uid, self.base_url, auth_headers=self._headers
            )
        elif desc["model_type"] == "audio":
            return RESTfulAudioModelHandle(
                model_uid, self.base_url, auth_headers=self._headers
            )
        elif desc["model_type"] == "video":
            return RESTfulVideoModelHandle(
                model_uid, self.base_url, auth_headers=self._headers
            )
        elif desc["model_type"] == "flexible":
            return RESTfulFlexibleModelHandle(
                model_uid, self.base_url, auth_headers=self._headers
            )
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
        response = requests.get(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get the model description, detail: {_get_error_string(response)}"
            )
        return response.json()

    def register_model(
        self,
        model_type: str,
        model: str,
        persist: bool,
        worker_ip: Optional[str] = None,
    ):
        """
        Register a custom model.

        Parameters
        ----------
        model_type: str
            The type of model.
        model: str
            The model definition. (refer to: https://inference.readthedocs.io/en/latest/models/custom.html)
        worker_ip: Optional[str]
            The IP address of the worker on which the model is running.
        persist: bool


        Raises
        ------
        RuntimeError
            Report failure to register the custom model. Provide details of failure through error message.
        """
        url = f"{self.base_url}/v1/model_registrations/{model_type}"
        request_body = {"model": model, "worker_ip": worker_ip, "persist": persist}
        response = requests.post(url, json=request_body, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to register model, detail: {_get_error_string(response)}"
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
        response = requests.delete(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to register model, detail: {_get_error_string(response)}"
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
        response = requests.get(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list model registration, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def list_cached_models(
        self, model_name: Optional[str] = None, worker_ip: Optional[str] = None
    ) -> List[Dict[Any, Any]]:
        """
        Get a list of cached models.
        Parameters
        ----------
        model_name: Optional[str]
            The name of model.
        worker_ip: Optional[str]
            Specify the worker ip where the model is located in a distributed scenario.

        Returns
        -------
        List[Dict[Any, Any]]
            The collection of cached models on the server.

        Raises
        ------
        RuntimeError
            Raised when the request fails, including the reason for the failure.
        """

        url = f"{self.base_url}/v1/cache/models"
        params = {
            "model_name": model_name,
            "worker_ip": worker_ip,
        }
        response = requests.get(url, headers=self._headers, params=params)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list cached model, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        response_data = response_data.get("list")
        return response_data

    def list_deletable_models(
        self, model_version: str, worker_ip: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the cached models with the model path cached on the server.
        Parameters
        ----------
        model_version: str
            The version of the model.
        worker_ip: Optional[str]
            Specify the worker ip where the model is located in a distributed scenario.
        Returns
        -------
        Dict[str, Dict[str,str]]]
            Dictionary with keys "model_name" and values model_file_location.
        """
        url = f"{self.base_url}/v1/cache/models/files"
        params = {
            "model_version": model_version,
            "worker_ip": worker_ip,
        }
        response = requests.get(url, headers=self._headers, params=params)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get paths by model name, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def confirm_and_remove_model(
        self, model_version: str, worker_ip: Optional[str] = None
    ) -> bool:
        """
        Remove the cached models with the model name cached on the server.
        Parameters
        ----------
        model_version: str
            The version of the model.
        worker_ip: Optional[str]
            Specify the worker ip where the model is located in a distributed scenario.
        Returns
        -------
        str
            The response of the server.
        """
        url = f"{self.base_url}/v1/cache/models"
        params = {
            "model_version": model_version,
            "worker_ip": worker_ip,
        }
        response = requests.delete(url, headers=self._headers, params=params)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to remove cached models, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data.get("result", False)

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
        response = requests.get(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to list model registration, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def query_engine_by_model_name(self, model_name: str):
        """
        Get the engine parameters with the model name registered on the server.

        Parameters
        ----------
        model_name: str
            The name of the model.
        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            The supported engine parameters of registered models on the server.
        """
        url = f"{self.base_url}/v1/engines/{model_name}"
        response = requests.get(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to query engine parameters by model name, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def abort_request(self, model_uid: str, request_id: str, block_duration: int = 30):
        """
        Abort a request.
        Abort a submitted request. If the request is finished or not found, this method will be a no-op.
        Currently, this interface is only supported when batching is enabled for models on transformers backend.

        Parameters
        ----------
        model_uid: str
            Model uid.
        request_id: str
            Request id.
        block_duration: int
            The duration to make the request id abort. If set to 0, the abort_request will be immediate, which may
            prevent it from taking effect if it arrives before the request operation.
        Returns
        -------
        Dict
            Return empty dict.
        """
        url = f"{self.base_url}/v1/models/{model_uid}/requests/{request_id}/abort"
        response = requests.post(
            url, headers=self._headers, json={"block_duration": block_duration}
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to abort request, detail: {_get_error_string(response)}"
            )

        response_data = response.json()
        return response_data

    def get_workers_info(self):
        url = f"{self.base_url}/v1/workers"
        response = requests.get(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get workers info, detail: {_get_error_string(response)}"
            )
        response_data = response.json()
        return response_data

    def get_supervisor_info(self):
        url = f"{self.base_url}/v1/supervisor"
        response = requests.get(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get supervisor info, detail: {_get_error_string(response)}"
            )
        response_json = response.json()
        return response_json

    def get_progress(self, request_id: str):
        url = f"{self.base_url}/v1/requests/{request_id}/progress"
        response = requests.get(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get progress, detail: {_get_error_string(response)}"
            )
        response_json = response.json()
        return response_json

    def abort_cluster(self):
        url = f"{self.base_url}/v1/clusters"
        response = requests.delete(url, headers=self._headers)
        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to abort cluster, detail: {_get_error_string(response)}"
            )
        response_json = response.json()
        return response_json
