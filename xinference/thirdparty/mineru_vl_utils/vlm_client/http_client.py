import asyncio
import json
import os
import re
from typing import AsyncIterable, Iterable, Sequence

import httpx
from PIL import Image

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    RequestError,
    SamplingParams,
    ServerError,
    VlmClient,
)
from .utils import (
    aio_load_resource,
    gather_tasks,
    get_image_data_url,
    get_png_bytes,
    load_resource,
)


def _get_env(key: str, default: str | None = None) -> str:
    value = os.getenv(key)
    if value not in (None, ""):
        return value
    if default is not None:
        return default
    raise ValueError(f"Environment variable {key} is not set.")


class HttpVlmClient(VlmClient):
    def __init__(
        self,
        model_name: str | None = None,
        server_url: str | None = None,
        server_headers: dict[str, str] | None = None,
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        max_concurrency: int = 100,
        http_timeout: int = 600,
        debug: bool = False,
    ) -> None:
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )
        self.max_concurrency = max_concurrency
        self.http_timeout = http_timeout
        self.debug = debug
        self.headers = server_headers

        if not server_url:
            server_url = _get_env("MINERU_VL_SERVER")

        self.server_url = self._get_base_url(server_url)

        if model_name:
            self._check_model_name(self.server_url, model_name)
            self.model_name = model_name
        else:
            self.model_name = self._get_model_name(self.server_url)

    @property
    def chat_url(self) -> str:
        return f"{self.server_url}/v1/chat/completions"

    def _get_base_url(self, server_url: str) -> str:
        matched = re.match(r"^(https?://[^/]+)", server_url)
        if not matched:
            raise RequestError(f"Invalid server URL: {server_url}")
        return matched.group(1)

    def _check_model_name(self, base_url: str, model_name: str):
        try:
            response = httpx.get(f"{base_url}/v1/models", headers=self.headers, timeout=self.http_timeout)
        except httpx.ConnectError:
            raise ServerError(f"Failed to connect to server {base_url}. Please check if the server is running.")
        if response.status_code != 200:
            raise ServerError(
                f"Failed to get model name from {base_url}. Status code: {response.status_code}, response body: {response.text}"
            )
        for model in response.json().get("data", []):
            if model.get("id") == model_name:
                return
        raise RequestError(
            f"Model '{model_name}' not found in the response from {base_url}/v1/models. "
            "Please check if the model is available on the server."
        )

    def _get_model_name(self, base_url: str) -> str:
        try:
            response = httpx.get(f"{base_url}/v1/models", headers=self.headers, timeout=self.http_timeout)
        except httpx.ConnectError:
            raise ServerError(f"Failed to connect to server {base_url}. Please check if the server is running.")
        if response.status_code != 200:
            raise ServerError(
                f"Failed to get model name from {base_url}. Status code: {response.status_code}, response body: {response.text}"
            )
        models = response.json().get("data", [])
        if not isinstance(models, list):
            raise RequestError(f"No models found in response from {base_url}. Response body: {response.text}")
        if len(models) != 1:
            raise RequestError(
                f"Expected exactly one model from {base_url}, but got {len(models)}. Please specify the model name."
            )
        model_name = models[0].get("id", "")
        if not model_name:
            raise RequestError(f"Model name is empty in response from {base_url}. Response body: {response.text}")
        return model_name

    def build_request_body(
        self,
        system_prompt: str,
        image: bytes,
        prompt: str,
        sampling_params: SamplingParams | None,
        image_format: str | None,
        priority: int | None,
    ) -> dict:
        image_url = get_image_data_url(image, image_format)
        prompt = prompt or self.prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if "<image>" in prompt:
            prompt_1, prompt_2 = prompt.split("<image>", 1)
            user_messages = [
                *([{"type": "text", "text": prompt_1}] if prompt_1.strip() else []),
                # {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                *([{"type": "text", "text": prompt_2}] if prompt_2.strip() else []),
            ]
        elif self.text_before_image:
            user_messages = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:  # image before text, which is the default behavior.
            user_messages = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": user_messages})

        sp = self.build_sampling_params(sampling_params)
        sp_dict = {}
        if sp.temperature is not None:
            sp_dict["temperature"] = sp.temperature
        if sp.top_p is not None:
            sp_dict["top_p"] = sp.top_p
        if sp.top_k is not None:
            sp_dict["top_k"] = sp.top_k
        if sp.presence_penalty is not None:
            sp_dict["presence_penalty"] = sp.presence_penalty
        if sp.frequency_penalty is not None:
            sp_dict["frequency_penalty"] = sp.frequency_penalty
        if sp.repetition_penalty is not None:
            sp_dict["repetition_penalty"] = sp.repetition_penalty
        if sp.no_repeat_ngram_size is not None:
            sp_dict["vllm_xargs"] = {
                "no_repeat_ngram_size": sp.no_repeat_ngram_size,
                "debug": self.debug,
            }
        if sp.max_new_tokens is not None:
            sp_dict["max_completion_tokens"] = sp.max_new_tokens
        sp_dict["skip_special_tokens"] = False

        if self.model_name.lower().startswith("gpt"):
            sp_dict.pop("top_k", None)
            sp_dict.pop("repetition_penalty", None)
            sp_dict.pop("skip_special_tokens", None)

        return {
            "model": self.model_name,
            "messages": messages,
            **({"priority": priority} if priority is not None else {}),
            **sp_dict,
        }

    def get_response_data(self, response: httpx.Response) -> dict:
        if response.status_code != 200:
            raise ServerError(f"Unexpected status code: [{response.status_code}], response body: {response.text}")
        try:
            response_data = response.json()
        except Exception as e:
            raise ServerError(f"Failed to parse response JSON: {e}, response body: {response.text}")
        if not isinstance(response_data, dict):
            raise ServerError(f"Response is not a JSON object: {response.text}")
        return response_data

    def get_response_content(self, response_data: dict) -> str:
        if response_data.get("object") == "error":
            raise ServerError(f"Error from server: {response_data}")
        choices = response_data.get("choices")
        if not (isinstance(choices, list) and choices):
            raise ServerError("No choices found in the response.")
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is None:
            raise ServerError("Finish reason is None in the response.")
        if finish_reason == "length":
            if not self.allow_truncated_content:
                raise RequestError("The response was truncated due to length limit.")
            else:
                print("Warning: The response was truncated due to length limit.")
        elif finish_reason != "stop":
            raise RequestError(f"Unexpected finish reason: {finish_reason}")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ServerError("Message not found in the response.")
        if "content" not in message:
            raise ServerError("Content not found in the message.")
        content = message["content"]
        if not (content is None or isinstance(content, str)):
            raise ServerError(f"Unexpected content type: {type(content)}.")
        return content or ""

    def predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        image_format = None
        if isinstance(image, str):
            image = load_resource(image)
        if isinstance(image, Image.Image):
            image = get_png_bytes(image)
            image_format = "png"

        request_body = self.build_request_body(
            system_prompt=self.system_prompt,
            image=image,
            prompt=prompt,
            sampling_params=sampling_params,
            image_format=image_format,
            priority=priority,
        )

        if self.debug:
            request_text = json.dumps(request_body, ensure_ascii=False)
            if len(request_text) > 4096:
                request_text = request_text[:2048] + "...(truncated)..." + request_text[-2048:]
            print(f"Request body: {request_text}")

        response = httpx.post(
            self.chat_url,
            json=request_body,
            headers=self.headers,
            timeout=self.http_timeout,
        )

        if self.debug:
            print(f"Response status code: {response.status_code}")
            print(f"Response body: {response.text}")

        response_data = self.get_response_data(response)
        return self.get_response_content(response_data)

    def batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        task = self.aio_batch_predict(
            images=images,
            prompts=prompts,
            sampling_params=sampling_params,
            priority=priority,
        )

        if loop is not None:
            return loop.run_until_complete(task)
        else:
            return asyncio.run(task)

    def stream_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> Iterable[str]:
        image_format = None
        if isinstance(image, str):
            image = load_resource(image)
        if isinstance(image, Image.Image):
            image = get_png_bytes(image)
            image_format = "png"

        request_body = self.build_request_body(
            system_prompt=self.system_prompt,
            image=image,
            prompt=prompt,
            sampling_params=sampling_params,
            image_format=image_format,
            priority=priority,
        )
        request_body["stream"] = True

        if self.debug:
            request_text = json.dumps(request_body, ensure_ascii=False)
            if len(request_text) > 4096:
                request_text = request_text[:2048] + "...(truncated)..." + request_text[-2048:]
            print(f"Request body: {request_text}")

        with httpx.stream(
            "POST",
            self.chat_url,
            json=request_body,
            headers=self.headers,
            timeout=self.http_timeout,
        ) as response:
            for chunk in response.iter_lines():
                chunk = chunk.strip()
                if not chunk.startswith("data:"):
                    continue
                chunk = chunk[5:].lstrip()
                if chunk == "[DONE]":
                    break
                response_data = json.loads(chunk)
                choices = response_data.get("choices") or []
                choice = choices[0] if choices else {}
                delta = choice.get("delta") or {}
                if "content" in delta:
                    yield delta["content"]

    def stream_test(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> None:
        """
        Test the streaming functionality by printing the output.
        """
        print("[Streaming Output]", flush=True)
        for chunk in self.stream_predict(
            image=image,
            prompt=prompt,
            sampling_params=sampling_params,
            priority=priority,
        ):
            print(chunk, end="", flush=True)
        print("\n[End of Streaming Output]", flush=True)

    async def aio_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> str:
        image_format = None
        if isinstance(image, str):
            image = await aio_load_resource(image)
        if isinstance(image, Image.Image):
            image = get_png_bytes(image)
            image_format = "png"

        request_body = self.build_request_body(
            system_prompt=self.system_prompt,
            image=image,
            prompt=prompt,
            sampling_params=sampling_params,
            image_format=image_format,
            priority=priority,
        )

        if self.debug:
            request_text = json.dumps(request_body, ensure_ascii=False)
            if len(request_text) > 4096:
                request_text = request_text[:2048] + "...(truncated)..." + request_text[-2048:]
            print(f"Request body: {request_text}")

        if async_client is None:
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                response = await client.post(self.chat_url, json=request_body, headers=self.headers)
                response_data = self.get_response_data(response)
        else:
            response = await async_client.post(self.chat_url, json=request_body, headers=self.headers)
            response_data = self.get_response_data(response)

        if self.debug:
            print(f"Response status code: {response.status_code}")
            print(f"Response body: {response.text}")

        return self.get_response_content(response_data)

    async def aio_batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm=False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        assert len(priority) == len(images), "Length of priority and images must match."

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        async def predict_with_semaphore(
            image: Image.Image | bytes | str,
            prompt: str,
            sampling_params: SamplingParams | None,
            priority: int | None,
            async_client: httpx.AsyncClient,
        ):
            async with semaphore:
                return await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    priority=priority,
                    async_client=async_client,
                )

        async with httpx.AsyncClient(
            timeout=self.http_timeout,
            headers=self.headers,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
        ) as client:
            return await gather_tasks(
                tasks=[
                    predict_with_semaphore(*args, client)
                    for args in zip(
                        images,
                        prompts,
                        sampling_params,
                        priority,
                    )
                ],
                use_tqdm=use_tqdm,
                tqdm_desc=tqdm_desc,
            )

    async def aio_batch_predict_as_iter(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> AsyncIterable[tuple[int, str]]:
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        assert len(priority) == len(images), "Length of priority and images must match."

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        async def predict_with_semaphore(
            idx: int,
            image: Image.Image | bytes | str,
            prompt: str,
            sampling_params: SamplingParams | None,
            priority: int | None,
            async_client: httpx.AsyncClient,
        ):
            async with semaphore:
                output = await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    priority=priority,
                    async_client=async_client,
                )
                return (idx, output)

        async with httpx.AsyncClient(
            timeout=self.http_timeout,
            headers=self.headers,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=20),
        ) as client:
            pending: set[asyncio.Task[tuple[int, str]]] = set()

            for idx, args in enumerate(zip(images, prompts, sampling_params, priority)):
                pending.add(asyncio.create_task(predict_with_semaphore(idx, *args, client)))

            while len(pending) > 0:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    yield task.result()

    # async def aio_stream_predict(
    #     self,
    #     image: Image.Image | bytes | str,
    #     prompt: str = "",
    #     temperature: Optional[float] = None,
    #     top_p: Optional[float] = None,
    #     top_k: Optional[int] = None,
    #     repetition_penalty: Optional[float] = None,
    #     presence_penalty: Optional[float] = None,
    #     no_repeat_ngram_size: Optional[int] = None,
    #     max_new_tokens: Optional[int] = None,
    # ) -> AsyncIterable[str]:
    #     prompt = self.build_prompt(prompt)

    #     sampling_params = self.build_sampling_params(
    #         temperature=temperature,
    #         top_p=top_p,
    #         top_k=top_k,
    #         repetition_penalty=repetition_penalty,
    #         presence_penalty=presence_penalty,
    #         no_repeat_ngram_size=no_repeat_ngram_size,
    #         max_new_tokens=max_new_tokens,
    #     )

    #     if isinstance(image, str):
    #         image = await aio_load_resource(image)

    #     request_body = self.build_request_body(image, prompt, sampling_params)
    #     request_body["stream"] = True

    #     async with httpx.AsyncClient(timeout=self.http_timeout) as client:
    #         async with client.stream(
    #             "POST",
    #             self.server_url,
    #             json=request_body,
    #         ) as response:
    #             pos = 0
    #             async for chunk in response.aiter_lines():
    #                 if not (chunk or "").startswith("data:"):
    #                     continue
    #                 if chunk == "data: [DONE]":
    #                     break
    #                 data = json.loads(chunk[5:].strip("\n"))
    #                 chunk_text = data["text"][pos:]
    #                 # meta_info = data["meta_info"]
    #                 pos += len(chunk_text)
    #                 yield chunk_text
