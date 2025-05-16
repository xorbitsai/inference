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
import concurrent.futures
import importlib.util
import logging
import os
import queue
from typing import Iterator, List, Optional, Union

import orjson

from ....types import ChatCompletion, ChatCompletionChunk, Completion, CompletionChunk
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import ChatModelMixin

logger = logging.getLogger(__name__)


class _Done:
    pass


class _Error:
    def __init__(self, msg):
        self.msg = msg


class XllamaCppModel(LLM, ChatModelMixin):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        llamacpp_model_config: Optional[dict] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._llamacpp_model_config = self._sanitize_model_config(llamacpp_model_config)
        self._llm = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def _sanitize_model_config(self, llamacpp_model_config: Optional[dict]) -> dict:
        if llamacpp_model_config is None:
            llamacpp_model_config = {}

        if self.model_family.context_length:
            llamacpp_model_config.setdefault("n_ctx", self.model_family.context_length)
        llamacpp_model_config.setdefault("use_mmap", False)
        llamacpp_model_config.setdefault("use_mlock", True)

        if (
            "llama-2" in self.model_family.model_name
            and self.model_spec.model_size_in_billions == 70
        ):
            llamacpp_model_config["use_mlock"] = False
            llamacpp_model_config["n_gqa"] = 8

        if self._is_darwin_and_apple_silicon():
            llamacpp_model_config.setdefault("n_gpu_layers", -1)
        elif self._is_linux():
            llamacpp_model_config.setdefault("n_gpu_layers", -1)
        llamacpp_model_config.setdefault("reasoning_content", False)

        return llamacpp_model_config

    @classmethod
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("xllamacpp") is not None

    @classmethod
    def match_json(
        cls, llm_family: LLMFamilyV1, llm_spec: LLMSpecV1, quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["ggufv2"]:
            return False
        if (
            "chat" not in llm_family.model_ability
            and "generate" not in llm_family.model_ability
        ):
            return False
        return True

    def load(self):
        try:
            from xllamacpp import CommonParams, Server
        except ImportError:
            error_message = "Failed to import module 'xllamacpp'"
            installation_guide = ["Please make sure 'xllamacpp' is installed. "]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        reasoning_content = self._llamacpp_model_config.pop("reasoning_content")
        enable_thinking = self._llamacpp_model_config.pop("enable_thinking", True)
        self.prepare_parse_reasoning_content(
            reasoning_content, enable_thinking=enable_thinking
        )

        if os.path.isfile(self.model_path):
            # mostly passed from --model_path
            model_path = self.model_path
        else:
            # handle legacy cache.
            if (
                self.model_spec.model_file_name_split_template
                and self.quantization in self.model_spec.quantization_parts
            ):
                part = self.model_spec.quantization_parts[self.quantization]
                model_path = os.path.join(
                    self.model_path,
                    self.model_spec.model_file_name_split_template.format(
                        quantization=self.quantization, part=part[0]
                    ),
                )
            else:
                model_path = os.path.join(
                    self.model_path,
                    self.model_spec.model_file_name_template.format(
                        quantization=self.quantization
                    ),
                )
                legacy_model_file_path = os.path.join(self.model_path, "model.bin")
                if os.path.exists(legacy_model_file_path):
                    model_path = legacy_model_file_path

        try:
            params = CommonParams()
            # Compatible with xllamacpp changes
            try:
                params.model = model_path
            except Exception:
                params.model.path = model_path
            if self.model_family.chat_template:
                params.chat_template = self.model_family.chat_template
            # This is the default value, could be overwritten by _llamacpp_model_config
            params.n_parallel = os.cpu_count()
            for k, v in self._llamacpp_model_config.items():
                try:
                    if "." in k:
                        parts = k.split(".")
                        sub_param = params
                        for p in parts[:-1]:
                            sub_param = getattr(sub_param, p)
                        setattr(sub_param, parts[-1], v)
                    else:
                        setattr(params, k, v)
                except Exception as e:
                    logger.error("Failed to set the param %s = %s, error: %s", k, v, e)
            n_threads = self._llamacpp_model_config.get("n_threads", os.cpu_count())
            params.cpuparams.n_threads = n_threads
            params.cpuparams_batch.n_threads = n_threads
            if params.n_gpu_layers == -1:
                # Number of layers to offload to GPU (-ngl). If -1, all layers are offloaded.
                # 0x7FFFFFFF is INT32 max, will be auto set to all layers
                params.n_gpu_layers = 0x7FFFFFFF
            self._llm = Server(params)
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(10, n_threads)
            )
        except AssertionError:
            raise RuntimeError(f"Load model {self.model_family.model_name} failed")

    def generate(
        self, prompt: str, generate_config: Optional[dict] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        generate_config = generate_config or {}
        stream = generate_config.get("stream", False)
        q: queue.Queue = queue.Queue()

        def _handle_completion():
            data = generate_config
            data.pop("stopping_criteria", None)
            data.pop("logits_processor", None)
            data.pop("suffix", None)
            data.pop("best_of", None)
            data.update(
                {
                    "prompt": prompt,
                    "stream": stream,
                }
            )
            prompt_json = orjson.dumps(data)

            def _error_callback(err):
                try:
                    msg = orjson.loads(err)
                    q.put(_Error(msg))
                except Exception as e:
                    q.put(_Error(str(e)))

            def _ok_callback(ok):
                try:
                    res = orjson.loads(ok)
                    res["model"] = self.model_uid
                    q.put(res)
                except Exception as e:
                    logger.exception("handle_completions callback failed: %s", e)

            try:
                self._llm.handle_completions(prompt_json, _error_callback, _ok_callback)
            except Exception as ex:
                logger.exception("handle_completions failed: %s", ex)
            q.put(_Done)

        assert self._executor
        self._executor.submit(_handle_completion)

        if stream:

            def _to_iterator():
                while (r := q.get()) is not _Done:
                    if type(r) is _Error:
                        raise Exception("Got error in generate stream: %s", r.msg)
                    yield r

            return _to_iterator()
        else:
            r = q.get()
            if type(r) is _Error:
                raise Exception("Got error in generate: %s", r.msg)
            return r

    def chat(
        self,
        messages: List[dict],
        generate_config: Optional[dict] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        generate_config = generate_config or {}
        stream = generate_config.get("stream", False)
        tools = generate_config.pop("tools", []) if generate_config else None
        q: queue.Queue = queue.Queue()

        def _handle_chat_completion():
            data = generate_config
            data.pop("stopping_criteria", None)
            data.pop("logits_processor", None)
            data.pop("suffix", None)
            data.pop("best_of", None)
            data.update(
                {
                    "messages": messages,
                    "stream": stream,
                    "tools": tools,
                }
            )
            prompt_json = orjson.dumps(data)

            def _error_callback(err):
                try:
                    msg = orjson.loads(err)
                    q.put(_Error(msg))
                except Exception as e:
                    q.put(_Error(str(e)))

            def _ok_callback(ok):
                try:
                    res = orjson.loads(ok)
                    res["model"] = self.model_uid
                    q.put(res)
                except Exception as e:
                    logger.exception("handle_chat_completions callback failed: %s", e)

            try:
                self._llm.handle_chat_completions(
                    prompt_json, _error_callback, _ok_callback
                )
            except Exception as ex:
                logger.exception("handle_chat_completions failed: %s", ex)
            q.put(_Done)

        assert self._executor
        self._executor.submit(_handle_chat_completion)

        if stream:

            def _to_iterator():
                while (r := q.get()) is not _Done:
                    if type(r) is _Error:
                        raise Exception("Got error in chat stream: %s", r.msg)
                    # Get valid keys (O(1) lookup)
                    chunk_keys = ChatCompletionChunk.__annotations__
                    # The chunk may contain additional keys (e.g., system_fingerprint),
                    # which might not conform to OpenAI/DeepSeek formats.
                    # Filter out keys that are not part of ChatCompletionChunk.
                    yield {key: r[key] for key in chunk_keys if key in r}

            return self._to_chat_completion_chunks(
                _to_iterator(), self.reasoning_parser
            )
        else:
            r = q.get()
            if type(r) is _Error:
                raise Exception("Got error in chat: %s", r.msg)
            return self._to_chat_completion(r, self.reasoning_parser)
