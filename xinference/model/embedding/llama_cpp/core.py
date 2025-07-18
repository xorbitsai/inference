# Copyright 2022-2025 XProbe Inc.
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
import platform
import pprint
import queue
import sys
from typing import List, Optional, Union

import orjson

from ....types import Embedding
from ..core import EmbeddingModel, EmbeddingModelFamilyV2, EmbeddingSpecV1

logger = logging.getLogger(__name__)


class _Done:
    pass


class _Error:
    def __init__(self, msg):
        self.msg = msg


class XllamaCppEmbeddingModel(EmbeddingModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._llm = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        llamacpp_model_config = self._kwargs.get("llamacpp_model_config")
        self._llamacpp_model_config = self._sanitize_model_config(llamacpp_model_config)

    def _sanitize_model_config(self, llamacpp_model_config: Optional[dict]) -> dict:
        if llamacpp_model_config is None:
            llamacpp_model_config = {}

        llamacpp_model_config.setdefault("embedding", True)
        llamacpp_model_config.setdefault("use_mmap", False)
        llamacpp_model_config.setdefault("use_mlock", True)

        if self._is_darwin_and_apple_silicon():
            llamacpp_model_config.setdefault("n_gpu_layers", -1)
        elif self._is_linux():
            llamacpp_model_config.setdefault("n_gpu_layers", -1)

        return llamacpp_model_config

    def _is_darwin_and_apple_silicon(self):
        return sys.platform == "darwin" and platform.processor() == "arm"

    def _is_linux(self):
        return sys.platform.startswith("linux")

    def load(self):
        try:
            from xllamacpp import (
                CommonParams,
                Server,
                estimate_gpu_layers,
                get_device_info,
                ggml_backend_dev_type,
                llama_pooling_type,
            )
        except ImportError:
            error_message = "Failed to import module 'xllamacpp'"
            installation_guide = ["Please make sure 'xllamacpp' is installed. "]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        # handle legacy cache.
        if (
            self._model_spec.model_file_name_split_template
            and self._quantization in self._model_spec.quantization_parts
        ):
            part = self._model_spec.quantization_parts[self._quantization]
            model_path = os.path.join(
                self._model_path,
                self._model_spec.model_file_name_split_template.format(
                    quantization=self._quantization, part=part[0]
                ),
            )
        else:
            model_path = os.path.join(
                self._model_path,
                self._model_spec.model_file_name_template.format(
                    quantization=self._quantization
                ),
            )

        try:
            params = CommonParams()
            params.embedding = True
            # Compatible with xllamacpp changes
            try:
                params.model = model_path
            except Exception:
                params.model.path = model_path

            # This is the default value, could be overwritten by _llamacpp_model_config
            params.n_parallel = min(8, os.cpu_count() or 1)
            params.pooling_type = llama_pooling_type.LLAMA_POOLING_TYPE_LAST
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
                try:
                    device_info = get_device_info()
                    gpus = [
                        info
                        for info in device_info
                        if info["type"]
                        == ggml_backend_dev_type.GGML_BACKEND_DEVICE_TYPE_GPU
                    ]
                    if gpus:
                        logger.info(
                            "Try to estimate num gpu layers, n_ctx: %s, n_batch: %s, n_parallel: %s, gpus:\n%s",
                            params.n_ctx,
                            params.n_batch,
                            params.n_parallel,
                            pprint.pformat(gpus),
                        )
                        estimate = estimate_gpu_layers(
                            gpus=gpus,
                            model_path=model_path,
                            projectors=[],
                            context_length=params.n_ctx,
                            batch_size=params.n_batch,
                            num_parallel=params.n_parallel,
                            kv_cache_type="",
                        )
                        logger.info("Estimate num gpu layers: %s", estimate)
                        if estimate.tensor_split:
                            params.tensor_split = estimate.tensor_split
                        else:
                            params.n_gpu_layers = estimate.layers
                except Exception as e:
                    logger.exception(
                        "Estimate num gpu layers for llama.cpp backend failed: %s", e
                    )

            self._llm = Server(params)
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(10, n_threads)
            )
        except AssertionError:
            raise RuntimeError(f"Load model {self._model_name} failed")

    def create_embedding(self, sentences: Union[str, List[str]], **kwargs) -> Embedding:
        if self._llm is None:
            raise RuntimeError("Model is not loaded.")

        q: queue.Queue = queue.Queue()
        if isinstance(sentences, str):
            sentences = [sentences]

        def _handle_embedding():
            data = {"input": sentences}
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
                    q.put(res)
                except Exception as e:
                    q.put(_Error(str(e)))

            try:
                self._llm.handle_embeddings(prompt_json, _error_callback, _ok_callback)
            except Exception as ex:
                q.put(_Error(str(ex)))
            q.put(_Done)

        assert self._executor
        self._executor.submit(_handle_embedding)

        r = q.get()
        if type(r) is _Error:
            raise Exception(f"Failed to create embedding: {r.msg}")
        r["model_replica"] = self._model_uid
        return Embedding(**r)  # type: ignore

    @classmethod
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("xllamacpp") is not None

    @classmethod
    def match_json(
        cls,
        model_family: EmbeddingModelFamilyV2,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ) -> bool:
        if model_spec.model_format not in ["ggufv2"]:
            return False
        return True
