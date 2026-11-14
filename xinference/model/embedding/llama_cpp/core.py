# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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
import logging
import os
import platform
import queue
import sys
from typing import Any, Optional, Tuple, Union

from packaging import version

from ....types import Embedding
from ....utils import dependency_import_guard
from ..._llamacpp import configure_gpu_layers
from ...batch import BatchMixin
from ...utils import check_dependency_available
from ..core import EmbeddingModel, EmbeddingModelFamilyV2, EmbeddingSpecV1

logger = logging.getLogger(__name__)


def _gguf_declares_pooling(model_path: Optional[str]) -> bool:
    if not model_path:
        return False
    from ...llm.gguf_model_metadata import read_gguf_config

    try:
        with open(model_path, "rb") as stream:
            config, _, _ = read_gguf_config(stream, limit=64 * 1024**2)
    except (OSError, ValueError):
        return False
    return "pooling_type" in config


class _Done:
    pass


class _Error:
    def __init__(self, msg):
        self.msg = msg


def _error_message(msg: Any) -> str:
    if isinstance(msg, dict):
        for key in ("message", "msg", "error"):
            value = msg.get(key)
            if value:
                return _error_message(value)
    return str(msg)


def _get_error_payload(response: Any) -> Optional[Any]:
    if not isinstance(response, dict):
        return None
    if response.get("error"):
        return response["error"]
    if response.get("code"):
        return response
    return None


class XllamaCppEmbeddingModel(EmbeddingModel, BatchMixin):
    supports_dimensions = False

    def __init__(self, *args, **kwargs) -> None:
        EmbeddingModel.__init__(self, *args, **kwargs)
        BatchMixin.__init__(self, self.create_embedding, **kwargs)  # type: ignore
        self._llm = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        llamacpp_model_config = self._kwargs.get("llamacpp_model_config")
        if llamacpp_model_config is None:
            # Launch-time llama.cpp options arrive as flat kwargs from the CLI / REST API.
            # Keep BatchMixin options out of the llama.cpp config.
            llamacpp_model_config = {
                k: v
                for k, v in self._kwargs.items()
                if k not in ("llamacpp_model_config", "batch_size", "batch_interval")
            }
        self._llamacpp_model_config = self._sanitize_model_config(llamacpp_model_config)

    def _sanitize_model_config(self, llamacpp_model_config: Optional[dict]) -> dict:
        if llamacpp_model_config is None:
            llamacpp_model_config = {}
        else:
            # Copy so defaults are not written back into the caller's config.
            llamacpp_model_config = dict(llamacpp_model_config)

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

    def _configure_params(self, params: Any, model_path: Optional[str] = None) -> None:
        from xllamacpp import llama_pooling_type

        params.n_parallel = min(8, os.cpu_count() or 1)
        # llama.cpp applies the pooling a GGUF declares (e.g. CLS for bge-m3).
        # GGUFs that declare none (e.g. the pinned Qwen3-Embedding files) would
        # get no pooling at all, so keep last-token pooling for those.
        if not _gguf_declares_pooling(model_path):
            params.pooling_type = llama_pooling_type.LLAMA_POOLING_TYPE_LAST

    def load(self):
        # add truncate_dim args hint
        if "dimensions" in self._kwargs and not self.supports_dimensions:
            raise NotImplementedError(
                "LlamaCpp embedder does not support dimensions argument now."
            )
        with dependency_import_guard("xllamacpp"):
            from xllamacpp import CommonParams, Server, __version__

            try:
                if version.parse(__version__) < version.parse("0.2.0"):
                    raise RuntimeError(
                        "Please update xllamacpp to >= 0.2.0 by `pip install -U xllamacpp`"
                    )
            except version.InvalidVersion:
                pass  # If the version parse failed, we just skip the version check.

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

            projector = getattr(self._model_spec, "multimodal_projector", None)
            projector_path = (
                os.path.join(self._model_path, projector) if projector else ""
            )
            if projector_path:
                params.mmproj.path = projector_path
            # Defaults may be overwritten by _llamacpp_model_config.
            self._configure_params(params, model_path)
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
            configure_gpu_layers(
                params, model_path, [projector_path] if projector_path else []
            )

            self._llm = Server(params)
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(10, n_threads)
            )
        except AssertionError:
            raise RuntimeError(f"Load model {self._model_name} failed")

    def _create_embedding(self, sentences: Any, **kwargs) -> Embedding:
        if self._llm is None:
            raise RuntimeError("Model is not loaded.")

        q: queue.Queue = queue.Queue()
        if isinstance(sentences, str):
            sentences = [sentences]

        def _handle_embedding():
            data = {"input": sentences}
            model_uid: Optional[str] = kwargs.pop("model_uid", None)
            if model_uid:
                data["model"] = model_uid
            try:
                res = self._llm.handle_embeddings(data)
                error = _get_error_payload(res)
                q.put(_Error(error) if error else res)
            except Exception as ex:
                q.put(_Error(str(ex)))
            q.put(_Done)

        assert self._executor
        self._executor.submit(_handle_embedding)

        r = q.get()
        if type(r) is _Error:
            raise Exception(_error_message(r.msg))
        r["model_replica"] = self._model_uid
        return Embedding(**r)  # type: ignore

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        dep_check = check_dependency_available("xllamacpp", "xllamacpp")
        if dep_check != True:
            return dep_check
        return True

    @classmethod
    def match_json(
        cls,
        model_family: EmbeddingModelFamilyV2,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ) -> Union[bool, Tuple[bool, str]]:
        if model_spec.model_format not in ["ggufv2"]:
            return False, "llama.cpp embedding engine only supports ggufv2 format"
        if model_family.model_name == "embeddinggemma-2":
            return False
        return True
