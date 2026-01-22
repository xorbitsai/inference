# Copyright 2022-2026 XProbe Inc.
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
import sys
import uuid
from typing import List, Optional

from packaging import version

from ....types import DocumentObj, Meta, Rerank, RerankTokens
from ..core import RerankModel, RerankModelFamilyV2, RerankSpecV1

logger = logging.getLogger(__name__)


class _Done:
    pass


class _Error:
    def __init__(self, msg):
        self.msg = msg


class XllamaCppRerankModel(RerankModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._llm = None
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        llamacpp_model_config = self._kwargs.get("llamacpp_model_config")
        self._llamacpp_model_config = self._sanitize_model_config(llamacpp_model_config)

    def _sanitize_model_config(self, llamacpp_model_config: Optional[dict]) -> dict:
        if llamacpp_model_config is None:
            llamacpp_model_config = {}

        llamacpp_model_config.setdefault("rerank", True)
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
                __version__,
                estimate_gpu_layers,
                get_device_info,
                ggml_backend_dev_type,
                llama_pooling_type,
            )

            try:
                if version.parse(__version__) < version.parse("0.2.2"):
                    raise RuntimeError(
                        "Please update xllamacpp to >= 0.2.2 by `pip install -U xllamacpp`"
                    )
            except version.InvalidVersion:
                pass  # If the version parse failed, we just skip the version check.
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
            params.pooling_type = llama_pooling_type.LLAMA_POOLING_TYPE_RANK
            for k, v in self._llamacpp_model_config.items():
                if k == "rerank":
                    continue
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
                            for i in range(len(estimate.tensor_split)):
                                params.tensor_split[i] = estimate.tensor_split[i]
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
        pass

    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
        **kwargs,
    ) -> Rerank:
        if kwargs:
            raise RuntimeError("Unexpected keyword arguments: {}".format(kwargs))
        assert self._llm is not None
        result = self._llm.handle_rerank({"query": query, "documents": documents})
        if top_n is not None:
            result["results"] = result["results"][:top_n]
        reranked_docs = list(
            map(
                lambda doc: DocumentObj(
                    index=doc["index"],
                    relevance_score=doc["relevance_score"],
                    document=documents[doc["index"]] if return_documents else None,
                ),
                result["results"],
            )
        )
        tokens = result["usage"]["total_tokens"]
        metadata = Meta(
            api_version=None,
            billed_units=None,
            tokens=(
                RerankTokens(input_tokens=tokens, output_tokens=tokens)
                if return_len
                else None
            ),
            warnings=None,
        )
        return Rerank(id=str(uuid.uuid4()), results=reranked_docs, meta=metadata)

    @classmethod
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("xllamacpp") is not None

    @classmethod
    def match_json(
        cls,
        model_family: RerankModelFamilyV2,
        model_spec: RerankSpecV1,
        quantization: str,
    ) -> bool:
        if model_spec.model_format not in ["ggufv2"]:
            return False
        return True
