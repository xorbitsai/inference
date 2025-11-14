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

import importlib.util
import json
import logging
from typing import List, Union

from ....types import Embedding, EmbeddingData, EmbeddingUsage
from ...batch import BatchMixin
from ...utils import cache_clean
from ..core import EmbeddingModel, EmbeddingModelFamilyV2, EmbeddingSpecV1

logger = logging.getLogger(__name__)
SUPPORTED_MODELS_PREFIXES = ["bge", "gte", "text2vec", "m3e", "gte", "qwen3"]


class VLLMEmbeddingModel(EmbeddingModel, BatchMixin):
    def __init__(self, *args, **kwargs):
        EmbeddingModel.__init__(self, *args, **kwargs)
        BatchMixin.__init__(self, self.create_embedding, **kwargs)  # type: ignore
        self._context_length = None

    def load(self):
        try:
            # Handle vLLM-transformers config conflict by setting environment variable
            import os

            os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache_vllm"

            from vllm import LLM

        except ImportError as e:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            # Check if it's a config conflict error
            if "aimv2" in str(e):
                error_message = (
                    "vLLM has a configuration conflict with transformers library"
                )
                installation_guide = [
                    "This is a known issue with certain vLLM and transformers versions.",
                    "Try upgrading transformers or using a different vLLM version.\n",
                ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        except Exception as e:
            # Handle config registration conflicts
            if "aimv2" in str(e) and "already used by a Transformers config" in str(e):
                error_message = (
                    "vLLM has a configuration conflict with transformers library"
                )
                installation_guide = [
                    "This is a known issue with certain vLLM and transformers versions.",
                    "Try: pip install --upgrade transformers vllm\n",
                ]
                raise RuntimeError(f"{error_message}\n\n{''.join(installation_guide)}")
            raise

        if self.model_family.model_name in {
            "Qwen3-Embedding-0.6B",
            "Qwen3-Embedding-4B",
            "Qwen3-Embedding-8B",
        }:
            if "hf_overrides" not in self._kwargs:
                self._kwargs["hf_overrides"] = {
                    "is_matryoshka": True,
                }
            elif isinstance(self._kwargs["hf_overrides"], dict):
                self._kwargs["hf_overrides"].update(
                    is_matryoshka=True,
                )
            elif isinstance(self._kwargs["hf_overrides"], str):
                self._kwargs["hf_overrides"] = json.loads(self._kwargs["hf_overrides"])
                self._kwargs["hf_overrides"].update(
                    is_matryoshka=True,
                )

        # Set appropriate VLLM configuration parameters based on model capabilities
        model_max_tokens = getattr(self.model_family, "max_tokens", 512)

        # Set max_model_len based on model family capabilities with reasonable limits
        max_model_len = min(model_max_tokens, 8192)
        if "max_model_len" not in self._kwargs:
            self._kwargs["max_model_len"] = max_model_len

        # Ensure max_num_batched_tokens is sufficient for large models
        if "max_num_batched_tokens" not in self._kwargs:
            # max_num_batched_tokens should be at least max_model_len
            # Set to a reasonable minimum that satisfies the constraint
            self._kwargs["max_num_batched_tokens"] = max(4096, max_model_len)

        # Configure other reasonable defaults for embedding models
        if "gpu_memory_utilization" not in self._kwargs:
            self._kwargs["gpu_memory_utilization"] = 0.7

        # Use a smaller block size for better compatibility
        if "block_size" not in self._kwargs:
            self._kwargs["block_size"] = 16

        logger.debug(
            f"VLLM configuration for {self.model_family.model_name}: "
            f"max_model_len={self._kwargs.get('max_model_len')}, "
            f"max_num_batched_tokens={self._kwargs.get('max_num_batched_tokens')}"
        )

        self._model = LLM(model=self._model_path, task="embed", **self._kwargs)
        self._tokenizer = self._model.get_tokenizer()

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"  # noqa: E231

    @cache_clean
    def _create_embedding(
        self,
        sentences: Union[str, List[str]],
        **kwargs,
    ):
        from packaging.version import Version
        from vllm import PoolingParams
        from vllm import __version__ as vllm_version

        sentences = self._fix_langchain_openai_inputs(sentences)
        model_uid = kwargs.pop("model_uid", None)

        normalize_embedding = kwargs.get("normalize_embedding", True)
        dimensions = kwargs.get("dimensions", None)

        assert self._model is not None

        # Check and truncate sentences that exceed context_length
        if self._context_length is not None:
            truncated_sentences = []
            for sentence in sentences if isinstance(sentences, list) else [sentences]:
                # Use tokenizer to check token length
                tokens = self._tokenizer(sentence, add_special_tokens=True)
                if len(tokens.input_ids) > self._context_length:
                    # Truncate to maximum length
                    # Truncate one extra token to prevent overflow
                    truncated_tokens = tokens.input_ids[: self._context_length - 1]
                    truncated_sentence = self._tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                    truncated_sentences.append(truncated_sentence)
                    logger.warning(
                        f"Sentence truncated from {len(tokens.input_ids)} tokens to {self._context_length} tokens"
                    )
                else:
                    truncated_sentences.append(sentence)

            # If original input is string, maintain string format
            if isinstance(sentences, str):
                sentences = truncated_sentences[0]
            else:
                sentences = truncated_sentences
        if Version(vllm_version) > Version("0.10.1"):
            pool_params = PoolingParams(
                dimensions=dimensions, normalize=normalize_embedding
            )
        else:
            if not normalize_embedding:
                raise ValueError(
                    f"vLLM version {vllm_version} does not support "
                    f"unnormalized embeddings. "
                    f"Please upgrade to v0.10.1 or later."
                )
            pool_params = PoolingParams(dimensions=dimensions)
        outputs = self._model.embed(
            sentences, use_tqdm=False, pooling_params=pool_params
        )
        embedding_list = []
        all_token_nums = 0
        for index, output in enumerate(outputs):
            embedding_list.append(
                EmbeddingData(
                    index=index, object="embedding", embedding=output.outputs.embedding
                )
            )
            all_token_nums += len(output.prompt_token_ids)
        usage = EmbeddingUsage(
            prompt_tokens=all_token_nums, total_tokens=all_token_nums
        )
        result = Embedding(
            object="list",
            model=model_uid,
            model_replica=self._model_uid,
            data=embedding_list,
            usage=usage,
        )

        return result

    @classmethod
    def check_lib(cls) -> Union[bool, str]:
        return (
            True
            if importlib.util.find_spec("vllm") is not None
            else "vllm library is not installed"
        )

    @classmethod
    def match_json(
        cls,
        model_family: EmbeddingModelFamilyV2,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ) -> Union[bool, str]:
        # Check library availability first
        lib_result = cls.check_lib()
        if lib_result != True:
            return lib_result

        # Check model format compatibility
        if model_spec.model_format not in ["pytorch"]:
            return f"VLLM Embedding engine only supports pytorch format models, got format: {model_spec.model_format}"

        # Check model name prefix matching
        prefix = model_family.model_name.split("-", 1)[0]
        if prefix.lower() not in [p.lower() for p in SUPPORTED_MODELS_PREFIXES]:
            return f"VLLM Embedding engine only supports models with prefixes {SUPPORTED_MODELS_PREFIXES}, got model: {model_family.model_name}"

        # Additional runtime compatibility checks for vLLM version
        try:
            import vllm
            from packaging.version import Version

            vllm_version = Version(vllm.__version__)

            # Check for vLLM version compatibility issues
            if vllm_version >= Version("0.10.0") and vllm_version < Version("0.11.0"):
                # vLLM 0.10.x has V1 engine issues on CPU
                import platform

                if platform.system() == "Darwin" and platform.machine() in [
                    "arm64",
                    "arm",
                ]:
                    # Check if this is likely to run on CPU (most common for testing)
                    return f"vLLM {vllm_version} has compatibility issues with embedding models on Apple Silicon CPUs. Consider using a different platform or vLLM version."
            elif vllm_version >= Version("0.11.0"):
                # vLLM 0.11+ should have fixed the config conflict issue
                pass
        except Exception:
            # If version check fails, continue with basic validation
            pass

        return True

    def wait_for_load(self):
        # set context length after engine inited
        self._set_context_length()

    def _set_context_length(self):
        """Set context length"""
        from vllm import envs

        if not (envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1):
            # v0
            self._context_length = (
                self._model.llm_engine.vllm_config.model_config.max_model_len
            )
        else:
            # v1 - Get max_model_len from the v1 engine configuration
            try:
                # For v1, access the config differently
                if hasattr(self._model.llm_engine, "vllm_config"):
                    self._context_length = (
                        self._model.llm_engine.vllm_config.model_config.max_model_len
                    )
                elif hasattr(self._model.llm_engine, "model_config"):
                    self._context_length = (
                        self._model.llm_engine.model_config.max_model_len
                    )
                else:
                    # Fallback to the configured value
                    self._context_length = self._kwargs.get("max_model_len", 512)
            except Exception as e:
                logger.warning(f"Failed to get context length from vLLM v1 engine: {e}")
                self._context_length = self._kwargs.get("max_model_len", 512)
        logger.debug("Model context length: %s", self._context_length)
