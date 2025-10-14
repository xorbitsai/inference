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
from ...utils import cache_clean
from ..core import EmbeddingModel, EmbeddingModelFamilyV2, EmbeddingSpecV1

logger = logging.getLogger(__name__)
SUPPORTED_MODELS_PREFIXES = ["bge", "gte", "text2vec", "m3e", "gte", "Qwen3"]


class VLLMEmbeddingModel(EmbeddingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context_length = None

    def load(self):
        try:
            from vllm import LLM

        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
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

        self._model = LLM(model=self._model_path, task="embed", **self._kwargs)
        self._tokenizer = self._model.get_tokenizer()

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"  # noqa: E231

    @cache_clean
    def create_embedding(
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
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("vllm") is not None

    @classmethod
    def match_json(
        cls,
        model_family: EmbeddingModelFamilyV2,
        model_spec: EmbeddingSpecV1,
        quantization: str,
    ) -> bool:
        if model_spec.model_format in ["pytorch"]:
            prefix = model_family.model_name.split("-", 1)[0]
            if prefix in SUPPORTED_MODELS_PREFIXES:
                return True
        return False

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
            # v1
            logger.warning("vLLM v1 is not supported, ignore context length setting")
        logger.debug("Model context length: %s", self._context_length)
