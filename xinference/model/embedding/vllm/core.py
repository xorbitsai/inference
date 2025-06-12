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
import logging
from typing import List, Union

from ....types import Embedding, EmbeddingData, EmbeddingUsage
from ..core import EmbeddingModel, EmbeddingModelSpec

logger = logging.getLogger(__name__)
SUPPORTED_MODELS_PREFIXES = ["bge", "gte", "text2vec", "m3e", "gte", "Qwen3"]


class VLLMEmbeddingModel(EmbeddingModel):
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

        self._model = LLM(model=self._model_path, task="embed")
        self._tokenizer = self._model.get_tokenizer()

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"

    def create_embedding(
        self,
        sentences: Union[str, List[str]],
        **kwargs,
    ):
        sentences = self._fix_langchain_openai_inputs(sentences)
        model_uid = kwargs.pop("model_uid", None)

        normalize_embedding = kwargs.get("normalize_embedding", True)
        if not normalize_embedding:
            raise ValueError(
                "vllm embedding engine does not support "
                "setting `normalize_embedding=False`"
            )

        assert self._model is not None
        outputs = self._model.embed(sentences, use_tqdm=False)
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
    def match_json(cls, model_spec: EmbeddingModelSpec) -> bool:
        prefix = model_spec.model_name.split("-", 1)[0]
        if prefix in SUPPORTED_MODELS_PREFIXES:
            return True
        return False
