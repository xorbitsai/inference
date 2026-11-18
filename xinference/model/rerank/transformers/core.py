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
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch

from ....utils import dependency_import_guard
from ...utils import allow_trust_remote_code
from ..core import RerankModelFamilyV2, RerankSpecV1
from ..nemotron import (
    NEMOTRON_RERANK_MODEL_NAME,
    check_nemotron_dependencies,
    nemotron_messages,
)
from ..sentence_transformers.core import SentenceTransformerRerankModel
from ..utils import preprocess_sentence


@dataclass(frozen=True)
class _TokenizedDocument:
    original: Any
    inputs: Any


class NemotronTransformersRerankModel(SentenceTransformerRerankModel):
    """Native sequence-classification backend with shared rerank batching/output."""

    def load(self):
        from packaging.version import Version

        with dependency_import_guard("transformers"):
            import transformers
            from transformers import AutoModelForSequenceClassification, AutoProcessor
        if Version(transformers.__version__) < Version("5.20.0"):
            raise ValueError("Nemotron reranker requires transformers>=5.20.0")
        self._kwargs.pop("batch_size", None)
        self._kwargs.pop("batch_interval", None)
        flash_attn = self._kwargs.pop("enable_flash_attn", False)
        model_kwargs = {
            "device_map": self._device or "auto",
            "dtype": torch.float16 if self._use_fp16 else "auto",
        }
        if flash_attn:
            model_kwargs.update(
                attn_implementation="flash_attention_2", dtype=torch.bfloat16
            )
        model_kwargs.update(self._kwargs)
        self._processor = AutoProcessor.from_pretrained(self._model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_path,
            trust_remote_code=allow_trust_remote_code(self.model_family),
            **model_kwargs,
        ).eval()
        self._tokenizer = self._processor.tokenizer

    def _tokenize_pair(self, query: Any, document: Any, instruction: Optional[str]):
        messages = nemotron_messages(query, document)
        if instruction:
            text = "\n\n".join(b["text"] for b in messages[0]["content"])
            messages[0]["content"] = [
                {
                    "type": "text",
                    "text": preprocess_sentence(
                        text, instruction, self.model_family.model_name
                    ),
                }
            ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        max_tokens = self.model_family.max_tokens or 32768
        if inputs["input_ids"].shape[-1] > max_tokens:
            raise ValueError(
                f"Nemotron reranker input exceeds {max_tokens} tokens; shorten the text or resize the image"
            )
        return inputs

    def _prepare_nemotron_batch_documents(
        self, documents: List[Any], query: Any, instruction: Optional[str]
    ) -> List[Any]:
        # Keep processor errors within this request before merging valid pairs.
        return [
            _TokenizedDocument(doc, self._tokenize_pair(query, doc, instruction))
            for doc in documents
        ]

    def _rerank_nemotron(
        self,
        documents: List[Any],
        queries: List[Any],
        return_len: Optional[bool],
        **kwargs,
    ) -> List[float]:
        instruction = kwargs.pop("instruction", None)
        # The native model card scores one pair per forward; API-level batching
        # still coalesces requests and splits their results through the parent.
        kwargs.pop("batch_size", None)
        if kwargs:
            raise RuntimeError("Unexpected keyword arguments: {}".format(kwargs))
        if len(documents) != len(queries):
            raise ValueError(
                "Nemotron reranker query and documents must have equal length"
            )
        self._reset_token_tracking()
        scores = []
        for query, document in zip(queries, documents):
            inputs = (
                document.inputs
                if isinstance(document, _TokenizedDocument)
                else self._tokenize_pair(query, document, instruction)
            )
            count = int(inputs["attention_mask"].sum())
            self._token_tracking_data.input_tokens.append(count)
            self._token_tracking_data.n_tokens += count
            inputs = inputs.to(self._model.device)
            if inputs.get("pixel_values") is not None:
                inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)
            with torch.inference_mode():
                scores.append(
                    float(self._model(**inputs).logits.squeeze().float().item())
                )
        return scores

    @classmethod
    def match_json(
        cls,
        model_family: RerankModelFamilyV2,
        model_spec: RerankSpecV1,
        quantization: str,
    ) -> Union[bool, Tuple[bool, str]]:
        if model_family.model_name != NEMOTRON_RERANK_MODEL_NAME:
            return (
                False,
                "Transformers rerank engine supports Nemotron-3.5-Rerank-8B only",
            )
        if model_spec.model_format != "pytorch":
            return False, "Transformers rerank engine requires pytorch format"
        return check_nemotron_dependencies("transformers")
