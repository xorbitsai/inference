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
import gc
import importlib.util
import logging
import threading
import uuid
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ....device_utils import empty_cache
from ....types import Document, DocumentObj, Meta, Rerank, RerankTokens
from ...utils import is_flash_attn_available
from ..core import (
    RERANK_EMPTY_CACHE_COUNT,
    RerankModel,
    RerankModelFamilyV2,
    RerankSpecV1,
)
from ..utils import preprocess_sentence

logger = logging.getLogger(__name__)


class _ModelWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.model = module
        self._local_data = threading.local()

    @property
    def n_tokens(self):
        return getattr(self._local_data, "n_tokens", 0)

    @n_tokens.setter
    def n_tokens(self, value):
        self._local_data.n_tokens = value

    def forward(self, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        # when batching, the attention mask 1 means there is a token
        # thus we just sum up it to get the total number of tokens
        if attention_mask is not None:
            self.n_tokens += attention_mask.sum().item()
        return self.model(**kwargs)

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)


class SentenceTransformerRerankModel(RerankModel):
    def load(self):
        # TODO: Split FlagReranker and sentence_transformers into different model_engines like FlagRerankModel
        logger.info("Loading rerank model: %s", self._model_path)
        enable_flash_attn = self._kwargs.pop(
            "enable_flash_attn", is_flash_attn_available()
        )
        if enable_flash_attn:
            logger.warning(
                "flash_attn can only support fp16 and bf16, will force set `use_fp16` to True"
            )
            self._use_fp16 = True

        if (
            self.model_family.type == "normal"
            and "qwen3" not in self.model_family.model_name.lower()
        ):
            try:
                import sentence_transformers
                from sentence_transformers.cross_encoder import CrossEncoder

                if sentence_transformers.__version__ < "3.1.0":
                    raise ValueError(
                        "The sentence_transformers version must be greater than 3.1.0. "
                        "Please upgrade your version via `pip install -U sentence_transformers` or refer to "
                        "https://github.com/UKPLab/sentence-transformers"
                    )
            except ImportError:
                error_message = "Failed to import module 'sentence-transformers'"
                installation_guide = [
                    "Please make sure 'sentence-transformers' is installed. ",
                    "You can install it by `pip install sentence-transformers`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = CrossEncoder(
                self._model_path,
                device=self._device,
                trust_remote_code=True,
                max_length=getattr(self.model_family, "max_tokens"),
                **self._kwargs,
            )
            if self._use_fp16:
                self._model.model.half()
        elif "qwen3" in self.model_family.model_name.lower():
            # qwen3-reranker
            # now we use transformers
            # TODO: support engines for rerank models
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError:
                error_message = "Failed to import module 'transformers'"
                installation_guide = [
                    "Please make sure 'transformers' is installed. ",
                    "You can install it by `pip install transformers`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, padding_side="left"
            )
            model_kwargs = {"device_map": "auto"}
            if enable_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs["torch_dtype"] = torch.float16
            model_kwargs.update(self._kwargs)
            logger.debug("Loading qwen3 rerank with kwargs %s", model_kwargs)
            model = self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path, **model_kwargs
            ).eval()
            max_length = getattr(self.model_family, "max_tokens")

            prefix = (
                "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
                'and the Instruct provided. Note that the answer can only be "yes" or "no".'
                "<|im_end|>\n<|im_start|>user\n"
            )
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
            suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

            def process_inputs(pairs):
                inputs = tokenizer(
                    pairs,
                    padding=False,
                    truncation="longest_first",
                    return_attention_mask=False,
                    max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
                )
                for i, ele in enumerate(inputs["input_ids"]):
                    inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens
                inputs = tokenizer.pad(
                    inputs, padding=True, return_tensors="pt", max_length=max_length
                )
                for key in inputs:
                    inputs[key] = inputs[key].to(model.device)
                return inputs

            token_false_id = tokenizer.convert_tokens_to_ids("no")
            token_true_id = tokenizer.convert_tokens_to_ids("yes")

            @torch.inference_mode()
            def compute_logits(inputs, **kwargs):
                batch_scores = model(**inputs).logits[:, -1, :]
                true_vector = batch_scores[:, token_true_id]
                false_vector = batch_scores[:, token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()
                return scores

            self.process_inputs = process_inputs
            self.compute_logits = compute_logits
        else:
            try:
                if self.model_family.type == "LLM-based":
                    from FlagEmbedding import FlagLLMReranker as FlagReranker
                elif self.model_family.type == "LLM-based layerwise":
                    from FlagEmbedding import LayerWiseFlagLLMReranker as FlagReranker
                else:
                    raise RuntimeError(
                        f"Unsupported Rank model type: {self.model_family.type}"
                    )
            except ImportError:
                error_message = "Failed to import module 'FlagEmbedding'"
                installation_guide = [
                    "Please make sure 'FlagEmbedding' is installed. ",
                    "You can install it by `pip install FlagEmbedding`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = FlagReranker(self._model_path, use_fp16=self._use_fp16)
        # Wrap transformers model to record number of tokens
        self._model.model = _ModelWrapper(self._model.model)

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
        assert self._model is not None
        if max_chunks_per_doc is not None:
            raise ValueError("rerank hasn't support `max_chunks_per_doc` parameter.")
        logger.info("Rerank with kwargs: %s, model: %s", kwargs, self._model)

        pre_query = preprocess_sentence(
            query, kwargs.get("instruction", None), self.model_family.model_name
        )
        sentence_combinations = [[pre_query, doc] for doc in documents]
        # reset n tokens
        self._model.model.n_tokens = 0
        if (
            self.model_family.type == "normal"
            and "qwen3" not in self.model_family.model_name.lower()
        ):
            logger.debug("Passing processed sentences: %s", sentence_combinations)
            similarity_scores = self._model.predict(
                sentence_combinations,
                convert_to_numpy=False,
                convert_to_tensor=True,
                **kwargs,
            ).cpu()
            if similarity_scores.dtype == torch.bfloat16:
                similarity_scores = similarity_scores.float()
        elif "qwen3" in self.model_family.model_name.lower():

            def format_instruction(instruction, query, doc):
                if instruction is None:
                    instruction = "Given a web search query, retrieve relevant passages that answer the query"
                output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
                    instruction=instruction, query=query, doc=doc
                )
                return output

            # reduce memory usage.
            micro_bs = 4
            similarity_scores = []
            for i in range(0, len(documents), micro_bs):
                sub_docs = documents[i : i + micro_bs]
                pairs = [
                    format_instruction(kwargs.get("instruction", None), query, doc)
                    for doc in sub_docs
                ]
                # Tokenize the input texts
                inputs = self.process_inputs(pairs)
                similarity_scores.extend(self.compute_logits(inputs))
        else:
            # Related issue: https://github.com/xorbitsai/inference/issues/1775
            similarity_scores = self._model.compute_score(
                sentence_combinations, **kwargs
            )

            if not isinstance(similarity_scores, Sequence):
                similarity_scores = [similarity_scores]
            elif (
                isinstance(similarity_scores, list)
                and len(similarity_scores) > 0
                and isinstance(similarity_scores[0], Sequence)
            ):
                similarity_scores = similarity_scores[0]

        sim_scores_argsort = list(reversed(np.argsort(similarity_scores)))
        if top_n is not None:
            sim_scores_argsort = sim_scores_argsort[:top_n]
        if return_documents:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=Document(text=documents[arg]),
                )
                for arg in sim_scores_argsort
            ]
        else:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=None,
                )
                for arg in sim_scores_argsort
            ]
        if return_len:
            input_len = self._model.model.n_tokens
            # Rerank Model output is just score or documents
            # while return_documents = True
            output_len = input_len

        # api_version, billed_units, warnings
        # is for Cohere API compatibility, set to None
        metadata = Meta(
            api_version=None,
            billed_units=None,
            tokens=(
                RerankTokens(input_tokens=input_len, output_tokens=output_len)
                if return_len
                else None
            ),
            warnings=None,
        )

        del similarity_scores
        # clear cache if possible
        self._counter += 1
        if self._counter % RERANK_EMPTY_CACHE_COUNT == 0:
            logger.debug("Empty rerank cache.")
            gc.collect()
            empty_cache()

        return Rerank(id=str(uuid.uuid1()), results=docs, meta=metadata)

    @classmethod
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("sentence_transformers") is not None

    @classmethod
    def match_json(
        cls,
        model_family: RerankModelFamilyV2,
        model_spec: RerankSpecV1,
        quantization: str,
    ) -> bool:
        # As default embedding engine, sentence-transformer support all models
        return model_spec.model_format in ["pytorch"]
