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
from collections import defaultdict
from typing import List, Optional, Union, no_type_check

import numpy as np
import torch

from ....types import Dict, Embedding, EmbeddingData, EmbeddingUsage
from ..core import EmbeddingModel, EmbeddingModelSpec

logger = logging.getLogger(__name__)

# Used for check whether the model is cached.
# Init when registering all the builtin models.
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
SENTENCE_TRANSFORMER_MODEL_LIST: List[str] = []


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def load(self):
        # TODO: load model
        try:
            import sentence_transformers
            from sentence_transformers import SentenceTransformer

            if sentence_transformers.__version__ < "3.1.0":
                raise ValueError(
                    "The sentence_transformers version must be greater than 3.1.0. "
                    "Please upgrade your version via `pip install -U sentence_transformers` or refer to "
                    "https://github.com/UKPLab/sentence-transformers"
                )
        except ImportError:
            error_message = "Failed to import module 'SentenceTransformer'"
            installation_guide = [
                "Please make sure 'sentence-transformers' is installed. ",
                "You can install it by `pip install sentence-transformers`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        class XSentenceTransformer(SentenceTransformer):
            def to(self, *args, **kwargs):
                pass

        torch_dtype = None
        if torch_dtype_str := self._kwargs.get("torch_dtype"):
            try:
                torch_dtype = getattr(torch, torch_dtype_str)
                if torch_dtype not in [
                    torch.float16,
                    torch.float32,
                    torch.bfloat16,
                ]:
                    logger.warning(
                        f"Load embedding model with unsupported torch dtype :  {torch_dtype_str}. Using default torch dtype: fp32."
                    )
                    torch_dtype = torch.float32
            except AttributeError:
                logger.warning(
                    f"Load embedding model with  unknown torch dtype '{torch_dtype_str}'. Using default torch dtype: fp32."
                )
                torch_dtype = torch.float32

        if (
            "gte" in self._model_spec.model_name.lower()
            and "qwen2" in self._model_spec.model_name.lower()
        ):
            model_kwargs = {"device_map": "auto"}
            if torch_dtype:
                model_kwargs["torch_dtype"] = torch_dtype
            self._model = XSentenceTransformer(
                self._model_path,
                device=self._device,
                model_kwargs=model_kwargs,
            )
        elif "qwen3" in self._model_spec.model_name.lower():
            # qwen3 embedding
            flash_attn_installed = importlib.util.find_spec("flash_attn") is not None
            flash_attn_enabled = self._kwargs.get("enable_flash_attn", True)
            model_kwargs = {"device_map": "auto"}
            tokenizer_kwargs = {}
            if flash_attn_installed and flash_attn_enabled:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                model_kwargs["torch_dtype"] = "bfloat16"
                tokenizer_kwargs["padding_side"] = "left"
            if torch_dtype:
                model_kwargs["torch_dtype"] = torch_dtype
            logger.debug(
                "Loading qwen3 embedding with model kwargs: %s, tokenizer kwargs: %s",
                model_kwargs,
                tokenizer_kwargs,
            )
            self._model = XSentenceTransformer(
                self._model_path,
                device=self._device,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
            )
        else:
            model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype else None
            self._model = SentenceTransformer(
                self._model_path,
                device=self._device,
                model_kwargs=model_kwargs,
                trust_remote_code=True,
            )

        self._tokenizer = self._model.tokenizer

    def create_embedding(
        self,
        sentences: Union[str, List[str]],
        **kwargs,
    ):
        sentences = self._fix_langchain_openai_inputs(sentences)
        model_uid = kwargs.pop("model_uid", None)

        from sentence_transformers import SentenceTransformer

        kwargs.setdefault("normalize_embeddings", True)
        if kwargs.get("return_sparse", False):
            raise ValueError(
                "`return_sparse` is not supported for `sentence_transformers` backend, "
                "please use `flag` instead"
            )

        # copied from sentence-transformers, and modify it to return tokens num
        @no_type_check
        def encode(
            model: SentenceTransformer,
            sentences: Union[str, List[str]],
            prompt_name: Optional[str] = None,
            prompt: Optional[str] = None,
            batch_size: int = 32,
            show_progress_bar: bool = None,
            output_value: str = "sentence_embedding",
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False,
            **kwargs,
        ):
            """
            Computes sentence embeddings

            :param sentences: the sentences to embed
            :param batch_size: the batch size used for the computation
            :param show_progress_bar: Output a progress bar when encode sentences
            :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
            :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
            :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
            :param device: Which torch.device to use for the computation
            :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

            :return:
               By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
            """
            import torch
            from sentence_transformers.util import batch_to_device
            from tqdm.autonotebook import trange

            model.eval()
            if show_progress_bar is None:
                show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO
                    or logger.getEffectiveLevel() == logging.DEBUG
                )

            if convert_to_tensor:
                convert_to_numpy = False

            if output_value != "sentence_embedding":
                convert_to_tensor = False
                convert_to_numpy = False

            input_was_string = False
            if isinstance(sentences, str) or not hasattr(
                sentences, "__len__"
            ):  # Cast an individual sentence to a list with length 1
                sentences = [sentences]
                input_was_string = True

            if prompt is None:
                if prompt_name is not None:
                    try:
                        prompt = model.prompts[prompt_name]
                    except KeyError:
                        raise ValueError(
                            f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(model.prompts.keys())!r}."
                        )
                elif model.default_prompt_name is not None:
                    prompt = model.prompts.get(model.default_prompt_name, None)
            else:
                if prompt_name is not None:
                    logger.warning(
                        "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                        "Ignoring the `prompt_name` in favor of `prompt`."
                    )

            extra_features = {}
            if prompt is not None:
                sentences = [prompt + sentence for sentence in sentences]

                # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
                # Tracking the prompt length allow us to remove the prompt during pooling
                tokenized_prompt = model.tokenize([prompt])
                if "input_ids" in tokenized_prompt:
                    extra_features["prompt_length"] = (
                        tokenized_prompt["input_ids"].shape[-1] - 1
                    )

            if device is None:
                device = model._target_device

            if (
                "gte" in self._model_spec.model_name.lower()
                and "qwen2" in self._model_spec.model_name.lower()
            ):
                model.to(device)

            all_embeddings = []
            all_token_nums = 0
            length_sorted_idx = np.argsort(
                [-self._text_length(sen) for sen in sentences]
            )
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

            for start_index in trange(
                0,
                len(sentences),
                batch_size,
                desc="Batches",
                disable=not show_progress_bar,
            ):
                sentences_batch = sentences_sorted[
                    start_index : start_index + batch_size
                ]
                features = model.tokenize(sentences_batch)
                features = batch_to_device(features, device)
                features.update(extra_features)
                # when batching, the attention mask 1 means there is a token
                # thus we just sum up it to get the total number of tokens
                if "clip" in self._model_spec.model_name.lower():
                    if "input_ids" in features and hasattr(
                        features["input_ids"], "numel"
                    ):
                        all_token_nums += features["input_ids"].numel()
                    if "pixel_values" in features and hasattr(
                        features["pixel_values"], "numel"
                    ):
                        all_token_nums += features["pixel_values"].numel()
                else:
                    all_token_nums += features["attention_mask"].sum().item()

                with torch.no_grad():
                    out_features = model.forward(features, **kwargs)

                    if output_value == "token_embeddings":
                        embeddings = []
                        for token_emb, attention in zip(
                            out_features[output_value], out_features["attention_mask"]
                        ):
                            last_mask_id = len(attention) - 1
                            while (
                                last_mask_id > 0 and attention[last_mask_id].item() == 0
                            ):
                                last_mask_id -= 1

                            embeddings.append(token_emb[0 : last_mask_id + 1])
                    elif output_value is None:  # Return all outputs
                        embeddings = []
                        for sent_idx in range(len(out_features["sentence_embedding"])):
                            row = {
                                name: out_features[name][sent_idx]
                                for name in out_features
                            }
                            embeddings.append(row)
                    else:  # Sentence embeddings
                        embeddings = out_features[output_value]
                        embeddings = embeddings.detach()
                        if normalize_embeddings:
                            embeddings = torch.nn.functional.normalize(
                                embeddings, p=2, dim=1
                            )

                        # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                        if convert_to_numpy:
                            embeddings = embeddings.cpu()

                    all_embeddings.extend(embeddings)

            all_embeddings = [
                all_embeddings[idx] for idx in np.argsort(length_sorted_idx)
            ]

            if convert_to_tensor:
                if len(all_embeddings):
                    all_embeddings = torch.stack(all_embeddings)
                else:
                    all_embeddings = torch.Tensor()
            elif convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

            if input_was_string:
                all_embeddings = all_embeddings[0]

            return all_embeddings, all_token_nums

        # seems already support prompt in embedding model
        if (
            "gte" in self._model_spec.model_name.lower()
            and "qwen2" in self._model_spec.model_name.lower()
        ):
            all_embeddings, all_token_nums = encode(
                self._model,
                sentences,
                prompt_name="query",
                convert_to_numpy=False,
                **kwargs,
            )
        elif "clip" in self._model_spec.model_name.lower():
            import base64
            import re
            from io import BytesIO

            from PIL import Image

            def base64_to_image(base64_str: str) -> Image.Image:
                # base64_data = re.sub("^data:image/.+;base64,", "", base64_str)
                base64_data = base64_str.split(",", 1)[1]
                byte_data = base64.b64decode(base64_data)
                image_data = BytesIO(byte_data)
                img = Image.open(image_data)
                return img

            objs: list[str] = []
            if isinstance(sentences, str):
                objs.append(sentences)
            else:
                for item in sentences:
                    if isinstance(item, dict):
                        if item.get("text") is not None:
                            objs.append(item["text"])
                        elif item.get("image") is not None:
                            if re.match(r"^data:image/.+;base64,", item["image"]):
                                image = base64_to_image(item["image"])
                                objs.append(image)
                            else:
                                objs.append(item["image"])
                        else:
                            raise ValueError("Please check the input data.")
                    elif isinstance(item, str):
                        objs.append(item)
                    else:
                        raise ValueError("Please check the input data.")

            all_embeddings, all_token_nums = encode(
                self._model,
                objs,
                convert_to_numpy=False,
                **kwargs,
            )
        else:
            all_embeddings, all_token_nums = encode(
                self._model,
                sentences,
                convert_to_numpy=False,
                **kwargs,
            )
        if isinstance(sentences, str):
            all_embeddings = [all_embeddings]
        embedding_list = []
        for index, data in enumerate(all_embeddings):
            embedding_list.append(
                EmbeddingData(index=index, object="embedding", embedding=data.tolist())
            )
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

        # clean cache if possible
        self._clean_cache_if_needed(all_token_nums)

        return result

    @classmethod
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("sentence_transformers") is not None

    @classmethod
    def match_json(cls, model_spec: EmbeddingModelSpec) -> bool:
        # As default embedding engine, sentence-transformer support all models
        return True
