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

import json
import logging
import os
from typing import Any, Dict, List, Tuple, Union, cast

from ....device_utils import is_vacc_available
from ....types import Embedding, EmbeddingData, EmbeddingUsage
from ...batch import BatchMixin
from ...utils import check_dependency_available
from ..core import EmbeddingModel, EmbeddingModelFamilyV2, EmbeddingSpecV1

logger = logging.getLogger(__name__)
SUPPORTED_MODELS_PREFIXES = ["bge", "gte", "text2vec", "m3e", "gte", "Qwen3"]


class VLLMEmbeddingModel(EmbeddingModel, BatchMixin):
    def __init__(self, *args, **kwargs):
        EmbeddingModel.__init__(self, *args, **kwargs)
        BatchMixin.__init__(self, self.create_embedding, **kwargs)  # type: ignore
        self._context_length = None

    def load(self):
        try:
            if is_vacc_available():
                import vllm_vacc  # noqa: F401
            from packaging.version import Version
            from vllm import LLM
            from vllm import __version__ as vllm_version

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

        if self.model_family.model_name.startswith("Qwen3-VL-Embedding"):
            if Version(vllm_version) < Version("0.14.0"):
                raise ValueError("Qwen3-VL embedding requires vLLM>=0.14.0")
            self._model = LLM(model=self._model_path, runner="pooling", **self._kwargs)
        else:
            if Version(vllm_version) >= Version("0.13.0"):
                self._model = LLM(
                    model=self._model_path, runner="pooling", **self._kwargs
                )
            else:
                self._model = LLM(model=self._model_path, task="embed", **self._kwargs)
        self._tokenizer = self._model.get_tokenizer()

    @staticmethod
    def _get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"  # noqa: E231

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
        if Version(vllm_version) > Version("0.16.0"):
            pool_params = PoolingParams(
                dimensions=dimensions, use_activation=normalize_embedding
            )
        elif Version(vllm_version) > Version("0.10.1"):
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
        if self.model_family.model_name.startswith("Qwen3-VL-Embedding"):
            outputs = self._embed_vl(sentences)
        else:
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
            if hasattr(output, "prompt_token_ids"):
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
        self._clean_cache_if_needed(all_token_nums)

        return result

    def _embed_vl(
        self, inputs: Union[str, List[str], Dict[str, Any], List[Dict[str, Any]]]
    ):
        import base64
        import re
        from io import BytesIO

        from PIL import Image
        from vllm.multimodal.utils import fetch_image

        if isinstance(inputs, str):
            normalized: List[Dict[str, Any]] = [{"text": inputs}]
        elif isinstance(inputs, dict):
            normalized = [inputs]
        elif isinstance(inputs, list) and inputs and isinstance(inputs[0], str):
            normalized = [{"text": item} for item in inputs]
        elif isinstance(inputs, list):
            normalized = cast(List[Dict[str, Any]], inputs)
        else:
            raise ValueError("Unsupported input type for Qwen3-VL embedding.")

        vllm_inputs: List[Dict[str, Any]] = []

        def _format_input_to_conversation(
            input_dict: Dict[str, Any], instruction: str = "Represent the user's input."
        ) -> List[Dict]:
            content = []

            text = input_dict.get("text")
            image = input_dict.get("image")

            if image:
                image_content = None
                if isinstance(image, str):
                    if re.match(r"^data:image/.+;base64,", image):
                        image_content = image
                    elif image.startswith(("http", "https", "oss")):
                        image_content = image
                    else:
                        abs_image_path = os.path.abspath(image)
                        image_content = "file://" + abs_image_path
                else:
                    image_content = image

                if image_content:
                    content.append(
                        {
                            "type": "image",
                            "image": image_content,
                        }
                    )

            if text:
                content.append({"type": "text", "text": text})

            if not content:
                content.append({"type": "text", "text": ""})

            conversation = [
                {"role": "system", "content": [{"type": "text", "text": instruction}]},
                {"role": "user", "content": content},
            ]

            return conversation

        def _prepare_vllm_inputs(
            input_dict: Dict[str, Any],
            llm,
            instruction: str = "Represent the user's input.",
        ) -> Dict[str, Any]:
            image = input_dict.get("image")

            conversation = _format_input_to_conversation(input_dict, instruction)

            assert self._model is not None
            prompt_text = self._model.llm_engine.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            multi_modal_data = None
            if image:
                if isinstance(image, str):
                    if re.match(r"^data:image/.+;base64,", image):
                        base64_data = image.split(",", 1)[1]
                        data = base64.b64decode(base64_data)
                        image_obj = Image.open(BytesIO(data)).convert("RGB")
                        multi_modal_data = {"image": image_obj}
                    elif image.startswith(("http", "https", "oss")):
                        try:
                            image_obj = fetch_image(image)
                            multi_modal_data = {"image": image_obj}
                        except Exception as e:
                            print(f"Warning: Failed to fetch image {image}: {e}")
                    else:
                        abs_image_path = os.path.abspath(image)
                        if os.path.exists(abs_image_path):
                            image_obj = Image.open(abs_image_path)
                            multi_modal_data = {"image": image_obj}
                        else:
                            print(f"Warning: Image file not found: {abs_image_path}")
                else:
                    multi_modal_data = {"image": image}

            result = {"prompt": prompt_text, "multi_modal_data": multi_modal_data}
            return result

        assert self._model is not None
        vllm_inputs = [_prepare_vllm_inputs(item, self._model) for item in normalized]
        return self._model.embed(vllm_inputs)

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        dep_check = check_dependency_available("vllm", "vLLM")
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

        if model_family.model_name.startswith("Qwen3-VL-Embedding"):
            import vllm
            from packaging import version

            if version.parse(vllm.__version__) < version.parse("0.14.0"):
                return (
                    False,
                    f"Qwen3-VL embedding requires vLLM>=0.14.0, current: {vllm.__version__}",
                )
        if model_spec.model_format not in ["pytorch"]:
            return False, "vLLM embedding engine only supports pytorch format"
        prefix = model_family.model_name.split("-", 1)[0]
        if prefix not in SUPPORTED_MODELS_PREFIXES:
            return (
                False,
                f"Model family {model_family.model_name} is not in the supported prefix list for vLLM embeddings",
            )
        return True

    def wait_for_load(self):
        # set context length after engine inited
        self._set_context_length()

    def _set_context_length(self):
        """Set context length"""
        from packaging import version
        from vllm import __version__ as vllm_version
        from vllm import envs

        # For vLLM >= 0.11.1, v1 is default; older versions rely on env var.
        if version.parse(vllm_version) > version.parse("0.11.0"):
            use_v1 = True
        else:
            use_v1 = False
            if hasattr(envs, "VLLM_USE_V1"):
                try:
                    use_v1 = envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1
                except AttributeError:
                    use_v1 = False

        if not use_v1:
            # v0
            self._context_length = (
                self._model.llm_engine.vllm_config.model_config.max_model_len
            )
        else:
            # v1
            logger.warning("vLLM v1 is not supported, ignore context length setting")
        logger.debug("Model context length: %s", self._context_length)
