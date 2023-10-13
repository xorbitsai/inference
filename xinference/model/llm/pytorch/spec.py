# Copyright 2022-2023 XProbe Inc.
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

import logging
import os
from typing import Optional, Union, Iterator, List

from ....types import Completion, CompletionChunk, Embedding
from .core import PytorchChatModel, PytorchModelConfig, PytorchGenerateConfig
from .. import LLMFamilyV1, LLMSpecV1

logger = logging.getLogger(__name__)


class SpeculativeModel(PytorchChatModel):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        draft_model_family: "LLMFamilyV1",
        draft_model_spec: "LLMSpecV1",
        draft_quantization: str,
        draft_model_path: str,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._pytorch_model_config: PytorchModelConfig = self._sanitize_model_config(
            PytorchModelConfig()
        )
        self._draft_model_family = draft_model_family
        self._draft_model_spec = draft_model_spec
        self._draft_quantization = draft_quantization
        self._draft_model_path = draft_model_path

    def load(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                f"Failed to import module 'torch'. Please make sure 'torch' is installed.\n\n"
            )

        cuda_visible_devices_env = os.getenv("CUDA_VISIBLE_DEVICES", None)
        cuda_visible_devices = (
            cuda_visible_devices_env.split(",") if cuda_visible_devices_env else []
        )

        quantization = self.quantization
        num_gpus = len(cuda_visible_devices) if cuda_visible_devices_env != "-1" else 0
        device = self._pytorch_model_config.get("device", "auto")
        self._pytorch_model_config["device"] = self._select_device(device)
        self._device = self._pytorch_model_config["device"]

        if self._device == "cpu":
            kwargs = {"torch_dtype": torch.float32}
        elif self._device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
        elif self._device == "mps":
            kwargs = {"torch_dtype": torch.float16}
        else:
            raise ValueError(f"Device {self._device} is not supported in temporary")
        kwargs["trust_remote_code"] = self._pytorch_model_config.get(
            "trust_remote_code"
        )

        if self.quantization != "none":
            raise ValueError("Quantization is not supported by speculative decoding yet")

        if num_gpus > 0 and self._device == "cuda":
            kwargs.update({"device_map": "auto"})

        self._model, self._tokenizer = self._load_model(
            revision=self.model_spec.model_revision, **kwargs
        )
        if self._device == "mps":
            self._model.to(self._device)
        logger.debug(
            f"Model {self.model_uid} memory footprint: {self._model.get_memory_footprint()}")

        self._draft_model, _ = self._load_model(
            revision=self._draft_model_spec.model_revision, **kwargs
        )
        if self._device == "mps":
            self._model.to(self._device)
        logger.debug(
            f"Draft model {self.model_uid} memory footprint: {self._model.get_memory_footprint()}")

    def generate(
        self, prompt: str, generate_config: Optional[PytorchGenerateConfig] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        from .spec_decoding_utils import speculative_generate_stream

        generate_config = self._sanitize_generate_config(generate_config)

        assert self._draft_model is not None
        assert self._model is not None
        assert self._tokenizer is not None

        stream = generate_config.get("stream", False)
        if not stream:
            for completion_chunk, completion_usage in speculative_generate_stream(
                    draft_model=self._draft_model,
                model=self._model,
                tokenizer=self._tokenizer,
                prompt=prompt,
                generate_config=generate_config
            ):
                pass

            completion = Completion(
                id=completion_chunk["id"],
                object=completion_chunk["object"],
                created=completion_chunk["created"],
                model=completion_chunk["model"],
                choices=completion_chunk["choices"],
                usage=completion_usage,
            )
            return completion
        else:
            for completion_chunk, completion_usage in speculative_generate_stream(
                    draft_model=self._draft_model,
                    model=self._model,
                    tokenizer=self._tokenizer,
                    prompt=prompt,
                    generate_config=generate_config
            ):
                yield completion_chunk

    def create_embedding(self, input: Union[str, List[str]]) -> Embedding:
        raise NotImplementedError
