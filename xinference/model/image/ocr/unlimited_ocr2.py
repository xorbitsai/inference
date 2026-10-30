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

from typing import TYPE_CHECKING, Any, List, Optional, Union

import PIL.Image
import torch

from ....device_utils import get_available_device
from ...utils import allow_trust_remote_code
from .ocr_family import OCRModel
from .unlimited_ocr import UnlimitedOCRModelSize

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2


class UnlimitedOCR2Model(OCRModel):
    required_libs = ("transformers",)
    DEFAULT_PROMPT = "Multi page parsing."

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return (
            model_family.model_name == "Unlimited-OCR-2"
            and getattr(model_family, "model_format", None) == "pytorch"
        )

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs: Any,
    ) -> None:
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._tokenizer = None
        self._abilities = model_spec.model_ability if model_spec is not None else []
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        device = torch.device(self._device or get_available_device())
        # The upstream vision encoder uses CUDA tensors internally.
        if device.type != "cuda":
            raise ValueError("Unlimited-OCR-2 requires a CUDA device")
        model_kwargs = self._kwargs.copy()
        model_kwargs.pop("cpu_offload", None)
        trust_remote_code = allow_trust_remote_code(self.model_family)
        model_kwargs.setdefault("trust_remote_code", trust_remote_code)
        model_kwargs.setdefault("use_safetensors", True)
        model_kwargs.setdefault("torch_dtype", torch.bfloat16)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=trust_remote_code
        )
        model = AutoModel.from_pretrained(self._model_path, **model_kwargs)
        if "device_map" not in model_kwargs:
            model = model.to(device)
        self._model = model.eval()

    @classmethod
    def _normalize_prompt(cls, prompt: Optional[str]) -> str:
        if prompt and prompt.startswith("<image>"):
            prompt = prompt[len("<image>") :].lstrip()
        if not prompt or prompt in (
            "OCR",
            "Free OCR.",
            "Free OCR. Extract all text content from the image.",
        ):
            return cls.DEFAULT_PROMPT
        return prompt

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        images = image if isinstance(image, list) else [image]
        if not images or not all(isinstance(img, PIL.Image.Image) for img in images):
            raise ValueError(
                "Input must be a PIL Image or a non-empty list of PIL Images"
            )
        save_results = kwargs.pop("save_results", False)
        save_dir = kwargs.pop("save_dir", None)
        if save_results and not save_dir:
            raise ValueError("save_dir must be provided when save_results=True")
        model_size = kwargs.pop("model_size", "gundam")
        model_config = UnlimitedOCRModelSize.from_string(
            "base" if isinstance(image, list) else model_size
        )
        kwargs.pop("request_id", None)
        kwargs.setdefault("base_size", model_config.base_size)
        kwargs.setdefault("image_size", model_config.image_size)
        kwargs.setdefault("crop_mode", model_config.crop_mode)
        kwargs.setdefault("max_length", 131072)
        # Always return text through the OCR API, including when files are saved.
        kwargs.pop("return_dict", None)
        if self._model is None or self._tokenizer is None:
            self.load()
        assert self._model is not None
        assert self._tokenizer is not None
        rgb_images = [img.convert("RGB") for img in images]
        # infer accepts PIL images and constructs <PAGE><image> per page itself.
        with torch.inference_mode():
            text = self._model.infer(
                self._tokenizer,
                prompt=self._normalize_prompt(prompt),
                image_file=rgb_images if isinstance(image, list) else rgb_images[0],
                output_path=save_dir if save_results else "",
                save_results=save_results,
                **kwargs,
            )
        return text.strip()
