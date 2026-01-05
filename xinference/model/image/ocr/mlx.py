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

import logging
import platform
import sys
from typing import List, Optional, Union

import PIL.Image

from .deepseek_ocr import DeepSeekOCRModel

logger = logging.getLogger(__name__)


class MLXDeepSeekOCRModel(DeepSeekOCRModel):
    @classmethod
    def match(cls, model_family) -> bool:
        return model_family.model_name == "DeepSeek-OCR"

    def load(self):
        if sys.platform != "darwin" or platform.processor() != "arm":
            raise RuntimeError("MLX OCR engine only works on Apple silicon Macs.")
        try:
            from mlx_vlm import load
        except ImportError:
            error_message = "Failed to import module 'mlx_vlm'"
            installation_guide = [
                "Please make sure 'mlx_vlm' is installed. ",
                "You can install it by `pip install mlx_vlm`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._model, self._processor = load(self._model_path)
        self._tokenizer = self._processor.tokenizer

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Optional[str] = None,
        **kwargs,
    ):
        if prompt is None:
            prompt = "<image>\nFree OCR."
        if isinstance(image, list):
            return [self._ocr_single(img, prompt, **kwargs) for img in image]
        return self._ocr_single(image, prompt, **kwargs)

    def _ocr_single(
        self,
        image: PIL.Image.Image,
        prompt: str,
        model_size: str = "gundam",
        test_compress: bool = False,
        save_results: bool = False,
        save_dir: Optional[str] = None,
        eval_mode: bool = False,
        **kwargs,
    ):
        if image.mode in ("RGBA", "CMYK"):
            image = image.convert("RGB")
        text = self._generate_text(image, prompt, **kwargs)
        return {
            "text": text,
            "model": "deepseek-ocr",
            "success": True,
            "model_size": model_size,
        }

    def _prepare_inputs(self, image: PIL.Image.Image, prompt: str):
        from mlx_vlm import prepare_inputs

        processor = self._processor
        inputs = prepare_inputs(processor=processor, images=[image], prompts=prompt)
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        mask = inputs["attention_mask"]
        extra = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        return (input_ids, pixel_values, mask, extra)

    def _generate_text(self, image: PIL.Image.Image, prompt: str, **kwargs) -> str:
        try:
            from mlx_vlm.generate import generate_step
        except ImportError:
            raise ImportError(
                "Failed to import mlx_vlm.generate.generate_step. "
                "Please make sure mlx_vlm is installed."
            )

        input_ids, pixel_values, mask, extra = self._prepare_inputs(image, prompt)
        stop_token_ids = kwargs.pop("stop_token_ids", [])
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        if max_new_tokens is None:
            max_new_tokens = kwargs.pop("max_tokens", 512)
        temperature = kwargs.pop("temperature", 0.0)
        top_p = kwargs.pop("top_p", None)

        gen_kwargs = {"max_tokens": max_new_tokens, "temperature": temperature}
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        gen_kwargs.update(extra)
        gen_kwargs.update(kwargs)

        detokenizer = self._processor.detokenizer
        tokenizer = self._processor.tokenizer
        detokenizer.reset()
        text_parts = []

        for token, _ in generate_step(
            input_ids, self._model, pixel_values, mask, **gen_kwargs
        ):
            if token == tokenizer.eos_token_id or token in stop_token_ids:
                break
            detokenizer.add_token(token)
            text_parts.append(detokenizer.last_segment)
        detokenizer.finalize()
        text_parts.append(detokenizer.last_segment)
        return "".join(text_parts)
