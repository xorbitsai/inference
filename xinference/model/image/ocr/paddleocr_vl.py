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

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import PIL.Image
import torch

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


class PaddleOCRVLModel(OCRModel):
    """PaddleOCR-VL model for OCR, table recognition, formula recognition, and chart recognition."""

    required_libs = ("transformers",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_name == "PaddleOCR-VL"

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        # model info when loading
        self._model = None
        self._processor = None
        # info
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def load(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info(f"Loading PaddleOCR-VL model from {self._model_path}")

        try:
            # Determine device and dtype
            if self._device == "cpu":
                device = "cpu"
                dtype = torch.float32
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16

            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self._model_path, trust_remote_code=True
            )

            # Load model
            self._model = (
                AutoModelForCausalLM.from_pretrained(
                    self._model_path,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                )
                .to(device)
                .eval()
            )

            logger.info(
                f"PaddleOCR-VL model loaded successfully on {device} with dtype {dtype}"
            )
        except Exception as e:
            logger.error(f"Failed to load PaddleOCR-VL model: {e}")
            raise

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        **kwargs,
    ) -> Union[str, List[str], Dict[str, Any]]:
        """
        Perform OCR, table recognition, formula recognition, or chart recognition.

        Args:
            image: PIL Image or list of PIL Images
            **kwargs: Additional parameters including:
                - task: Task type ('ocr', 'table', 'formula', 'chart'), default: 'ocr'
                - prompt: Custom prompt (optional, overrides task-based prompt)
                - max_new_tokens: Maximum number of tokens to generate (default: 1024)
                - return_dict: Whether to return a dictionary with metadata (default: False)

        Returns:
            OCR results as string, list of strings, or dict
        """
        logger.info("PaddleOCR-VL kwargs: %s", kwargs)

        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        # Extract parameters
        task = kwargs.get("task", "ocr")
        custom_prompt = kwargs.get("prompt", None)
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        return_dict = kwargs.get("return_dict", False)

        # Define task prompts
        PROMPTS = {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
        }

        # Use custom prompt if provided, otherwise use task-based prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = PROMPTS.get(task, PROMPTS["ocr"])

        # Handle single image input
        if isinstance(image, PIL.Image.Image):
            result = self._process_single(image, prompt, max_new_tokens)
            if return_dict:
                return {
                    "text": result,
                    "model": "paddleocr-vl",
                    "task": task,
                    "success": True,
                }
            return result

        # Handle batch image input
        elif isinstance(image, list):
            results = [
                self._process_single(img, prompt, max_new_tokens) for img in image
            ]
            if return_dict:
                return {
                    "text": results,
                    "model": "paddleocr-vl",
                    "task": task,
                    "success": True,
                    "num_images": len(results),
                }
            return results

        else:
            raise ValueError("Input must be a PIL Image or list of PIL Images")

    def _process_single(
        self, image: PIL.Image.Image, prompt: str, max_new_tokens: int
    ) -> str:
        """Process a single image with the given prompt."""
        # Ensure model and processor are loaded
        assert self._model is not None, "Model not loaded. Call load() first."
        assert self._processor is not None, "Processor not loaded. Call load() first."

        # Convert image to RGB if needed
        if image.mode in ["RGBA", "CMYK"]:
            image = image.convert("RGB")

        # Get device
        device = next(self._model.parameters()).device

        # Prepare messages in the format expected by PaddleOCR-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        # Generate
        with torch.inference_mode():
            outputs = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode output
        # Slice to remove input prompt from output
        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        result = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return result
