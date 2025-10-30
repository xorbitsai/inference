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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import PIL.Image

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

logger = logging.getLogger(__name__)


class DeepSeekOCRModel:
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
        self._tokenizer = None
        # info
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def load(self):
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading DeepSeek-OCR model from {self._model_path}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                use_fast=False,
            )
            model = AutoModel.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cuda" if self._device != "cpu" else "cpu",
                use_safetensors=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            self._model = model.eval()
            if self._device != "cpu":
                self._model = self._model.cuda()
            logger.info("DeepSeek-OCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DeepSeek-OCR model: {e}")
            raise

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: str = "<image>\nFree OCR.",
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform OCR on single or multiple images.

        Args:
            image: PIL Image or list of PIL Images
            prompt: OCR prompt, defaults to "<image>\nFree OCR."
            **kwargs: Additional parameters

        Returns:
            OCR results as dict or list of dicts
        """
        logger.info("DeepSeek-OCR kwargs: %s", kwargs)

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        # Handle single image input
        if isinstance(image, PIL.Image.Image):
            return self._ocr_single(image, prompt, **kwargs)
        # Handle batch image input
        elif isinstance(image, list):
            return [self._ocr_single(img, prompt, **kwargs) for img in image]
        else:
            raise ValueError("Input must be a PIL Image or list of PIL Images")

    def _ocr_single(
        self, image: PIL.Image.Image, prompt: str, **kwargs
    ) -> Dict[str, Any]:
        """Perform OCR on a single image."""
        # Convert image to RGB if needed
        if image.mode in ["RGBA", "CMYK"]:
            image = image.convert("RGB")

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        try:
            # Save image to temporary file for DeepSeek-OCR's infer method
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                image.save(temp_file.name, "JPEG")
                temp_image_path = temp_file.name

            try:
                # Use DeepSeek-OCR's infer method
                result = self._model.infer(
                    tokenizer=self._tokenizer,
                    prompt=prompt,
                    image_file=temp_image_path,
                    output_path=tempfile.mkdtemp(),
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    eval_mode=True,
                )

                # DeepSeek-OCR infer method returns the OCR text directly
                return {"text": result, "model": "deepseek-ocr", "success": True}
            finally:
                # Clean up temporary file
                os.unlink(temp_image_path)

        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return {
                "text": "",
                "model": "deepseek-ocr",
                "success": False,
                "error": str(e),
            }

    def infer(
        self,
        image_paths: Union[str, List[str]],
        prompt: str = "<image>\nFree OCR.",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Inference method for compatibility with Xinference interface.

        Args:
            image_paths: Single path or list of paths to images
            prompt: OCR prompt
            **kwargs: Additional parameters

        Returns:
            Dictionary containing OCR results
        """
        from PIL import Image

        # Convert string input to list
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        # Load images
        images = []
        for path in image_paths:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                logger.error(f"Failed to load image {path}: {e}")
                images.append(None)

        # Process images
        results = []
        for i, img in enumerate(images):
            if img is None:
                results.append(
                    {
                        "image": image_paths[i],
                        "text": "",
                        "success": False,
                        "error": "Failed to load image",
                    }
                )
            else:
                text_result = self._ocr_single(img, prompt, **kwargs)
                results.append(
                    {"image": image_paths[i], "text": text_result, "success": True}
                )

        return {"results": results}
