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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import PIL.Image

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


class MinerUModel(OCRModel):
    """MinerU Vision-Language Model for document parsing and OCR.

    MinerU2.5 is a 1.2B parameter vision-language model designed for
    efficient high-resolution document parsing. It employs a two-stage
    strategy: global layout analysis followed by fine-grained content recognition.
    """

    required_libs = ("transformers",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        model_name = model_family.model_name
        return model_name.startswith("MinerU")

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
        self._client = None
        # info
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        try:
            from mineru_vl_utils import MinerUClient
        except ImportError:
            raise ImportError(
                "mineru-vl-utils is required for MinerU models. "
                "Please install it with: pip install 'mineru-vl-utils[transformers]'"
            )

        logger.info(f"Loading MinerU model from {self._model_path}")

        try:
            # Determine device and dtype
            if self._device == "cpu":
                device_map = "cpu"
                dtype = torch.float32
            else:
                device_map = "auto"
                dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )

            # Get torch_dtype from kwargs if specified
            torch_dtype = self._kwargs.get("torch_dtype", dtype)
            if isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype, dtype)

            # Load model with Qwen2VL architecture
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self._model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )

            # Load processor
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self._model_path,
                    trust_remote_code=True,
                    use_fast=True,
                )
            except ValueError:
                # Fallback for when AutoProcessor cannot identify the processor type
                from transformers import Qwen2VLProcessor

                self._processor = Qwen2VLProcessor.from_pretrained(
                    self._model_path,
                    trust_remote_code=True,
                    use_fast=True,
                )

            # Create MinerU client
            self._client = MinerUClient(
                backend="transformers",
                model=self._model,
                processor=self._processor,
            )

            logger.info(
                f"MinerU model loaded successfully with device_map={device_map}, dtype={torch_dtype}"
            )
        except Exception as e:
            logger.error(f"Failed to load MinerU model: {e}")
            raise

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        **kwargs,
    ) -> Union[str, List[str], Dict[str, Any]]:
        """
        Perform document parsing and OCR using MinerU vision-language model.

        Args:
            image: PIL Image or list of PIL Images
            **kwargs: Additional parameters including:
                - output_format: Output format ('markdown', 'json', 'text'), default: 'markdown'
                - return_dict: Whether to return a dictionary with metadata (default: False)
                - extract_mode: Extraction mode ('two_step', 'single_step'), default: 'two_step'

        Returns:
            Document content as string, list of strings, or dict
        """
        logger.info("MinerU OCR kwargs: %s", kwargs)

        if self._client is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        # Extract parameters
        output_format = kwargs.get("output_format", "markdown")
        return_dict = kwargs.get("return_dict", False)
        extract_mode = kwargs.get("extract_mode", "two_step")

        # Handle single image input
        if isinstance(image, PIL.Image.Image):
            result = self._process_single(image, output_format, extract_mode)
            if return_dict:
                return {
                    "text": result,
                    "model": "mineru",
                    "output_format": output_format,
                    "success": True,
                }
            return result

        # Handle batch image input
        elif isinstance(image, list):
            results = [
                self._process_single(img, output_format, extract_mode) for img in image
            ]
            if return_dict:
                return {
                    "text": results,
                    "model": "mineru",
                    "output_format": output_format,
                    "success": True,
                    "num_images": len(results),
                }
            return results

        else:
            raise ValueError("Input must be a PIL Image or list of PIL Images")

    def _process_single(
        self, image: PIL.Image.Image, output_format: str, extract_mode: str
    ) -> str:
        """Process a single image with MinerU."""
        assert self._client is not None, "Client not loaded. Call load() first."

        # Convert image to RGB if needed
        if image.mode in ["RGBA", "CMYK"]:
            image = image.convert("RGB")

        try:
            # Use two-step extraction for better accuracy (default)
            if extract_mode == "two_step":
                extracted_blocks = self._client.two_step_extract(image)
            else:
                # Single step extraction (faster but less accurate)
                extracted_blocks = self._client.extract(image)

            # Format output based on requested format
            if output_format == "json":
                return json.dumps(extracted_blocks, ensure_ascii=False, indent=2)
            elif output_format == "text":
                return self._blocks_to_text(extracted_blocks)
            else:  # markdown (default)
                return self._blocks_to_markdown(extracted_blocks)

        except Exception as e:
            logger.error(f"MinerU processing failed: {e}")
            raise

    def _blocks_to_markdown(self, blocks: Any) -> str:
        """Convert extracted blocks to markdown format."""
        if isinstance(blocks, str):
            return blocks

        if isinstance(blocks, dict):
            # Handle dict response with content
            if "content" in blocks:
                return str(blocks["content"])
            if "text" in blocks:
                return str(blocks["text"])
            if "markdown" in blocks:
                return str(blocks["markdown"])
            return json.dumps(blocks, ensure_ascii=False, indent=2)

        if isinstance(blocks, list):
            result_parts = []
            for block in blocks:
                if isinstance(block, str):
                    result_parts.append(block)
                elif isinstance(block, dict):
                    block_type = block.get("type", "text")
                    content = block.get("content", block.get("text", ""))

                    if block_type == "title":
                        level = block.get("level", 1)
                        result_parts.append(f"{'#' * level} {content}")
                    elif block_type == "table":
                        result_parts.append(str(content))
                    elif block_type == "formula":
                        result_parts.append(f"$${content}$$")
                    elif block_type == "image":
                        caption = block.get("caption", "")
                        result_parts.append(f"![{caption}]({content})")
                    else:
                        result_parts.append(str(content))
            return "\n\n".join(result_parts)

        return str(blocks)

    def _blocks_to_text(self, blocks: Any) -> str:
        """Convert extracted blocks to plain text format."""
        if isinstance(blocks, str):
            return blocks

        if isinstance(blocks, dict):
            if "content" in blocks:
                return str(blocks["content"])
            if "text" in blocks:
                return str(blocks["text"])
            return json.dumps(blocks, ensure_ascii=False)

        if isinstance(blocks, list):
            result_parts = []
            for block in blocks:
                if isinstance(block, str):
                    result_parts.append(block)
                elif isinstance(block, dict):
                    content = block.get("content", block.get("text", ""))
                    result_parts.append(str(content))
            return "\n".join(result_parts)

        return str(blocks)
