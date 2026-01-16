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
import os
import re
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from torchvision import transforms

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


class DeepSeekOCRModelSize:
    """DeepSeek-OCR model size configurations."""

    TINY = ("tiny", 512, 512, False)
    SMALL = ("small", 640, 640, False)
    BASE = ("base", 1024, 1024, False)
    LARGE = ("large", 1280, 1280, False)
    GUNDAM = ("gundam", 1024, 640, True)

    def __init__(self, size_type: str):
        self.size_type = size_type
        # Map size type to configuration
        self._config_map = {
            "tiny": self.TINY,
            "small": self.SMALL,
            "base": self.BASE,
            "large": self.LARGE,
            "gundam": self.GUNDAM,
        }

        if size_type in self._config_map:
            self.name, self.base_size, self.image_size, self.crop_mode = (
                self._config_map[size_type]
            )
        else:
            # Default to Gundam
            self.name, self.base_size, self.image_size, self.crop_mode = self.GUNDAM

    @classmethod
    def from_string(cls, size_str: str) -> "DeepSeekOCRModelSize":
        """Get model size from string."""
        return cls(size_str.lower())

    def __str__(self) -> str:
        return self.name


def load_image(image_path: str) -> Optional[PIL.Image.Image]:
    """Load image with EXIF correction."""
    try:
        image = PIL.Image.open(image_path)
        # Correct image orientation based on EXIF data
        corrected_image = PIL.ImageOps.exif_transpose(image)
        return corrected_image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        try:
            return PIL.Image.open(image_path)
        except:
            return None


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: List[Tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> Tuple[int, int]:
    """Find the closest aspect ratio to target."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(
    image: PIL.Image.Image,
    min_num: int = 2,
    max_num: int = 9,
    image_size: int = 640,
    use_thumbnail: bool = False,
) -> Tuple[List[PIL.Image.Image], Tuple[int, int]]:
    """Dynamically preprocess image by cropping."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate target ratios
    target_ratios = [
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    ]
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images, target_aspect_ratio


def normalize_transform(
    mean: Optional[Union[Tuple[float, float, float], List[float]]],
    std: Optional[Union[Tuple[float, float, float], List[float]]],
):
    """Create normalization transform."""
    if mean is None and std is None:
        return None
    elif mean is None and std is not None:
        mean = [0.0] * len(std)
        return transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.0] * len(mean)
        return transforms.Normalize(mean=mean, std=std)
    else:
        return transforms.Normalize(mean=mean, std=std)


class BasicImageTransform:
    """Basic image transformation for DeepSeek-OCR."""

    def __init__(
        self,
        mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
        normalize: bool = True,
    ):
        self.mean = mean
        self.std = std

        transform_pipelines = [transforms.ToTensor()]

        if normalize:
            normalize_transform_func = normalize_transform(mean, std)
            if normalize_transform_func is not None:
                transform_pipelines.append(normalize_transform_func)
            else:
                transform_pipelines.append(nn.Identity())

        self.transform = transforms.Compose(transform_pipelines)

    def __call__(self, x: PIL.Image.Image) -> torch.Tensor:
        return self.transform(x)


def re_match(text: str) -> Tuple[List[Tuple], List[str], List[str]]:
    """Extract references and detections from text."""
    pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
    matches = re.findall(pattern, text, re.DOTALL)

    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if "<|ref|>image<|/ref|>" in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(
    ref_text: Tuple, image_width: int, image_height: int
) -> Optional[Tuple]:
    """Extract coordinates and label from reference text."""
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        logger.error(f"Error extracting coordinates: {e}")
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(
    image: PIL.Image.Image, refs: List[Tuple], output_path: str
) -> PIL.Image.Image:
    """Draw bounding boxes on image with labels."""
    image_width, image_height = image.size

    img_draw = image.copy()
    draw = PIL.ImageDraw.Draw(img_draw)

    overlay = PIL.Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = PIL.ImageDraw.Draw(overlay)

    # Use default font
    try:
        font = PIL.ImageFont.load_default()
    except:
        font = None

    img_idx = 0

    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                color = (
                    np.random.randint(0, 200),
                    np.random.randint(0, 200),
                    np.random.randint(0, 255),
                )
                color_a = color + (20,)

                for points in points_list:
                    x1, y1, x2, y2 = points

                    # Convert from relative coordinates (0-999) to absolute pixel coordinates
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    if label_type == "image":
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(f"{output_path}/images/{img_idx}.jpg")
                        except Exception as e:
                            logger.error(f"Error saving cropped image: {e}")
                        img_idx += 1

                    try:
                        if label_type == "title":
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )

                        if font:
                            text_x = x1
                            text_y = max(0, y1 - 15)

                            text_bbox = draw.textbbox((0, 0), label_type, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]

                            draw.rectangle(
                                [
                                    text_x,
                                    text_y,
                                    text_x + text_width,
                                    text_y + text_height,
                                ],
                                fill=(255, 255, 255, 30),
                            )

                            draw.text(
                                (text_x, text_y), label_type, font=font, fill=color
                            )
                    except Exception as e:
                        logger.error(f"Error drawing text: {e}")
                        pass
        except Exception as e:
            logger.error(f"Error processing reference: {e}")
            continue

    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(
    image: PIL.Image.Image, ref_texts: List[Tuple], output_path: str
) -> PIL.Image.Image:
    """Process image with reference texts and draw bounding boxes."""
    result_image = draw_bounding_boxes(image, ref_texts, output_path)
    return result_image


def clean_ocr_annotations(text: str) -> str:
    """
    Clean OCR annotations and return plain text.

    Removes <|ref|>...<|/ref|><|det|>...<|/det|> annotations while preserving the text content.

    Args:
        text: Raw OCR output with annotations

    Returns:
        Cleaned plain text
    """
    if not isinstance(text, str):
        return str(text)

    # Pattern to match the full annotation blocks
    annotation_pattern = r"<\|ref\|>.*?<\|/ref\|><\|det\|>\[\[.*?\]\]<\|/det\|>"

    # Remove all annotation blocks
    cleaned_text = re.sub(annotation_pattern, "", text, flags=re.DOTALL)

    # Clean up extra whitespace and line breaks
    cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text.strip())

    return cleaned_text


def extract_text_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Extract text blocks with their coordinates from OCR annotations.

    Args:
        text: Raw OCR output with annotations

    Returns:
        List of dictionaries containing text and coordinates
    """
    if not isinstance(text, str):
        return []

    # Pattern to extract text and coordinates
    block_pattern = (
        r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[(.*?)\]\]<\|/det\|>(.*?)(?=<\|ref\|>|$)"
    )

    blocks = []
    for match in re.finditer(block_pattern, text, re.DOTALL):
        label_type = match.group(1).strip()
        coords_str = match.group(2).strip()
        content = match.group(3).strip()

        try:
            coords = eval(f"[{coords_str}]")  # Convert string coordinates to list
            if isinstance(coords, list) and len(coords) > 0:
                blocks.append(
                    {
                        "label_type": label_type,
                        "coordinates": coords,
                        "text": content,
                        "bbox": coords[0] if len(coords) == 1 else coords,
                    }
                )
        except:
            # Skip if coordinates can't be parsed
            continue

    return blocks


class DeepSeekOCRModel(OCRModel):
    required_libs: Tuple[str, ...] = ("transformers",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        model_format = getattr(model_family, "model_format", None)
        return model_family.model_name == "DeepSeek-OCR" and model_format != "mlx"

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
            if self._device != "cpu":
                # Use CUDA if available
                model = AutoModel.from_pretrained(
                    self._model_path,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="auto",
                    use_safetensors=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                self._model = model.eval()
            else:
                # Force CPU-only execution
                model = AutoModel.from_pretrained(
                    self._model_path,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="cpu",
                    use_safetensors=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                )
                self._model = model.eval()
            logger.info("DeepSeek-OCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load DeepSeek-OCR model: {e}")
            raise

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform OCR on single or multiple images with enhanced features.

        Args:
            image: PIL Image or list of PIL Images
            **kwargs: Additional parameters including:
                - prompt: OCR prompt (default: "<image>\nFree OCR.")
                - model_size: Model size (default: "gundam")
                - test_compress: Whether to test compression ratio (default: False)
                - save_results: Whether to save results (default: False)
                - save_dir: Directory to save results
                - eval_mode: Whether to use evaluation mode (default: False)

        Returns:
            OCR results as dict or list of dicts
        """
        logger.info("DeepSeek-OCR kwargs: %s", kwargs)

        # Set default values for DeepSeek-OCR specific parameters
        prompt = kwargs.pop("prompt", "<image>\nFree OCR.")
        model_size = kwargs.pop("model_size", "gundam")
        test_compress = kwargs.pop("test_compress", False)
        save_results = kwargs.pop("save_results", False)
        save_dir = kwargs.pop("save_dir", None)
        eval_mode = kwargs.pop("eval_mode", False)

        # Smart detection: Check if this should be a visualization request
        # Visualization is triggered when:
        # 1. prompt contains grounding keywords
        # 2. save_results is True (default behavior for visualization)
        # 3. Explicit visualization parameters are provided
        is_visualization_request = (
            "<|grounding|>" in prompt
            or save_results
            or any(
                key in kwargs
                for key in ["save_results", "output_format", "annotations", "visualize"]
            )
        )

        if is_visualization_request:
            logger.info("Detected visualization request, delegating to visualize_ocr")
            # Delegate to visualize_ocr for visualization functionality
            # Pass all parameters through kwargs to avoid duplication
            return self.visualize_ocr(
                image=image,
                prompt=prompt,
                model_size=model_size,
                save_results=save_results,
                save_dir=save_dir,
                eval_mode=eval_mode,
                **kwargs,
            )

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        # Validate parameters
        if save_results and not save_dir:
            raise ValueError("save_dir must be provided when save_results=True")

        # Handle single image input
        if isinstance(image, PIL.Image.Image):
            return self._ocr_single(
                image,
                prompt,
                model_size,
                test_compress,
                save_results,
                save_dir,
                eval_mode,
                **kwargs,
            )
        # Handle batch image input
        elif isinstance(image, list):
            return [
                self._ocr_single(
                    img,
                    prompt,
                    model_size,
                    test_compress,
                    save_results,
                    save_dir,
                    eval_mode,
                    **kwargs,
                )
                for img in image
            ]
        else:
            raise ValueError("Input must be a PIL Image or list of PIL Images")

    def visualize_ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
        model_size: str = "gundam",
        save_results: bool = True,
        save_dir: Optional[str] = None,
        eval_mode: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Perform OCR with visualization (bounding boxes and annotations).

        Args:
            image: PIL Image or list of PIL Images
            prompt: OCR prompt with grounding, defaults to document conversion
            model_size: Model size configuration
            save_results: Whether to save results with annotations
            save_dir: Directory to save results
            eval_mode: Whether to use evaluation mode
            **kwargs: Additional parameters

        Returns:
            OCR results with visualization information
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        # Handle single image input
        if isinstance(image, PIL.Image.Image):
            result = self._visualize_single(
                image, prompt, model_size, save_results, save_dir, eval_mode, **kwargs
            )

            # Apply LaTeX post-processing using unified function
            try:
                from ...ui.gradio.utils.latex import process_ocr_result_with_latex

                result = process_ocr_result_with_latex(
                    result, output_format="markdown", debug_info=True
                )
            except ImportError:
                # Fallback: no LaTeX processing if import fails
                pass

            return result
        # Handle batch image input
        elif isinstance(image, list):
            results = []
            for img in image:
                result = self._visualize_single(
                    img, prompt, model_size, save_results, save_dir, eval_mode, **kwargs
                )

                # Apply LaTeX post-processing using unified function
                try:
                    from ...ui.gradio.utils.latex import process_ocr_result_with_latex

                    result = process_ocr_result_with_latex(
                        result, output_format="markdown", debug_info=False
                    )
                except ImportError:
                    # Fallback: no LaTeX processing if import fails
                    pass

                results.append(result)
            return results
        else:
            raise ValueError("Input must be a PIL Image or list of PIL Images")

    def _visualize_single(
        self,
        image: PIL.Image.Image,
        prompt: str,
        model_size: str,
        save_results: bool,
        save_dir: Optional[str],
        eval_mode: bool,
        **kwargs,
    ) -> Dict[str, Any]:
        """Perform OCR with visualization for a single image."""
        # Convert image to RGB if needed
        if image.mode in ["RGBA", "CMYK"]:
            image = image.convert("RGB")

        # Get model configuration
        model_config = DeepSeekOCRModelSize.from_string(model_size)

        # Create save directory if needed
        if save_results and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(f"{save_dir}/images", exist_ok=True)

        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load() method first.")

        try:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                image.save(temp_file.name, "JPEG")
                temp_image_path = temp_file.name

            # Create output directory
            output_path = tempfile.mkdtemp() if not save_dir else save_dir

            try:
                # Use DeepSeek-OCR's infer method with save_results enabled
                result = self._model.infer(
                    tokenizer=self._tokenizer,
                    prompt=prompt,
                    image_file=temp_image_path,
                    output_path=output_path,
                    base_size=model_config.base_size,
                    image_size=model_config.image_size,
                    crop_mode=model_config.crop_mode,
                    save_results=save_results,
                    eval_mode=eval_mode,
                )

                # Process visualization if save_results is enabled
                visualization_info = {}
                if save_results and save_dir and isinstance(result, str):
                    try:
                        # Extract references from result
                        matches_ref, matches_images, matches_other = re_match(result)

                        # Process image with references
                        if matches_ref:
                            result_image = process_image_with_refs(
                                image.copy(), matches_ref, save_dir
                            )
                            result_image.save(f"{save_dir}/result_with_boxes.jpg")

                            # Process image references in text
                            processed_text = result
                            for idx, match_image in enumerate(matches_images):
                                processed_text = processed_text.replace(
                                    match_image, f"![](images/{idx}.jpg)\n"
                                )

                            # Remove other reference markers
                            for idx, match_other in enumerate(matches_other):
                                processed_text = processed_text.replace(match_other, "")

                            # Save processed text as markdown
                            with open(
                                f"{save_dir}/result.mmd", "w", encoding="utf-8"
                            ) as f:
                                f.write(processed_text)

                            visualization_info = {
                                "has_annotations": True,
                                "num_bounding_boxes": len(matches_ref),
                                "num_extracted_images": len(matches_images),
                                "annotated_image_path": f"{save_dir}/result_with_boxes.jpg",
                                "markdown_path": f"{save_dir}/result.mmd",
                                "extracted_images_dir": f"{save_dir}/images/",
                            }
                        else:
                            visualization_info = {
                                "has_annotations": False,
                                "message": "No annotations found in OCR result",
                            }
                    except Exception as e:
                        logger.error(f"Error processing visualization: {e}")
                        visualization_info = {"error": str(e)}

                # Prepare response
                response = {
                    "text": result,
                    "model": "deepseek-ocr",
                    "success": True,
                    "model_size": model_size,
                    "base_size": model_config.base_size,
                    "image_size": model_config.image_size,
                    "crop_mode": model_config.crop_mode,
                    "visualization": visualization_info,
                }

                # Add file info if saved
                if save_results and save_dir:
                    response["saved_files"] = {
                        "output_dir": save_dir,
                        "result_file": (
                            f"{save_dir}/result.mmd"
                            if os.path.exists(f"{save_dir}/result.mmd")
                            else None
                        ),
                        "annotated_image": (
                            f"{save_dir}/result_with_boxes.jpg"
                            if os.path.exists(f"{save_dir}/result_with_boxes.jpg")
                            else None
                        ),
                    }

                return response

            finally:
                # Clean up temporary file
                os.unlink(temp_image_path)

        except Exception as e:
            logger.error(f"OCR visualization failed: {e}")
            return {
                "text": "",
                "model": "deepseek-ocr",
                "success": False,
                "error": str(e),
                "model_size": model_size,
                "visualization": {"error": str(e)},
            }

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
    ) -> Dict[str, Any]:
        """Perform OCR on a single image with all enhanced features."""
        # Convert image to RGB if needed
        if image.mode in ["RGBA", "CMYK"]:
            image = image.convert("RGB")

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        # Get model configuration
        model_config = DeepSeekOCRModelSize.from_string(model_size)

        # Create save directory if needed
        if save_results and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            os.makedirs(f"{save_dir}/images", exist_ok=True)

        try:
            # Save image to temporary file for DeepSeek-OCR's infer method
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                image.save(temp_file.name, "JPEG")
                temp_image_path = temp_file.name

            # Create output directory
            output_path = tempfile.mkdtemp() if not save_dir else save_dir

            try:
                # Use DeepSeek-OCR's infer method with all parameters
                result = self._model.infer(
                    tokenizer=self._tokenizer,
                    prompt=prompt,
                    image_file=temp_image_path,
                    output_path=output_path,
                    base_size=model_config.base_size,
                    image_size=model_config.image_size,
                    crop_mode=model_config.crop_mode,
                    test_compress=test_compress,
                    save_results=save_results,
                    eval_mode=eval_mode,
                )

                # Apply LaTeX post-processing using unified function
                try:
                    from ...ui.gradio.utils.latex import process_ocr_result_with_latex

                    # Process the result and extract LaTeX info
                    processed_result = process_ocr_result_with_latex(
                        result, output_format="markdown", debug_info=True
                    )

                    # Extract text and LaTeX info
                    if isinstance(processed_result, dict):
                        latex_info = processed_result.get("latex_processing")
                        processed_result = processed_result.get("text", result)
                    else:
                        processed_result = (
                            processed_result if processed_result else result
                        )
                        latex_info = None

                except ImportError:
                    processed_result = result
                    latex_info = None

                # Prepare response
                response = {
                    "text": processed_result,
                    "model": "deepseek-ocr",
                    "success": True,
                    "model_size": model_size,
                    "base_size": model_config.base_size,
                    "image_size": model_config.image_size,
                    "crop_mode": model_config.crop_mode,
                }

                # If the model returned an empty result, fall back to visualization
                # mode (same path as Gradio) to give users a usable response.
                if processed_result is None or (
                    isinstance(processed_result, str) and not processed_result.strip()
                ):
                    logger.warning(
                        "DeepSeek-OCR returned empty text, falling back to visualization mode."
                    )
                    return self.visualize_ocr(
                        image=image,
                        prompt=prompt,
                        model_size=model_size,
                        save_results=save_results,
                        save_dir=save_dir,
                        eval_mode=True,
                        **kwargs,
                    )

                # Include LaTeX processing info in response
                if latex_info:
                    response["latex_processing"] = latex_info

                # Add compression info if tested
                if test_compress:
                    # Calculate compression ratio (simplified version)
                    if hasattr(self._model, "_last_compression_info"):
                        response.update(self._model._last_compression_info)

                # Add file info if saved
                if save_results and save_dir:
                    response["saved_files"] = {
                        "output_dir": save_dir,
                        "result_file": (
                            f"{save_dir}/result.mmd"
                            if os.path.exists(f"{save_dir}/result.mmd")
                            else None
                        ),
                        "annotated_image": (
                            f"{save_dir}/result_with_boxes.jpg"
                            if os.path.exists(f"{save_dir}/result_with_boxes.jpg")
                            else None
                        ),
                    }

                return response

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
                "model_size": model_size,
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
