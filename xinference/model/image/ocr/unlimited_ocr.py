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
import shutil
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import PIL.Image
import torch

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

from ...utils import allow_trust_remote_code
from .ocr_family import OCRModel

logger = logging.getLogger(__name__)


class UnlimitedOCRModelSize:
    """Unlimited-OCR model size configurations.

    Unlimited-OCR supports two image configs (see model README):
      - gundam: base_size=1024, image_size=640, crop_mode=True  (single image)
      - base:   base_size=1024, image_size=1024, crop_mode=False (single / multi)
    Multi-page / PDF parsing only uses base.
    """

    GUNDAM: Tuple[str, int, int, bool] = ("gundam", 1024, 640, True)
    BASE: Tuple[str, int, int, bool] = ("base", 1024, 1024, False)

    _CONFIG_MAP: Dict[str, Tuple[str, int, int, bool]] = {
        "gundam": GUNDAM,
        "base": BASE,
    }

    def __init__(self, size_type: str):
        self.size_type = size_type
        if size_type in self._CONFIG_MAP:
            self.name, self.base_size, self.image_size, self.crop_mode = (
                self._CONFIG_MAP[size_type]
            )
        else:
            # Default to gundam for single-image OCR
            self.name, self.base_size, self.image_size, self.crop_mode = self.GUNDAM

    @classmethod
    def from_string(cls, size_str: str) -> "UnlimitedOCRModelSize":
        return cls(size_str.lower())

    def __str__(self) -> str:
        return self.name


class UnlimitedOCRModel(OCRModel):
    """Unlimited-OCR model for one-shot long-horizon document parsing.

    Built on top of Deepseek-OCR with a DeepseekV2-style MoE language backbone
    and SAM-ViT-B / CLIP-L-14 vision encoders. Loaded via ``trust_remote_code``
    (custom ``modeling_unlimitedocr.UnlimitedOCRForCausalLM``).
    """

    required_libs: Tuple[str, ...] = ("transformers",)

    @classmethod
    def match(cls, model_family: "ImageModelFamilyV2") -> bool:
        return model_family.model_name == "Unlimited-OCR"

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
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        # Compatibility shim for transformers >= 5.x: the bundled
        # ``modeling_deepseekv2.py`` shipped with Unlimited-OCR imports
        # ``is_torch_fx_available`` from ``transformers.utils.import_utils``,
        # which was removed upstream. Inject a stub returning False so the
        # FX-tracing fast path stays disabled (the model never traces).
        from transformers.utils import import_utils as _tf_import_utils

        if not hasattr(_tf_import_utils, "is_torch_fx_available"):
            _tf_import_utils.is_torch_fx_available = lambda: False  # type: ignore[attr-defined]

        logger.info(f"Loading Unlimited-OCR model from {self._model_path}")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=allow_trust_remote_code(self.model_family),
            )
            # The bundled ``modeling_deepseekv2.DeepseekV2Model.__init__``
            # accesses ``config.pad_token_id`` directly. Unlimited-OCR ships
            # a config.json without that field, so prime it from the
            # tokenizer before instantiating the model.
            config = AutoConfig.from_pretrained(
                self._model_path,
                trust_remote_code=allow_trust_remote_code(self.model_family),
            )
            # Unlimited-OCR's config.json nests the DeepseekV2 backbone
            # parameters under ``language_config``. The bundled model code
            # passes ``self.config`` straight to ``DeepseekV2Model.__init__``
            # which reads fields such as ``hidden_size`` directly off the
            # top-level config. Flatten the nested dict onto the wrapper
            # (without overriding anything already set) so those reads
            # succeed.
            language_config = getattr(config, "language_config", None)
            if isinstance(language_config, dict):
                for key, value in language_config.items():
                    if not hasattr(config, key) or getattr(config, key, None) is None:
                        setattr(config, key, value)
            if getattr(config, "pad_token_id", None) is None:
                config.pad_token_id = self._tokenizer.eos_token_id
            # Resolve the target device once so the model lands on the GPU
            # picked by Xinference's scheduler (``self._device``). When no
            # device is configured, choose CUDA only if it is actually available
            # so CPU-only launches do not accidentally try to load on CUDA.
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device != "cpu":
                model = AutoModel.from_pretrained(
                    self._model_path,
                    config=config,
                    trust_remote_code=allow_trust_remote_code(self.model_family),
                    use_safetensors=True,
                    torch_dtype=torch.bfloat16,
                )
                self._model = model.eval().to(device)
            else:
                model = AutoModel.from_pretrained(
                    self._model_path,
                    config=config,
                    trust_remote_code=allow_trust_remote_code(self.model_family),
                    use_safetensors=True,
                    torch_dtype=torch.float32,
                )
                self._model = model.eval()
            # Ensure the model config carries pad_token_id, which the bundled
            # generation code reads as ``model.config.pad_token_id``.
            if getattr(self._model.config, "pad_token_id", None) is None:
                self._model.config.pad_token_id = self._tokenizer.eos_token_id
            logger.info("Unlimited-OCR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Unlimited-OCR model: {e}")
            raise

    def ocr(
        self,
        image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        **kwargs,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Perform document parsing / OCR on single or multiple images.

        Args:
            image: PIL Image or list of PIL Images. A list triggers
                multi-page parsing (``infer_multi``), which only supports the
                ``base`` image config.
            **kwargs: Additional parameters including:
                - prompt: OCR prompt (default: ``"<image>document parsing."``)
                - model_size: Image config for single image, ``"gundam"`` or
                  ``"base"`` (default: ``"gundam"``). Ignored for multi-page.
                - max_length: Max generated tokens (default: 32768)
                - no_repeat_ngram_size: No-repeat ngram size (default: 35)
                - ngram_window: Ngram window size (default: 128 for single,
                  1024 for multi-page)
                - save_results: Whether to save results (default: False)
                - save_dir: Directory to save results (required when
                  ``save_results=True``)

        Returns:
            OCR result dict (single image) or list of dicts (multi image).
        """
        logger.info("Unlimited-OCR kwargs: %s", kwargs)

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        prompt = kwargs.pop("prompt", "<image>document parsing.")
        model_size = kwargs.pop("model_size", "gundam")
        max_length = kwargs.pop("max_length", 32768)
        no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 35)
        save_results = kwargs.pop("save_results", False)
        save_dir = kwargs.pop("save_dir", None)

        if save_results and not save_dir:
            raise ValueError("save_dir must be provided when save_results=True")

        # Single image path
        if isinstance(image, PIL.Image.Image):
            return self._ocr_single(
                image,
                prompt,
                model_size,
                max_length,
                no_repeat_ngram_size,
                save_results,
                save_dir,
                **kwargs,
            )
        # Multi image path -> infer_multi, base config only
        elif isinstance(image, list):
            return self._ocr_multi(
                image,
                prompt,
                max_length,
                no_repeat_ngram_size,
                save_results,
                save_dir,
                **kwargs,
            )
        else:
            raise ValueError("Input must be a PIL Image or list of PIL Images")

    @staticmethod
    def _read_result_md(output_path: str) -> Optional[str]:
        """Read OCR text persisted by the bundled model into ``output_path``.

        ``model.infer`` / ``model.infer_multi`` only stream output to a
        ``transformers`` text streamer and persist the recognized markdown to
        ``output_path/result.md`` when ``save_results=True``. Return its
        contents (or ``None`` if the file is missing).
        """
        md_path = os.path.join(output_path, "result.md")
        if not os.path.exists(md_path):
            return None
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.warning(f"Failed to read OCR result from {md_path}: {e}")
            return None

    def _ocr_single(
        self,
        image: PIL.Image.Image,
        prompt: str,
        model_size: str,
        max_length: int,
        no_repeat_ngram_size: int,
        save_results: bool,
        save_dir: Optional[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """Single-image OCR via ``model.infer``."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        if image.mode in ["RGBA", "CMYK"]:
            image = image.convert("RGB")

        model_config = UnlimitedOCRModelSize.from_string(model_size)
        ngram_window = kwargs.pop("ngram_window", 128)

        temp_image_path = None
        output_path = None
        is_temp_dir = False
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                image.save(f.name, "JPEG")
                temp_image_path = f.name

            if save_results and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                output_path = save_dir
            else:
                output_path = tempfile.mkdtemp()
                is_temp_dir = True

            self._model.infer(
                self._tokenizer,
                prompt=prompt,
                image_file=temp_image_path,
                output_path=output_path,
                base_size=model_config.base_size,
                image_size=model_config.image_size,
                crop_mode=model_config.crop_mode,
                max_length=max_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                ngram_window=ngram_window,
                save_results=True,
            )

            text = self._read_result_md(output_path)

            response = {
                "text": text,
                "model": "unlimited-ocr",
                "success": True,
                "model_size": model_config.name,
                "base_size": model_config.base_size,
                "image_size": model_config.image_size,
                "crop_mode": model_config.crop_mode,
            }

            if save_results and save_dir:
                response["saved_files"] = {
                    "output_dir": save_dir,
                    "result_file": (
                        os.path.join(save_dir, "result.md")
                        if os.path.exists(os.path.join(save_dir, "result.md"))
                        else None
                    ),
                }

            return response
        finally:
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except OSError:
                    pass
            if is_temp_dir and output_path and os.path.exists(output_path):
                shutil.rmtree(output_path, ignore_errors=True)

    def _ocr_multi(
        self,
        images: List[PIL.Image.Image],
        prompt: str,
        max_length: int,
        no_repeat_ngram_size: int,
        save_results: bool,
        save_dir: Optional[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Multi-page OCR via ``model.infer_multi`` (base config only)."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Please call load() first.")

        # Multi-page only supports base config (image_size=1024)
        model_config = UnlimitedOCRModelSize.from_string("base")
        ngram_window = kwargs.pop("ngram_window", 1024)

        temp_paths: List[str] = []
        output_path = None
        is_temp_dir = False
        try:
            for img in images:
                if img.mode in ["RGBA", "CMYK"]:
                    img = img.convert("RGB")
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                    img.save(f.name, "JPEG")
                    temp_paths.append(f.name)

            if save_results and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                output_path = save_dir
            else:
                output_path = tempfile.mkdtemp()
                is_temp_dir = True

            self._model.infer_multi(
                self._tokenizer,
                prompt=prompt,
                image_files=temp_paths,
                output_path=output_path,
                image_size=model_config.image_size,
                max_length=max_length,
                no_repeat_ngram_size=no_repeat_ngram_size,
                ngram_window=ngram_window,
                save_results=True,
            )

            text = self._read_result_md(output_path)

            response = {
                "text": text,
                "model": "unlimited-ocr",
                "success": True,
                "model_size": model_config.name,
                "image_size": model_config.image_size,
                "num_pages": len(temp_paths),
            }

            if save_results and save_dir:
                response["saved_files"] = {
                    "output_dir": save_dir,
                    "result_file": (
                        os.path.join(save_dir, "result.md")
                        if os.path.exists(os.path.join(save_dir, "result.md"))
                        else None
                    ),
                }

            return [response]
        finally:
            for p in temp_paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if is_temp_dir and output_path and os.path.exists(output_path):
                shutil.rmtree(output_path, ignore_errors=True)
