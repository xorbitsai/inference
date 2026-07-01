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
import platform
import sys
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import PIL.Image

from .deepseek_ocr import DeepSeekOCRModel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2


class MLXDeepSeekOCRModel(DeepSeekOCRModel):
    required_libs: Tuple[str, ...] = ("mlx_vlm", "mlx")

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs,
    ):
        super().__init__(model_uid, model_path, device, model_spec, **kwargs)
        self._processor: Optional[Any] = None

    @classmethod
    def match(cls, model_family) -> bool:
        model_format = getattr(model_family, "model_format", None)
        return model_family.model_name == "DeepSeek-OCR" and model_format == "mlx"

    @classmethod
    def check_lib(cls):
        if sys.platform != "darwin" or platform.processor() != "arm":
            return False, "MLX engine is only supported on Apple silicon Macs."
        return super().check_lib()

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

        try:
            import mlx.core as mx

            orig_uniform = mx.random.uniform
            orig_zeros = mx.zeros

            def _uniform(*args, **kwargs):
                def _normalize_shape(shape):
                    if shape is None:
                        return []
                    if isinstance(shape, dict):
                        if "shape" in shape:
                            return _normalize_shape(shape.get("shape"))
                        if "size" in shape:
                            return _normalize_shape(shape.get("size"))
                        return []
                    if isinstance(shape, (list, tuple)):
                        normalized = []
                        for value in shape:
                            if isinstance(value, dict):
                                extracted = _normalize_shape(value)
                                if extracted:
                                    return extracted
                                return []
                            try:
                                normalized.append(int(value))
                            except Exception:
                                return []
                        return normalized
                    if hasattr(shape, "tolist"):
                        try:
                            return [int(x) for x in shape.tolist()]
                        except Exception:
                            return shape
                    return shape

                if kwargs:
                    dtype = kwargs.get("dtype") or mx.float32
                    low = kwargs.get("low", 0)
                    high = kwargs.get("high", 1)
                    shape = _normalize_shape(kwargs.get("shape", []))
                    key = kwargs.get("key", None)
                    stream = kwargs.get("stream", None)
                    call_kwargs = {"low": low, "high": high, "shape": shape}
                    if key is not None:
                        call_kwargs["key"] = key
                    if stream is not None:
                        call_kwargs["stream"] = stream
                    try:
                        call_kwargs["dtype"] = dtype
                        return orig_uniform(**call_kwargs)
                    except TypeError:
                        call_kwargs.pop("dtype", None)
                        try:
                            return orig_uniform(**call_kwargs)
                        except TypeError:
                            import numpy as np

                            return mx.array(
                                np.random.uniform(low, high, size=shape),
                                dtype=dtype,
                            )

                if args:
                    if len(args) == 3:
                        low, high, shape = args
                        shape = _normalize_shape(shape)
                        try:
                            return orig_uniform(low, high, shape, mx.float32)
                        except TypeError:
                            import numpy as np

                            return mx.array(
                                np.random.uniform(low, high, size=shape),
                                dtype=mx.float32,
                            )
                    if len(args) == 4:
                        low, high, shape, dtype = args
                        shape = _normalize_shape(shape)
                        try:
                            return orig_uniform(low, high, shape, dtype)
                        except TypeError:
                            import numpy as np

                            return mx.array(
                                np.random.uniform(low, high, size=shape),
                                dtype=dtype,
                            )
                    raise TypeError(
                        f"mlx.random.uniform unsupported positional args: {args}"
                    )

                return orig_uniform(dtype=mx.float32)

            mx.random.uniform = _uniform

            def _zeros(shape, dtype=None, stream=None):
                def _normalize_shape(value):
                    if value is None:
                        return []
                    if isinstance(value, dict):
                        if "shape" in value:
                            return _normalize_shape(value.get("shape"))
                        if "size" in value:
                            return _normalize_shape(value.get("size"))
                        return []
                    if isinstance(value, (list, tuple)):
                        normalized = []
                        for item in value:
                            if isinstance(item, dict):
                                extracted = _normalize_shape(item)
                                if extracted:
                                    return extracted
                                return []
                            try:
                                normalized.append(int(item))
                            except Exception:
                                return []
                        if len(normalized) == 1:
                            return normalized[0]
                        return normalized
                    try:
                        return int(value)
                    except Exception:
                        return value

                shape = _normalize_shape(shape)
                if dtype is None:
                    dtype = mx.float32
                try:
                    return orig_zeros(shape, dtype=dtype, stream=stream)
                except TypeError:
                    return orig_zeros(shape)

            mx.zeros = _zeros
        except Exception:
            logger.debug("mlx random.uniform patch skipped.")

        try:
            from mlx_vlm import utils as mlx_utils

            orig_load_config = mlx_utils.load_config

            def _patched_load_config(model_path, **kwargs):
                config = orig_load_config(model_path, **kwargs)
                vision_config = config.get("vision_config", {})
                width_config = vision_config.get("width")
                if isinstance(width_config, dict):
                    preferred_key = vision_config.get("model_name") or "clip-l-14-224"
                    if preferred_key not in width_config:
                        preferred_key = next(iter(width_config))
                    selected = width_config.get(preferred_key, {})
                    vision_config = dict(vision_config)
                    vision_config["width"] = selected.get(
                        "width", vision_config.get("width")
                    )
                    vision_config["layers"] = selected.get(
                        "layers", vision_config.get("layers")
                    )
                    if "patch_size" in selected:
                        vision_config["patch_size"] = selected["patch_size"]
                    if "image_size" in selected:
                        vision_config["image_size"] = selected["image_size"]
                    if "heads" in selected:
                        vision_config["num_attention_heads"] = selected["heads"]
                    config["vision_config"] = vision_config
                return config

            mlx_utils.load_config = _patched_load_config
            self._model, self._processor = load(self._model_path)
        finally:
            try:
                mlx_utils.load_config = orig_load_config
            except Exception:
                pass
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

        processor = self._processor
        assert processor is not None
        detokenizer = processor.detokenizer
        tokenizer = processor.tokenizer
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
