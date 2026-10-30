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

from typing import Any, Optional

import PIL.Image

from ...utils import allow_trust_remote_code
from .ovisocr3 import ThreadedOvisOCR3Model


class SGLangOvisOCR3Model(ThreadedOvisOCR3Model):
    required_libs = ("sglang", "transformers")  # type: ignore[assignment]

    def load(self) -> None:
        from sglang import Engine
        from transformers import AutoTokenizer

        model_kwargs = self._kwargs.copy()
        for key in ("cpu_offload", "device_map", "torch_dtype", "attn_implementation"):
            model_kwargs.pop(key, None)
        min_pixels = model_kwargs.pop("min_pixels", self.MIN_PIXELS)
        max_pixels = model_kwargs.pop("max_pixels", self.MAX_PIXELS)
        model_kwargs.setdefault(
            "trust_remote_code", allow_trust_remote_code(self.model_family)
        )
        model_kwargs.setdefault("mem_fraction_static", 0.8)
        model_kwargs.setdefault(
            "mm_process_config",
            {"image": {"min_pixels": min_pixels, "max_pixels": max_pixels}},
        )
        # Load the tokenizer before allocating the engine's worker processes.
        self._processor = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=allow_trust_remote_code(self.model_family),
        )
        try:
            self._model = self._run_on_engine_thread(
                Engine, model_path=self._model_path, **model_kwargs
            )
        except Exception:
            self.stop()
            raise

    def ocr(
        self,
        image: PIL.Image.Image,
        prompt: Optional[str] = None,
        filter_imgtags: bool = True,
        **kwargs: Any,
    ) -> str:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError("Input must be a PIL Image")
        if self._model is None or self._processor is None:
            self.load()
        assert self._model is not None
        assert self._processor is not None
        normalized_prompt = self._normalize_prompt(prompt)
        text = self._processor.apply_chat_template(
            self._messages(normalized_prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        sampling_params = self._generation_kwargs(kwargs)
        sampling_params.pop("use_cache", None)
        do_sample = sampling_params.pop("do_sample", False)
        sampling_params.setdefault("temperature", 1.0 if do_sample else 0.0)
        output = self._run_on_engine_thread(
            self._model.generate,
            prompt=text,
            image_data=image.convert("RGB"),
            sampling_params=sampling_params,
        )
        return self._process_output(output["text"], normalized_prompt, filter_imgtags)
