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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from .....core.model import register_batching_multimodal_models
from .....device_utils import is_npu_available
from .....types import PytorchModelConfig
from ....scheduler.request import InferenceRequest
from ....utils import is_flash_attn_available, select_device
from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ..core import register_non_default_model
from .core import PytorchMultiModalModel

logger = logging.getLogger(__name__)


@register_batching_multimodal_models(
    "qwen2-vl-instruct",
    "qwen2.5-vl-instruct",
    "QvQ-72B-Preview",
    "Qwen3-VL-Instruct",
    "Qwen3-VL-Thinking",
)
@register_transformer
@register_non_default_model(
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
)
class Qwen2VLChatModel(PytorchMultiModalModel):
    QWEN2_VL_ARCHITECTURES = {
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
    }

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        pytorch_model_config = super()._sanitize_model_config(pytorch_model_config)
        assert pytorch_model_config is not None
        pytorch_model_config.setdefault("min_pixels", 256 * 28 * 28)
        pytorch_model_config.setdefault("max_pixels", 1280 * 28 * 28)
        return pytorch_model_config

    @classmethod
    def match_json(
        cls, model_family: "LLMFamilyV2", model_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if model_spec.model_format not in [
            "pytorch",
            "gptq",
            "awq",
            "bnb",
            "fp8",
            "fp4",
        ]:
            return (
                False,
                "Qwen2 VL transformer supports pytorch/gptq/awq/bnb/fp8/fp4 formats only",
            )
        if not model_family.has_architecture(*cls.QWEN2_VL_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not a supported Qwen2/3 VL variant",
            )
        if "vision" not in model_family.model_ability:
            return False, "Qwen2 VL transformer requires vision ability"
        return True

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        # for multiple GPU, set back to auto to make multiple devices work
        self._device = device

    def load_processor(self):
        from transformers import AutoProcessor

        min_pixels = self._pytorch_model_config.get("min_pixels")
        max_pixels = self._pytorch_model_config.get("max_pixels")
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self._tokenizer = self._processor.tokenizer

    def load_multimodal_model(self):
        from transformers import Qwen2VLForConditionalGeneration

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
        except ImportError:
            Qwen2_5_VLForConditionalGeneration = None

        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            AutoModelForImageTextToText = None

        kwargs = self.apply_quantization_config()
        if self.model_family.has_architecture("Qwen2_5_VLForConditionalGeneration"):
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif self.model_family.has_architecture("Qwen3VLMoeForConditionalGeneration"):
            model_cls = AutoModelForImageTextToText
        else:
            model_cls = Qwen2VLForConditionalGeneration
        if model_cls is None:
            raise ImportError("`transformers` version is too old, please upgrade it")
        device = "auto" if self._device == "cuda" else self._device

        enable_flash_attn = self._pytorch_model_config.get(
            "enable_flash_attn", is_flash_attn_available()
        )

        if enable_flash_attn:
            self._model = model_cls.from_pretrained(
                self.model_path,
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_2",
                device_map=device,
                trust_remote_code=True,
                **kwargs,
            ).eval()
        elif is_npu_available():
            # Ascend do not support bf16
            self._model = model_cls.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="float16",
                **kwargs,
            ).eval()
        elif device == "mps":
            # MacOS special, see https://github.com/QwenLM/Qwen2.5-VL/issues/761
            self._model = model_cls.from_pretrained(
                self.model_path,
                torch_dtype="bfloat16",
                device_map=device,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()
        else:
            self._model = model_cls.from_pretrained(
                self.model_path,
                device_map=device,
                trust_remote_code=True,
                **kwargs,
            ).eval()

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        from qwen_vl_utils import process_vision_info

        messages = self._transform_messages(messages)
        # Preparation for inference
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._device)
        return inputs

    def build_generate_kwargs(self, generate_config: Dict) -> Dict[str, Any]:
        max_new_tokens = generate_config.get("max_tokens") or 512
        temperature = generate_config.get("temperature", 1)
        return {"max_new_tokens": max_new_tokens, "temperature": temperature}

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        tokenizer = self._tokenizer
        streamer = TextIteratorStreamer(
            tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )

        inputs = self.build_inputs_from_messages(messages, generate_config)
        config = self.build_generate_kwargs(generate_config)

        def model_generate():
            try:
                return self._model.generate(**inputs, **config, streamer=streamer)
            except Exception:
                streamer.end()
                raise

        thread = Thread(target=model_generate)
        thread.start()
        return streamer, len(inputs.input_ids[0])

    def prepare_sanitize_generate_config(self, req: InferenceRequest):
        """
        This file corresponds to multiple models,
        so the corresponding configuration is read directly through the transformers interface.
        """
        from transformers import GenerationConfig

        gen_config = GenerationConfig.from_pretrained(self.model_path).to_dict()
        raw_config = req.inference_kwargs.get("raw_params", {})
        gen_config.update(raw_config)
        return gen_config

    def _get_full_prompt(self, messages: List[Dict], tools, generate_config: dict):
        return self._transform_messages(messages)

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        import torch
        from qwen_vl_utils import process_vision_info

        batch_text = self._processor.apply_chat_template(
            prompts, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(prompts)
        inputs = self._processor(
            text=batch_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        for r, _ids, attn_mask in zip(
            req_list, inputs["input_ids"], inputs["attention_mask"]
        ):
            r.prompt_tokens = _ids.tolist()
            real_len = torch.sum(attn_mask).item()
            r.padding_len = attn_mask.numel() - real_len
            r.extra_kwargs["attention_mask_seq_len"] = real_len
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        position_ids = self.build_prefill_position_ids(batch_size, seq_len, req_list)
        return {**inputs, "position_ids": position_ids}
