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


@register_batching_multimodal_models("MiniCPM-V-4.6", "MiniCPM-V-4.6-Thinking")
@register_transformer
@register_non_default_model("MiniCPMV4_6ForConditionalGeneration")
class MiniCPMV46Model(PytorchMultiModalModel):
    """Adapter for MiniCPM-V 4.6 / 4.6-Thinking.

    v4.6 is registered as a native architecture (`MiniCPMV4_6ForConditionalGeneration`)
    in transformers>=5.7.0. It uses the standard `AutoProcessor` /
    `AutoModelForImageTextToText` API; the processor's `apply_chat_template`
    fully handles image / video inputs from chat-style messages, so we do not
    need any custom message-to-prompt conversion (unlike v4.5).
    """

    MINICPMV46_ARCHITECTURES = {"MiniCPMV4_6ForConditionalGeneration"}

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        pytorch_model_config = super()._sanitize_model_config(pytorch_model_config)
        assert pytorch_model_config is not None
        # Visual token compression rate. "16x" is the default per the model card;
        # "4x" gives finer detail at the cost of more tokens.
        pytorch_model_config.setdefault("downsample_mode", "16x")
        # Image slicing budget. The model card uses 36 for single-image and 1
        # (combined with use_image_id=False) for video.
        pytorch_model_config.setdefault("max_slice_nums", 36)
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
                "MiniCPM-V-4.6 transformer supports pytorch/gptq/awq/bnb/fp8/fp4 formats only",
            )
        if not model_family.has_architecture(*cls.MINICPMV46_ARCHITECTURES):
            return (
                False,
                f"Model architectures {model_family.architectures} are not MiniCPM-V-4.6",
            )
        if "vision" not in model_family.model_ability:
            return False, "MiniCPM-V-4.6 transformer requires vision ability"
        return True

    def decide_device(self):
        device = self._pytorch_model_config.get("device", "auto")
        device = select_device(device)
        self._device = device

    def load_processor(self):
        from transformers import AutoProcessor

        # MiniCPM-V 4.6 is a native transformers architecture (no remote code).
        # Do NOT pass min_pixels / max_pixels (those are Qwen2-VL parameters).
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        # The processor exposes the underlying tokenizer.
        self._tokenizer = self._processor.tokenizer

    def load_multimodal_model(self):
        from transformers import AutoModelForImageTextToText

        kwargs = self.apply_quantization_config()
        device = "auto" if self._device == "cuda" else self._device

        enable_flash_attn = self._pytorch_model_config.get(
            "enable_flash_attn", is_flash_attn_available()
        )

        if enable_flash_attn:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_2",
                device_map=device,
                **kwargs,
            ).eval()
        elif is_npu_available():
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype="float16",
                **kwargs,
            ).eval()
        elif device == "mps":
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype="bfloat16",
                device_map=device,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            ).eval()
        else:
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map=device,
                **kwargs,
            ).eval()

    def _normalize_messages(self, messages: List[Dict]) -> List[Dict]:
        """Normalize OpenAI-style content into the format MiniCPM-V 4.6 expects.

        The official model card uses ``{"type": "image", "url": "..."}`` and
        ``{"type": "video", "url": "..."}``. xinference clients typically send
        OpenAI-style ``image_url`` / ``video_url``; convert them in-place.
        """
        normalized: List[Dict] = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if not isinstance(item, dict):
                        new_content.append(item)
                        continue
                    item_type = item.get("type")
                    if item_type == "image_url":
                        image_url = item.get("image_url")
                        url = (
                            image_url.get("url", "")
                            if isinstance(image_url, dict)
                            else ""
                        )
                        new_content.append({"type": "image", "url": url})
                    elif item_type == "video_url":
                        video_url = item.get("video_url")
                        url = (
                            video_url.get("url", "")
                            if isinstance(video_url, dict)
                            else ""
                        )
                        new_content.append({"type": "video", "url": url})
                    else:
                        new_content.append(item)
                normalized.append({**msg, "content": new_content})
            else:
                normalized.append(msg)
        return normalized

    def _has_visual_content(self, messages: List[Dict]) -> bool:
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in (
                        "image",
                        "image_url",
                        "video",
                        "video_url",
                    ):
                        return True
        return False

    def build_inputs_from_messages(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ):
        messages = self._normalize_messages(messages)

        # Only pass downsample_mode / max_slice_nums when there is actual
        # visual content; otherwise the image processor is not invoked and
        # passing these kwargs raises in pure-text turns.
        kwargs: Dict[str, Any] = {}
        if self._has_visual_content(messages):
            kwargs["downsample_mode"] = self._pytorch_model_config.get(
                "downsample_mode", "16x"
            )
            kwargs["max_slice_nums"] = self._pytorch_model_config.get(
                "max_slice_nums", 36
            )

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            **kwargs,
        ).to(self._model.device)
        return inputs

    def build_generate_kwargs(self, generate_config: Dict) -> Dict[str, Any]:
        max_new_tokens = generate_config.get("max_tokens") or 512
        temperature = generate_config.get("temperature", 0.7)
        top_p = generate_config.get("top_p", 0.8)
        top_k = generate_config.get("top_k", 100)
        repetition_penalty = generate_config.get("repetition_penalty", 1.05)

        # Note: do NOT include downsample_mode here. If the inputs were built
        # without visual content, model.generate() should not receive that
        # kwarg either (passing it triggers vision-tower paths).
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }

    def build_streaming_iter(
        self,
        messages: List[Dict],
        generate_config: Dict,
    ) -> Tuple[Iterator, int]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer,
            timeout=60.0,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        normalized = self._normalize_messages(messages)
        has_visual = self._has_visual_content(normalized)

        inputs = self.build_inputs_from_messages(messages, generate_config)
        config = self.build_generate_kwargs(generate_config)
        if has_visual:
            config["downsample_mode"] = self._pytorch_model_config.get(
                "downsample_mode", "16x"
            )

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
        raw_config = req.inference_kwargs.get("raw_params", {})
        raw_config.setdefault("temperature", 0.7)
        raw_config.setdefault("top_p", 0.8)
        raw_config.setdefault("top_k", 100)
        raw_config.setdefault("repetition_penalty", 1.05)
        return raw_config

    def _get_full_prompt(self, messages: List[Dict], tools, generate_config: dict):
        # Continuous-batching path: return the normalized messages and let
        # build_prefill_kwargs build the actual tensors via the processor.
        return self._normalize_messages(messages)

    def build_prefill_kwargs(self, prompts: List, req_list: List[InferenceRequest]):
        import torch

        downsample_mode = self._pytorch_model_config.get("downsample_mode", "16x")
        max_slice_nums = self._pytorch_model_config.get("max_slice_nums", 36)

        # `prompts` is a list of normalized message lists, one per request.
        # A batch can mix text-only and multimodal requests, so the
        # downsample_mode / max_slice_nums kwargs must be conditioned on
        # whether each individual request has visual content.
        all_inputs = []
        visual_flags = []
        for messages in prompts:
            has_visual = self._has_visual_content(messages)
            visual_flags.append(has_visual)
            kwargs: Dict[str, Any] = {}
            if has_visual:
                kwargs["downsample_mode"] = downsample_mode
                kwargs["max_slice_nums"] = max_slice_nums
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **kwargs,
            )
            all_inputs.append(inputs)

        # Pad input_ids on the left to the longest sequence.
        max_len = max(int(x["input_ids"].shape[-1]) for x in all_inputs)
        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        input_ids_list = []
        attention_mask_list = []
        for r, inputs in zip(req_list, all_inputs):
            ids = inputs["input_ids"][0]
            real_len = int(ids.shape[-1])
            pad_len = max_len - real_len
            padded = torch.cat(
                [torch.full((pad_len,), pad_id, dtype=ids.dtype), ids], dim=0
            )
            mask = torch.cat(
                [
                    torch.zeros((pad_len,), dtype=torch.long),
                    torch.ones((real_len,), dtype=torch.long),
                ],
                dim=0,
            )
            input_ids_list.append(padded)
            attention_mask_list.append(mask)
            r.prompt_tokens = ids.tolist()
            r.padding_len = pad_len
            r.extra_kwargs["attention_mask_seq_len"] = real_len

        device = self._model.device
        merged: Dict[str, Any] = {
            "input_ids": torch.stack(input_ids_list).to(device),
            "attention_mask": torch.stack(attention_mask_list).to(device),
        }

        # Concatenate multimodal tensors across all requests in the batch.
        # If a batch has multiple multimodal requests, simply taking the
        # tensors from the first one would silently drop the rest, leading
        # to wrong outputs. Concatenate along the leading dimension so the
        # vision tower sees every request's visual tokens.
        #
        # Fail fast on incompatible shapes: continuing with only the first
        # request's tensor would let later requests in the batch run
        # against the wrong (or missing) visual data, which is far worse
        # than rejecting the batch outright. The caller is expected to
        # split / retry incompatible multimodal requests one-by-one.
        multimodal_keys = ("pixel_values", "image_grid_thw", "image_sizes", "tgt_sizes")
        for key in multimodal_keys:
            tensors = [inputs[key] for inputs in all_inputs if key in inputs]
            if not tensors:
                continue
            try:
                merged[key] = torch.cat(tensors, dim=0).to(device)
            except (RuntimeError, TypeError) as e:
                shapes = [tuple(t.shape) for t in tensors]
                raise RuntimeError(
                    f"Cannot batch MiniCPM-V-4.6 multimodal tensor {key!r} "
                    f"across requests: incompatible shapes {shapes} "
                    f"({e}). Disable continuous batching for this request "
                    f"or split incompatible visual inputs into separate "
                    f"batches."
                ) from e

        batch_size, seq_len = merged["input_ids"].shape
        position_ids = self.build_prefill_position_ids(batch_size, seq_len, req_list)
        if position_ids is not None:
            merged["position_ids"] = position_ids
        return merged
