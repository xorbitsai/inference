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
from typing import Any, Dict, Iterator, List, Tuple, Union

from ...llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ...media import (
    load_media_bytes,
    local_path,
    materialize_messages_media,
    media_workspace,
)
from ..core import register_non_default_model
from .minicpmv46 import MiniCPMV46Model


@register_transformer
@register_non_default_model("MiniCPMV4_7ForConditionalGeneration")
class MiniCPMV47Model(MiniCPMV46Model):
    """Native Transformers adapter for MiniCPM-V 4.7, including canvas M-RoPE.

    Keep the processor's complete output for generation. The 4.6 continuous
    batching adapter does not preserve the image/video canvas metadata needed
    by 4.7, so this model uses the shared direct-chat path instead.
    """

    @classmethod
    def match_json(
        cls, model_family: LLMFamilyV2, model_spec: LLMSpecV1, quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not model_family.has_architecture("MiniCPMV4_7ForConditionalGeneration"):
            return False, "Model architecture is not MiniCPM-V-4.7"
        if "vision" not in model_family.model_ability:
            return False, "MiniCPM-V-4.7 transformer requires vision ability"
        if model_spec.model_format not in ("pytorch", "awq", "gptq", "fp8", "bnb"):
            return False, "MiniCPM-V-4.7 transformer requires safetensors weights"
        return True

    def load_processor(self):
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("5.18.0"):
            raise ImportError("MiniCPM-V-4.7 requires transformers>=5.18.0")
        super().load_processor()

    def _get_downsample_mode(self, generate_config: Dict) -> str:
        processor_kwargs = generate_config.get("mm_processor_kwargs") or {}
        mode = processor_kwargs.get(
            "downsample_mode", self._pytorch_model_config.get("downsample_mode", "16x")
        )
        if mode not in ("4x", "16x"):
            raise ValueError("MiniCPM-V-4.7 downsample_mode must be '4x' or '16x'")
        return mode

    def build_inputs_from_messages(self, messages: List[Dict], generate_config: Dict):
        from copy import deepcopy

        messages = deepcopy(self._normalize_messages(messages))
        modalities = {
            item.get("type")
            for message in messages
            if isinstance(message.get("content"), list)
            for item in message["content"]
            if isinstance(item, dict)
        }
        template_kwargs = self._get_chat_template_kwargs_from_generate_config(
            generate_config, self.reasoning_parser
        ) or {"enable_thinking": False}
        kwargs: Dict[str, Any] = {}
        processor_kwargs = generate_config.get("mm_processor_kwargs") or {}
        if "image" in modalities:
            kwargs["images_kwargs"] = {
                "downsample_mode": self._get_downsample_mode(generate_config),
                "max_slice_nums": processor_kwargs.get(
                    "max_slice_nums",
                    self._pytorch_model_config.get("max_slice_nums", 36),
                ),
                "use_image_id": processor_kwargs.get(
                    "use_image_id", self._pytorch_model_config.get("use_image_id", True)
                ),
            }
        if "video" in modalities:
            kwargs["videos_kwargs"] = {
                "downsample_mode": self._get_downsample_mode(generate_config),
                "max_slice_nums": processor_kwargs.get(
                    "max_slice_nums",
                    self._pytorch_model_config.get("video_max_slice_nums", 1),
                ),
                "max_num_frames": processor_kwargs.get(
                    "max_num_frames",
                    self._pytorch_model_config.get("max_num_frames", 128),
                ),
                "stack_frames": processor_kwargs.get(
                    "stack_frames", self._pytorch_model_config.get("stack_frames", 1)
                ),
            }
        with media_workspace(prefix="xinference-minicpmv47-") as workspace:
            # Use the shared downloader so redirects cannot bypass media policy.
            materialize_messages_media(messages, workspace)
            for message in messages:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if not isinstance(item, dict) or item.get("type") != "video":
                        continue
                    url = item.get("url")
                    if isinstance(url, str) and url.startswith("data:"):
                        # Transformers' video reader accepts paths, not data URIs.
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            dir=workspace, suffix=".mp4", delete=False
                        ) as video:
                            video.write(load_media_bytes(url))
                        item["url"] = video.name
                    elif isinstance(url, str) and url.startswith("file:"):
                        item["url"] = local_path(url)
            return self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs=kwargs,
                **template_kwargs,
            ).to(self._model.device)

    def build_streaming_iter(
        self, messages: List[Dict], generate_config: Dict
    ) -> Tuple[Iterator, int]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self.build_inputs_from_messages(messages, generate_config)
        config = self.build_generate_kwargs(generate_config)
        if self._has_visual_content(messages):
            # Placeholder count and vision encoding must use the same mode.
            config["downsample_mode"] = self._get_downsample_mode(generate_config)
        streamer = TextIteratorStreamer(
            self._tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )
        errors: List[Exception] = []

        def generate():
            try:
                self._model.generate(**inputs, **config, streamer=streamer)
            except Exception as exc:
                errors.append(exc)
                streamer.end()

        thread = Thread(target=generate, daemon=True)
        thread.start()

        def iterate():
            yield from streamer
            thread.join()
            if errors:
                raise errors[0]

        return iterate(), len(inputs.input_ids[0])
