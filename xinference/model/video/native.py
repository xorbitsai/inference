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

import asyncio
import os
import platform
import uuid
from typing import Any, List, Optional

from PIL import Image

from ...constants import XINFERENCE_VIDEO_DIR
from ...types import VideoList
from ..image.sglang.core import SGLangDiffusionModel
from ..image.vllm.core import VLLMDiffusionModel
from ..utils import has_cuda_device
from .core import VideoModelFamilyV2
from .diffusers import DiffusersVideoModel
from .engine_family import VideoEngineModel

NATIVE_VIDEO_MODELS = {"Wan2.1-1.3B", "Wan2.1-14B", "Wan2.2-A14B"}


def _export_videos(videos: List[Any], fps: int, response_format: str) -> VideoList:
    from diffusers.utils import export_to_video

    os.makedirs(XINFERENCE_VIDEO_DIR, exist_ok=True)
    paths = []
    try:
        for frames in videos:
            path = os.path.join(XINFERENCE_VIDEO_DIR, uuid.uuid4().hex + ".mp4")
            paths.append(path)
            export_to_video(frames, path, fps=fps)
        return DiffusersVideoModel._video_urls_to_response(paths, response_format)
    except BaseException:
        for path in paths:
            if os.path.exists(path):
                os.remove(path)
        raise


class _NativeVideoModel:
    engine_name: str
    _model_spec: Any
    supported_abilities = ("text2video",)
    engine_model_format = "diffusers"
    engine_quantization = "none"

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: VideoModelFamilyV2,
        lightning_version: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if lightning_version:
            raise ValueError(
                "Lightning adapters are not supported by native video engines"
            )
        super().__init__(model_uid, model_path, model_spec=model_spec, **kwargs)  # type: ignore[call-arg]

    @classmethod
    def check_host(cls):
        if platform.system() != "Linux" or not has_cuda_device():
            return False, "Native diffusion video engines require Linux and CUDA"
        return True

    @classmethod
    def is_model_family_supported(cls, model_family: VideoModelFamilyV2) -> bool:
        return (
            model_family.model_name in NATIVE_VIDEO_MODELS
            and (model_family.engine or "").lower() == cls.engine_name.lower()
        )

    @classmethod
    def match(cls, model_family: VideoModelFamilyV2) -> bool:
        return cls.is_model_family_supported(model_family) and cls.check_host() is True

    async def _generate_videos(
        self, prompt: str, n: int, width: int, height: int, config: dict
    ) -> List[Any]:
        raise NotImplementedError

    async def text_to_video(
        self,
        prompt: str,
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> VideoList:
        if response_format not in ("url", "b64_json"):
            raise ValueError(f"Unsupported response format: {response_format}")
        if n < 1:
            raise ValueError("n must be at least 1")
        config = (self._model_spec.default_generate_config or {}).copy()
        config.update({k: v for k, v in kwargs.items() if v is not None})
        config.pop("progressor", None)
        config.pop("request_id", None)
        if num_inference_steps is not None:
            config["num_inference_steps"] = num_inference_steps
        width, height = config.pop("width", 832), config.pop("height", 480)
        fps = config.pop("fps", 16)
        videos = await self._generate_videos(prompt, n, width, height, config)
        if len(videos) != n:
            raise RuntimeError(f"Expected {n} videos, received {len(videos)}")
        return await asyncio.to_thread(_export_videos, videos, fps, response_format)


class SGLangVideoModel(_NativeVideoModel, SGLangDiffusionModel, VideoEngineModel):
    engine_name = "SGLang"
    required_libs = ("sglang",)

    async def _generate_videos(self, prompt, n, width, height, config):
        params = self._build_sampling_params(prompt, n, width, height, config)
        result = await asyncio.to_thread(
            self._model.generate, sampling_params_kwargs=params
        )
        results = result if isinstance(result, list) else [result]
        videos = []
        for item in results:
            frames = getattr(item, "frames", None)
            if frames is None or len(frames) == 0:
                raise RuntimeError("SGLang returned no video frames")
            videos.append(frames)
        return videos


class VLLMVideoModel(_NativeVideoModel, VLLMDiffusionModel, VideoEngineModel):
    engine_name = "vLLM"
    generation_mode = "text-to-video"
    required_libs = ("vllm_omni",)

    async def _generate_videos(self, prompt, n, width, height, config):
        self._raise_if_unavailable()
        payload = {"prompt": prompt, "modalities": ["video"]}
        negative_prompt = config.pop("negative_prompt", None)
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        # Wan's native postprocessor returns one list of PIL frames per video.
        config["output_type"] = "pil"
        params = self._build_sampling_params(n, width, height, config)
        if not self._concurrency_available():
            raise RuntimeError("Video concurrency requires vllm-omni 0.28 engine APIs")
        outputs = await self._submit_and_wait(payload, params)
        videos = []
        for output in outputs:
            frames = getattr(output, "images", None)
            if frames is None or len(frames) == 0:
                frames = (getattr(output, "multimodal_output", None) or {}).get("video")
            if frames is not None and len(frames):
                if isinstance(frames[0], Image.Image):
                    videos.append(frames)
                else:
                    videos.extend(frames)
        return videos
