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
import math
import os
import tempfile
import uuid
from contextlib import suppress
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from ...constants import XINFERENCE_VIDEO_DIR
from ...core.exceptions import InvalidModelRequestError
from ...types import VideoList
from ..image.sglang.core import SGLangDiffusionModel
from ..image.vllm.core import VLLMDiffusionModel
from ..utils import resolve_media_seed
from .core import VideoModelFamilyV2
from .diffusers import DiffusersVideoModel, _resolve_minimax_h3_lightning_config
from .engine_family import VideoEngineModel
from .native import _NativeVideoModel


async def _run_with_request_files(func, *args, **kwargs):
    # A cancelled to_thread await does not stop the engine or encoder thread.
    # Keep its reference/output directory alive until that thread finishes.
    task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await task
        raise


class _MiniMaxH3VideoModel(_NativeVideoModel):
    supported_abilities: Tuple[str, ...] = (
        "text2video",
        "image2video",
        "firstlastframe2video",
    )

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: VideoModelFamilyV2,
        lightning_version: Optional[str] = None,
        lightning_model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._lightning_version = lightning_version
        self._lightning_model_path = lightning_model_path
        model_config = dict(model_spec.default_model_config or {})
        model_config.update(kwargs)
        # Consume Lightning here; the shared image engine rejects these options.
        super().__init__(model_uid, model_path, model_spec, **model_config)
        self._lightning_config = _resolve_minimax_h3_lightning_config(
            model_spec, lightning_version, lightning_model_path
        )

    @classmethod
    def is_model_family_supported(cls, model_family: VideoModelFamilyV2) -> bool:
        return (
            model_family.model_name == "MiniMax-H3"
            and (model_family.engine or "").lower() == cls.engine_name.lower()
        )

    async def text_to_video(
        self,
        prompt: str,
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> VideoList:
        return await self._generate(
            prompt, n, num_inference_steps, response_format, [], **kwargs
        )

    async def image_to_video(
        self,
        image: Image.Image,
        prompt: str,
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> VideoList:
        return await self._generate(
            prompt, n, num_inference_steps, response_format, [image], **kwargs
        )

    async def firstlastframe_to_video(
        self,
        first_frame: Image.Image,
        last_frame: Image.Image,
        prompt: str,
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> VideoList:
        return await self._generate(
            prompt,
            n,
            num_inference_steps,
            response_format,
            [first_frame, last_frame],
            **kwargs,
        )

    def _sampling_config(
        self, num_inference_steps: Optional[int], kwargs: Dict[str, Any]
    ) -> Tuple[int, int, float, Dict[str, Any]]:
        config = dict(self._model_spec.default_generate_config or {})
        config.update({k: v for k, v in kwargs.items() if v is not None})
        for key in ("progressor", "request_id", "negative_prompt", "guidance_scale"):
            config.pop(key, None)
        # H3 generates a joint soundtrack at a fixed 24 fps.
        config.pop("fps", None)
        width, height = config.pop("width", 1344), config.pop("height", 768)
        if width <= 0 or height <= 0:
            raise InvalidModelRequestError("width and height must be positive")
        frames = config.pop("num_frames", 124)
        # A 15-second request is delivered in H3's 362-frame temporal bucket.
        duration = config.pop(
            "duration", min(frames / 24, 15) if frames <= 362 else frames / 24
        )
        if not math.isfinite(duration) or not 4 <= duration <= 15:
            raise InvalidModelRequestError(
                "MiniMax-H3 duration must be between 4 and 15 seconds"
            )
        steps = num_inference_steps
        if steps is None:
            steps = (
                self._lightning_config["num_inference_steps"]
                if self._lightning_config
                else config.get("num_inference_steps", 50)
            )
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise InvalidModelRequestError("num_inference_steps must be at least 1")
        config["num_inference_steps"] = steps
        config.setdefault("flow_shift", 12.0)
        config.setdefault("audio_flow_shift", 3.0)
        if self._lightning_config:
            required = {
                "num_inference_steps": self._lightning_config["num_inference_steps"],
                "flow_shift": self._lightning_config["video_shift"],
                "audio_flow_shift": self._lightning_config["audio_shift"],
            }
            for key, value in required.items():
                if key in kwargs and kwargs[key] is not None and kwargs[key] != value:
                    raise InvalidModelRequestError(
                        f"This Lightning version requires {key}={value}"
                    )
                if key == "num_inference_steps" and steps != value:
                    raise InvalidModelRequestError(
                        f"This Lightning version requires {key}={value}"
                    )
                config[key] = value
        config["seed"] = resolve_media_seed(config.get("seed", -1))
        return width, height, duration, config

    async def _generate(
        self,
        prompt: str,
        n: int,
        num_inference_steps: Optional[int],
        response_format: str,
        images: List[Image.Image],
        **kwargs: Any,
    ) -> VideoList:
        if response_format not in ("url", "b64_json"):
            raise InvalidModelRequestError(
                f"Unsupported response format: {response_format}"
            )
        if n < 1:
            raise InvalidModelRequestError("n must be at least 1")
        progressor = kwargs.get("progressor")
        width, height, duration, config = self._sampling_config(
            num_inference_steps, kwargs
        )
        os.makedirs(XINFERENCE_VIDEO_DIR, exist_ok=True)
        paths: List[str] = []
        try:
            # Reference images and backend output files live for this request only.
            with tempfile.TemporaryDirectory(
                prefix="h3-", dir=XINFERENCE_VIDEO_DIR
            ) as workdir:
                outputs = await self._generate_h3(
                    prompt, n, width, height, duration, config, images, workdir
                )
                if len(outputs) != n:
                    raise RuntimeError(f"Expected {n} videos, received {len(outputs)}")
                for output in outputs:
                    path = os.path.join(XINFERENCE_VIDEO_DIR, uuid.uuid4().hex + ".mp4")
                    paths.append(path)
                    os.replace(output, path)
                result = await _run_with_request_files(
                    DiffusersVideoModel._video_urls_to_response, paths, response_format
                )
            if progressor and progressor.request_id:
                progressor.set_progress(1.0)
            return result
        except BaseException:
            for path in paths:
                with suppress(OSError):
                    os.remove(path)
            raise

    async def _generate_h3(
        self,
        prompt: str,
        n: int,
        width: int,
        height: int,
        duration: float,
        config: Dict[str, Any],
        images: List[Image.Image],
        workdir: str,
    ) -> List[str]:
        raise NotImplementedError


class SGLangMiniMaxH3Model(
    _MiniMaxH3VideoModel, SGLangDiffusionModel, VideoEngineModel
):
    engine_name = "SGLang"
    required_libs = ("sglang",)

    def load(self):
        from importlib.metadata import version

        from packaging.version import Version

        if Version(version("sglang")) < Version("0.5.22"):
            raise ImportError("MiniMax-H3 requires sglang[diffusion]>=0.5.22")
        if self._kwargs.get("model_variant", "fl2va") != "fl2va":
            raise ValueError(
                "Xinference MiniMax-H3 video endpoints require model_variant=fl2va"
            )
        self._kwargs["model_variant"] = "fl2va"
        if self._lightning_config:
            self._kwargs.update(
                lora_path=self._lightning_model_path,
                lora_nickname="h3-lightning",
                lora_scale=1.0,
                lora_alpha=self._lightning_config["lora_alpha"],
                lora_merge_mode="dynamic",
            )
        super().load()

    async def _generate_h3(
        self, prompt, n, width, height, duration, config, images, workdir
    ):
        conditions = []
        for index, image in enumerate(images):
            path = Path(workdir) / f"keyframe-{index}.png"
            image.convert("RGB").save(path)
            conditions.append(
                dict(
                    type="image",
                    uri=path.as_uri(),
                    role="keyframe",
                    frame_index=0 if index == 0 else -1,
                )
            )
        config.update(
            prompt=prompt,
            task="fl2va" if images else "t2va",
            conditions=conditions,
            target=dict(
                short_edge=min(width, height),
                aspect_ratio=f"{width}:{height}",
                duration_seconds=duration,
            ),
            num_outputs_per_prompt=n,
            save_output=True,
            output_path=workdir,
            output_mode="decoded_files",
            return_frames=False,
        )
        # The generic image helper filters against SamplingParams and would drop
        # H3's task, conditions, target and audio scheduler parameters.
        result = await _run_with_request_files(
            self._model.generate, sampling_params_kwargs=config
        )
        results = result if isinstance(result, list) else [result]
        paths = [getattr(item, "output_file_path", None) for item in results]
        if any(not path or not os.path.isfile(path) for path in paths):
            raise RuntimeError("SGLang returned no MiniMax-H3 audio/video file")
        return paths


class VLLMMiniMaxH3Model(_MiniMaxH3VideoModel, VLLMDiffusionModel, VideoEngineModel):
    engine_name = "vLLM"
    generation_mode = "text-to-video"
    required_libs = ("vllm_omni",)

    def load(self):
        from importlib.metadata import version

        from packaging.version import Version

        omni_version = Version(version("vllm-omni"))
        vllm_version = Version(version("vllm"))
        if (
            omni_version < Version("0.32.0")
            or omni_version.release[:2] != vllm_version.release[:2]
        ):
            raise ImportError("MiniMax-H3 requires matching vllm-omni and vllm 0.32.x")
        if self._kwargs.get("task_type", "fl2va") != "fl2va":
            raise ValueError(
                "Xinference MiniMax-H3 video endpoints require task_type=fl2va"
            )
        self._kwargs["task_type"] = "fl2va"
        if self._lightning_config:
            expected = self._model_spec.lightning_model_file_name_template.format(
                lightning_version=self._lightning_config["version"]
            )
            if os.path.basename(self._lightning_model_path) != expected:
                raise ValueError(
                    f"vLLM MiniMax-H3 Lightning requires the published filename {expected}"
                )
            self._kwargs.update(
                lora_backend="peft", lora_path=self._lightning_model_path
            )
        super().load()

    async def _generate_h3(
        self, prompt, n, width, height, duration, config, images, workdir
    ):
        self._raise_if_unavailable()
        extra = dict(config.pop("extra_args", {}))
        extra.update(
            task="fl2va" if images else "t2va",
            duration=duration,
            flow_shift=config.pop("flow_shift"),
            audio_flow_shift=config.pop("audio_flow_shift"),
        )
        payload = {"prompt": prompt, "modalities": ["video"]}
        if images:
            payload["multi_modal_data"] = {"image": images}
            extra["frame_indices"] = [0] if len(images) == 1 else [0, -1]
        if self._lightning_config:
            from vllm_omni.lora.request import LoRARequest
            from vllm_omni.lora.utils import stable_lora_int_id

            config["lora_request"] = LoRARequest(
                lora_name="h3-lightning",
                lora_int_id=stable_lora_int_id(self._lightning_model_path),
                lora_path=self._lightning_model_path,
            )
            config["lora_scale"] = 1.0
        config.update(extra_args=extra, output_type="np")
        params = self._build_sampling_params(n, width, height, config)
        if not self._concurrency_available():
            raise RuntimeError("MiniMax-H3 concurrency requires vllm-omni engine APIs")
        outputs = await self._submit_and_wait(payload, params)
        return await _run_with_request_files(self._save_outputs, outputs, workdir)

    @staticmethod
    def _save_outputs(outputs: List[Any], workdir: str) -> List[str]:
        import numpy as np
        from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

        paths: List[str] = []
        for output in outputs:
            media = getattr(output, "multimodal_output", None) or {}
            if "video" not in media:
                images = getattr(output, "images", None)
                if images and isinstance(images[0], dict):
                    media = images[0]
            videos, audios = media.get("video"), media.get("audio")
            if videos is None or audios is None or len(videos) != len(audios):
                raise RuntimeError(
                    "vLLM returned incomplete MiniMax-H3 video/audio output"
                )
            for frames, audio in zip(videos, audios):
                if audio is None:
                    raise RuntimeError("vLLM returned no MiniMax-H3 audio")
                path = Path(workdir) / f"{len(paths)}.mp4"
                path.write_bytes(
                    mux_video_audio_bytes(
                        np.asarray(frames),
                        np.asarray(audio, dtype=np.float32),
                        fps=24.0,
                        audio_sample_rate=media.get("audio_sample_rate", 32000),
                    )
                )
                paths.append(str(path))
        return paths
