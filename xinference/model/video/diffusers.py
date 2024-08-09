# Copyright 2022-2023 XProbe Inc.
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

import base64
import logging
import os
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import PIL.Image
import torch

from ...constants import XINFERENCE_VIDEO_DIR
from ...device_utils import move_model_to_available_device
from ...types import Video, VideoList

if TYPE_CHECKING:
    from .core import VideoModelFamilyV1


logger = logging.getLogger(__name__)


def export_to_video_imageio(
    video_frames: Union[List[np.ndarray], List["PIL.Image.Image"]],
    output_video_path: str,
    fps: int = 8,
) -> str:
    """
    Export the video frames to a video file using imageio lib to Avoid "green screen" issue (for example CogVideoX)
    """
    import imageio

    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)
    return output_video_path


class DiffUsersVideoModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "VideoModelFamilyV1",
        **kwargs,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._model = None
        self._kwargs = kwargs

    @property
    def model_spec(self):
        return self._model_spec

    def load(self):
        import torch

        torch_dtype = self._kwargs.get("torch_dtype")
        if sys.platform != "darwin" and torch_dtype is None:
            # The following params crashes on Mac M2
            self._kwargs["torch_dtype"] = torch.float16
            self._kwargs["variant"] = "fp16"
            self._kwargs["use_safetensors"] = True
        if isinstance(torch_dtype, str):
            self._kwargs["torch_dtype"] = getattr(torch, torch_dtype)

        if self._model_spec.model_family == "CogVideoX":
            from diffusers import CogVideoXPipeline

            self._model = CogVideoXPipeline.from_pretrained(
                self._model_path, **self._kwargs
            )
        else:
            raise Exception(
                f"Unsupported model family: {self._model_spec.model_family}"
            )

        if self._kwargs.get("cpu_offload", False):
            logger.debug("CPU offloading model")
            self._model.enable_model_cpu_offload()
        elif not self._kwargs.get("device_map"):
            logger.debug("Loading model to available device")
            self._model = move_model_to_available_device(self._model)
        # Recommended if your computer has < 64 GB of RAM
        self._model.enable_attention_slicing()

    def text_to_video(
        self,
        prompt: str,
        n: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: int = 6,
        response_format: str = "b64_json",
        **kwargs,
    ) -> VideoList:
        import gc

        # cv2 bug will cause the video cannot be normally displayed
        # thus we use the imageio one
        # from diffusers.utils import export_to_video
        from ...device_utils import empty_cache

        logger.debug(
            "diffusers text_to_video args: %s",
            kwargs,
        )
        assert self._model is not None
        if self._kwargs.get("cpu_offload"):
            # if enabled cpu offload,
            # the model.device would be CPU
            device = "cuda"
        else:
            device = self._model.device
        prompt_embeds, _ = self._model.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=n,
            max_sequence_length=226,
            device=device,
            dtype=torch.float16,
        )
        assert callable(self._model)
        output = self._model(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            **kwargs,
        )

        # clean cache
        gc.collect()
        empty_cache()

        os.makedirs(XINFERENCE_VIDEO_DIR, exist_ok=True)
        urls = []
        for f in output.frames:
            path = os.path.join(XINFERENCE_VIDEO_DIR, uuid.uuid4().hex + ".mp4")
            p = export_to_video_imageio(f, path, fps=8)
            urls.append(p)
        if response_format == "url":
            return VideoList(
                created=int(time.time()),
                data=[Video(url=url, b64_json=None) for url in urls],
            )
        elif response_format == "b64_json":

            def _gen_base64_video(_video_url):
                try:
                    with open(_video_url, "rb") as f:
                        return base64.b64encode(f.read()).decode()
                finally:
                    os.remove(_video_url)

            with ThreadPoolExecutor() as executor:
                results = list(map(partial(executor.submit, _gen_base64_video), urls))  # type: ignore
                video_list = [Video(url=None, b64_json=s.result()) for s in results]
            return VideoList(created=int(time.time()), data=video_list)
        else:
            raise ValueError(f"Unsupported response format: {response_format}")
