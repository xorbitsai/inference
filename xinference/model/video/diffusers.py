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

import logging
import os
import sys
import uuid
import torch
import time

from ...constants import XINFERENCE_VIDEO_DIR
from ...device_utils import move_model_to_available_device
from ...types import VideoList

logger = logging.getLogger(__name__)


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
                self._model_path, torch_dtype=torch.float16
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

    def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        **kwargs,
    ) -> VideoList:
        from diffusers.utils import export_to_video

        logger.debug(
            "diffusers args: %s",
            kwargs,
        )
        # assert callable(self._model)
        prompt_embeds, _ = self._model.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=n,
            max_sequence_length=226,
            device=self._model.device,
            dtype=torch.float16,
        )
        output = self._model(
            num_inference_steps=50,
            guidance_scale=6,
            prompt_embeds=prompt_embeds,
        )
        urls = []
        for f in output.frames:
            path = os.path.join(XINFERENCE_VIDEO_DIR, uuid.uuid4().hex + ".jpg")
            p = export_to_video(f, path, fps=8)
            urls.append(p)
        return VideoList(created=int(time.time()), data=urls)
