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

import base64
import importlib
import json
import logging
import operator
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import PIL.Image

from ...constants import XINFERENCE_VIDEO_DIR
from ...device_utils import gpu_count, move_model_to_available_device
from ...types import Video, VideoList

if TYPE_CHECKING:
    from ...core.progress_tracker import Progressor
    from .core import VideoModelFamilyV2


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


class DiffusersVideoModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "VideoModelFamilyV2",
        gguf_model_path: Optional[str] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._model = None
        self._kwargs = kwargs
        self._gguf_model_path = gguf_model_path

    @property
    def model_spec(self):
        return self._model_spec

    @property
    def model_ability(self):
        return self._abilities

    def _get_layer_cls(self, layer: str):
        with open(os.path.join(self._model_path, "model_index.json")) as f:
            model_index = json.load(f)
            layer_info = model_index[layer]
            module_name, class_name = layer_info
            module = importlib.import_module(module_name)
            return getattr(module, class_name)

    def _load_transformer_gguf(self, torch_dtype):
        from diffusers import GGUFQuantizationConfig

        logger.debug("Loading gguf transformer from %s", self._gguf_model_path)
        return self._get_layer_cls("transformer").from_single_file(
            self._gguf_model_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            torch_dtype=torch_dtype,
            config=os.path.join(self._model_path, "transformer"),
        )

    @staticmethod
    def _register_transformer(pipeline, transformer):
        if transformer is None:
            return
        if hasattr(pipeline, "register_modules"):
            pipeline.register_modules(transformer=transformer)
        else:
            pipeline.transformer = transformer

    def load(self):
        import torch

        kwargs = self._model_spec.default_model_config.copy()
        kwargs.update(self._kwargs)

        scheduler_cls_name = kwargs.pop("scheduler", None)

        torch_dtype = kwargs.get("torch_dtype")
        if isinstance(torch_dtype, str):
            kwargs["torch_dtype"] = getattr(torch, torch_dtype)
            torch_dtype = kwargs["torch_dtype"]
        logger.debug("Loading video model with kwargs: %s", kwargs)

        transformer = None
        if self._gguf_model_path and self._model_spec.model_family != "HunyuanVideo":
            transformer = self._load_transformer_gguf(torch_dtype)

        if self._model_spec.model_family == "CogVideoX":
            import diffusers
            from diffusers import CogVideoXPipeline

            pipeline = self._model = CogVideoXPipeline.from_pretrained(
                self._model_path, **kwargs
            )
            self._register_transformer(pipeline, transformer)
        elif self._model_spec.model_family == "HunyuanVideo":
            from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

            transformer_torch_dtype = kwargs.pop("transformer_torch_dtype", None)
            if isinstance(transformer_torch_dtype, str):
                transformer_torch_dtype = getattr(torch, transformer_torch_dtype)
            if transformer_torch_dtype is None:
                transformer_torch_dtype = torch_dtype
            if self._gguf_model_path:
                transformer = self._load_transformer_gguf(transformer_torch_dtype)
            else:
                transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                    self._model_path,
                    subfolder="transformer",
                    torch_dtype=transformer_torch_dtype,
                )
            pipeline = self._model = HunyuanVideoPipeline.from_pretrained(
                self._model_path, transformer=transformer, **kwargs
            )
        elif self.model_spec.model_family == "Wan":
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
            from transformers import CLIPVisionModel

            if "text2video" in self.model_spec.model_ability:
                pipeline = self._model = WanPipeline.from_pretrained(
                    self._model_path, **kwargs
                )
                self._register_transformer(pipeline, transformer)
            else:
                assert (
                    "image2video" in self.model_spec.model_ability
                    or "firstlastframe2video" in self.model_spec.model_ability
                )

                image_encoder = CLIPVisionModel.from_pretrained(
                    self._model_path,
                    subfolder="image_encoder",
                    torch_dtype=torch.float32,
                )
                vae = AutoencoderKLWan.from_pretrained(
                    self._model_path, subfolder="vae", torch_dtype=torch.float32
                )
                pipeline = self._model = WanImageToVideoPipeline.from_pretrained(
                    self._model_path, vae=vae, image_encoder=image_encoder, **kwargs
                )
                self._register_transformer(pipeline, transformer)
        else:
            raise Exception(
                f"Unsupported model family: {self._model_spec.model_family}"
            )

        if scheduler_cls_name:
            logger.debug("Using scheduler: %s", scheduler_cls_name)
            pipeline.scheduler = getattr(diffusers, scheduler_cls_name).from_config(
                pipeline.scheduler.config, timestep_spacing="trailing"
            )
        if kwargs.get("compile_graph", False):
            pipeline.transformer = torch.compile(
                pipeline.transformer, mode="max-autotune", fullgraph=True
            )
        if kwargs.get("layerwise_cast", False):
            compute_dtype = pipeline.transformer.dtype
            pipeline.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=compute_dtype
            )
        if kwargs.get("cpu_offload", False):
            logger.debug("CPU offloading model")
            pipeline.enable_model_cpu_offload()
            if kwargs.get("sequential_cpu_offload", True):
                pipeline.enable_sequential_cpu_offload()
            try:
                pipeline.vae.enable_slicing()
            except AttributeError:
                # model does not support slicing
                pass
            try:
                pipeline.vae.enable_tiling()
            except AttributeError:
                # model does support tiling
                pass
        elif kwargs.get("group_offload", False):
            from diffusers.hooks.group_offloading import apply_group_offloading

            onload_device = torch.device("cuda")
            offload_device = torch.device("cpu")

            apply_group_offloading(
                pipeline.text_encoder,
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="block_level",
                num_blocks_per_group=4,
            )
            group_offload_kwargs = {}
            if kwargs.get("use_stream", False):
                group_offload_kwargs["offload_type"] = "block_level"
                group_offload_kwargs["num_blocks_per_group"] = 4
            else:
                group_offload_kwargs["offload_type"] = "leaf_level"
                group_offload_kwargs["use_stream"] = True
            pipeline.transformer.enable_group_offload(
                onload_device=onload_device,
                offload_device=offload_device,
                **group_offload_kwargs,
            )
            # Since we've offloaded the larger models already, we can move the rest of the model components to GPU
            pipeline = move_model_to_available_device(pipeline)
        elif not kwargs.get("device_map"):
            logger.debug("Loading model to available device")
            if gpu_count() > 1:
                kwargs["device_map"] = "balanced"
            else:
                pipeline = move_model_to_available_device(self._model)
        # Recommended if your computer has < 64 GB of RAM
        pipeline.enable_attention_slicing()

    @staticmethod
    def _process_progressor(kwargs: dict):
        import diffusers

        progressor: Progressor = kwargs.pop("progressor", None)

        def report_status_callback(
            pipe: diffusers.DiffusionPipeline,
            step: int,
            timestep: int,
            callback_kwargs: dict,
        ):
            num_steps = pipe.num_timesteps
            progressor.set_progress((step + 1) / num_steps)

            return callback_kwargs

        if progressor and progressor.request_id:
            kwargs["callback_on_step_end"] = report_status_callback

    def text_to_video(
        self,
        prompt: str,
        n: int = 1,
        num_inference_steps: int = 50,
        response_format: str = "b64_json",
        **kwargs,
    ) -> VideoList:
        assert self._model is not None
        assert callable(self._model)
        generate_kwargs = self._model_spec.default_generate_config.copy()
        generate_kwargs.update(kwargs)
        generate_kwargs["num_videos_per_prompt"] = n
        fps = generate_kwargs.pop("fps", 10)
        logger.debug(
            "diffusers text_to_video args: %s",
            generate_kwargs,
        )
        self._process_progressor(generate_kwargs)
        output = self._model(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            **generate_kwargs,
        )
        return self._output_to_video(output, fps, response_format)

    def image_to_video(
        self,
        image: PIL.Image.Image,
        prompt: str,
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs,
    ):
        assert self._model is not None
        assert callable(self._model)
        generate_kwargs = self._model_spec.default_generate_config.copy()
        generate_kwargs.update(kwargs)
        generate_kwargs["num_videos_per_prompt"] = n
        if num_inference_steps:
            generate_kwargs["num_inference_steps"] = num_inference_steps
        fps = generate_kwargs.pop("fps", 10)

        # process image
        max_area = generate_kwargs.pop("max_area")
        if isinstance(max_area, str):
            max_area = [int(v) for v in max_area.split("*")]
        max_area = reduce(operator.mul, max_area, 1)
        image = self._process_image(image, max_area)

        height, width = image.height, image.width
        generate_kwargs.pop("width", None)
        generate_kwargs.pop("height", None)
        self._process_progressor(generate_kwargs)
        output = self._model(
            image=image, prompt=prompt, height=height, width=width, **generate_kwargs
        )
        return self._output_to_video(output, fps, response_format)

    def firstlastframe_to_video(
        self,
        first_frame: PIL.Image.Image,
        last_frame: PIL.Image.Image,
        prompt: str,
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs,
    ):
        assert self._model is not None
        assert callable(self._model)
        generate_kwargs = self._model_spec.default_generate_config.copy()
        generate_kwargs.update(kwargs)
        generate_kwargs["num_videos_per_prompt"] = n
        if num_inference_steps:
            generate_kwargs["num_inference_steps"] = num_inference_steps
        fps = generate_kwargs.pop("fps", 10)

        # process first and last frame
        max_area = generate_kwargs.pop("max_area")
        if isinstance(max_area, str):
            max_area = [int(v) for v in max_area.split("*")]
        max_area = reduce(operator.mul, max_area, 1)
        first_frame = self._process_image(first_frame, max_area)
        width, height = first_frame.size
        if last_frame.size != first_frame.size:
            last_frame = self._center_crop_resize(last_frame, height, width)

        generate_kwargs.pop("width", None)
        generate_kwargs.pop("height", None)
        self._process_progressor(generate_kwargs)
        output = self._model(
            image=first_frame,
            last_image=last_frame,
            prompt=prompt,
            height=height,
            width=width,
            **generate_kwargs,
        )
        return self._output_to_video(output, fps, response_format)

    def _process_image(self, image: PIL.Image.Image, max_area: int) -> PIL.Image.Image:
        assert self._model is not None
        aspect_ratio = image.height / image.width
        mod_value = (
            self._model.vae_scale_factor_spatial
            * self._model.transformer.config.patch_size[1]
        )
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        return image.resize((width, height))

    @classmethod
    def _center_crop_resize(
        cls, image: PIL.Image.Image, height: int, width: int
    ) -> PIL.Image.Image:
        import torchvision.transforms.functional as TF

        # Calculate resize ratio to match first frame dimensions
        resize_ratio = max(width / image.width, height / image.height)

        # Resize the image
        width = round(image.width * resize_ratio)
        height = round(image.height * resize_ratio)
        size = [width, height]
        image = TF.center_crop(image, size)

        return image

    def _output_to_video(self, output: Any, fps: int, response_format: str):
        import gc

        # cv2 bug will cause the video cannot be normally displayed
        # thus we use the imageio one
        from diffusers.utils import export_to_video

        from ...device_utils import empty_cache

        # clean cache
        gc.collect()
        empty_cache()

        os.makedirs(XINFERENCE_VIDEO_DIR, exist_ok=True)
        urls = []
        for f in output.frames:
            path = os.path.join(XINFERENCE_VIDEO_DIR, uuid.uuid4().hex + ".mp4")
            export = (
                export_to_video
                if self.model_spec.model_family != "CogVideoX"
                else export_to_video_imageio
            )
            p = export(f, path, fps=fps)
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
