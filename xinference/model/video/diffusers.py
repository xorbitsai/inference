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

import base64
import importlib
import json
import logging
import operator
import os
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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

    @staticmethod
    def _enable_minimax_h3_group_offload(pipeline, kwargs: dict):
        import torch
        from diffusers.hooks import apply_group_offloading

        onload_device = torch.device(kwargs.pop("onload_device", "cuda"))
        offload_device = torch.device(kwargs.pop("offload_device", "cpu"))
        offload_kwargs = {
            "onload_device": onload_device,
            "offload_device": offload_device,
            "use_stream": kwargs.pop("use_stream", True),
        }
        num_blocks_per_group = kwargs.pop("num_blocks_per_group", 1)

        pipeline.transformer.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.transformer.enable_group_offload(
            offload_type="block_level",
            num_blocks_per_group=num_blocks_per_group,
            **offload_kwargs,
        )
        text_encoder = getattr(pipeline.text_encoder, "model", pipeline.text_encoder)
        apply_group_offloading(
            text_encoder,
            offload_type="leaf_level",
            **offload_kwargs,
        )
        pipeline.vae.to(onload_device)
        pipeline.audio_vae.to(onload_device)

    def _load_minimax_h3_quantized(self, kwargs: dict, torch_dtype, quantization: str):
        import torch
        from diffusers import (
            MiniMaxH3Transformer3DModel,
            ModularPipeline,
            TorchAoConfig,
        )
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
        from transformers import Qwen3VLForConditionalGeneration
        from transformers import TorchAoConfig as TransformersTorchAoConfig

        @contextmanager
        def skip_accelerate_dispatch(module, attribute):
            # The mixed CUDA/CPU map is needed only while each weight is quantized.
            # Installing runtime dispatch hooks afterwards briefly moves CPU weights
            # back to CUDA and exceeds 24GB before we can hand off to group offload.
            original = getattr(module, attribute)
            setattr(module, attribute, lambda model, *args, **kwargs: model)
            try:
                yield
            finally:
                setattr(module, attribute, original)

        def make_quant_type():
            if quantization == "int4":
                # The plain INT4 layout requires mslk, which is not published on
                # PyPI. Use PyTorch's native CUDA packing instead.
                return Int4WeightOnlyConfig(
                    group_size=128,
                    int4_packing_format="tile_packed_to_4d",
                    version=2,
                )
            return Int8WeightOnlyConfig(version=2)

        transformer_quantization_config = TorchAoConfig(
            make_quant_type(),
            modules_to_not_convert=[
                "proj_in",
                "audio_proj_in",
                "context_embedder",
                "time_embedder",
                "time_proj",
                "token_refiner",
                "norm_out",
                "proj_out",
                "audio_proj_out",
            ],
        )
        text_encoder_quantization_config = TransformersTorchAoConfig(
            make_quant_type(),
            modules_to_not_convert=[
                "model.visual",
                "model.language_model.embed_tokens",
                "model.language_model.norm",
                "lm_head",
            ],
        )

        pipeline = ModularPipeline.from_pretrained(self._model_path)
        onload_device = torch.device(kwargs.get("onload_device", "cuda"))
        offload_device = torch.device(kwargs.get("offload_device", "cpu"))
        transformer_load_kwargs = {}
        text_encoder_load_kwargs = {}
        if quantization == "int4":
            # Native INT4 packing runs on CUDA. Keep modules that stay in BF16 on
            # CPU, along with a few transformer blocks, so quantized weights never
            # accumulate beyond the capacity of a 24GB card while loading.
            transformer_device_map = {
                "": onload_device,
                "proj_in": "cpu",
                "audio_proj_in": "cpu",
                "context_embedder": "cpu",
                "time_embedder": "cpu",
                "time_proj": "cpu",
                "token_refiner": "cpu",
                "norm_out": "cpu",
                "proj_out": "cpu",
                "audio_proj_out": "cpu",
            }
            for block_index in range(46, 50):
                transformer_device_map[f"transformer_blocks.{block_index}"] = "cpu"

            transformer_load_kwargs["device_map"] = transformer_device_map
            text_encoder_load_kwargs["device_map"] = {
                "": onload_device,
                "model.visual": "cpu",
                "model.language_model.embed_tokens": "cpu",
                "model.language_model.norm": "cpu",
                "lm_head": "cpu",
            }

        if quantization == "int4":
            from diffusers.models import modeling_utils as diffusers_modeling_utils

            with skip_accelerate_dispatch(diffusers_modeling_utils, "dispatch_model"):
                transformer = MiniMaxH3Transformer3DModel.from_pretrained(
                    self._model_path,
                    subfolder="transformer",
                    dtype=torch_dtype,
                    quantization_config=transformer_quantization_config,
                    low_cpu_mem_usage=True,
                    **transformer_load_kwargs,
                )
        else:
            transformer = MiniMaxH3Transformer3DModel.from_pretrained(
                self._model_path,
                subfolder="transformer",
                dtype=torch_dtype,
                quantization_config=transformer_quantization_config,
                low_cpu_mem_usage=True,
            )
        if quantization == "int4":
            transformer.to(offload_device)

        if quantization == "int4":
            from transformers import modeling_utils as transformers_modeling_utils

            with skip_accelerate_dispatch(
                transformers_modeling_utils, "accelerate_dispatch"
            ):
                text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                    self._model_path,
                    subfolder="text_encoder",
                    dtype=torch_dtype,
                    quantization_config=text_encoder_quantization_config,
                    low_cpu_mem_usage=True,
                    **text_encoder_load_kwargs,
                )
        else:
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                self._model_path,
                subfolder="text_encoder",
                dtype=torch_dtype,
                quantization_config=text_encoder_quantization_config,
                low_cpu_mem_usage=True,
            )
        if quantization == "int4":
            text_encoder.to(offload_device)

        pipeline.update_components(transformer=transformer, text_encoder=text_encoder)
        pipeline.load_components(
            workflow="t2va",
            dtype=torch_dtype,
            pretrained_model_name_or_path=self._model_path,
        )

        if kwargs.pop("group_offload", False):
            if quantization == "int4":
                kwargs.setdefault("use_stream", False)
            self._enable_minimax_h3_group_offload(pipeline, kwargs)
        else:
            device = torch.device(kwargs.pop("onload_device", "cuda"))
            for component_name in ("transformer", "text_encoder", "vae", "audio_vae"):
                getattr(pipeline, component_name).to(device)
        self._model = pipeline

    def _load_minimax_h3(self, kwargs: dict, torch_dtype):
        quantization = kwargs.pop("quantization", None)
        if quantization in ("int4", "int8", "torchao"):
            self._load_minimax_h3_quantized(
                kwargs,
                torch_dtype,
                "int8" if quantization == "torchao" else quantization,
            )
            return

        if quantization not in (None, "none", "bf16"):
            raise ValueError(
                f"Unsupported MiniMax-H3 quantization: {quantization}. "
                "Supported values are int4, int8, none, and bf16."
            )

        from diffusers import ComponentsManager, ModularPipeline

        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(
            device=kwargs.pop("onload_device", "cuda"),
            memory_reserve_margin=kwargs.pop("memory_reserve_margin", "12GB"),
        )
        pipeline = ModularPipeline.from_pretrained(
            self._model_path, components_manager=manager
        )
        pipeline.load_components(
            workflow="t2va",
            dtype=torch_dtype,
            pretrained_model_name_or_path=self._model_path,
        )
        self._model = pipeline

    @staticmethod
    def _is_minimax_h3(model_spec) -> bool:
        return model_spec.model_family == "MiniMax-H3"

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

        if self._is_minimax_h3(self._model_spec):
            self._load_minimax_h3(kwargs, torch_dtype)
            return

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
        elif self.model_spec.model_family == "WanAnimate2":
            from diffusers import WanAnimate2Pipeline

            pipeline = self._model = WanAnimate2Pipeline.from_pretrained(
                self._model_path, **kwargs
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
        if self._is_minimax_h3(self._model_spec):
            return self._minimax_h3_generate(
                prompt=prompt,
                n=n,
                num_inference_steps=num_inference_steps,
                response_format=response_format,
                **kwargs,
            )
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
        if self._is_minimax_h3(self._model_spec):
            return self._minimax_h3_generate(
                prompt=prompt,
                image=image,
                n=n,
                num_inference_steps=num_inference_steps,
                response_format=response_format,
                **kwargs,
            )
        generate_kwargs = self._model_spec.default_generate_config.copy()
        generate_kwargs.update(kwargs)
        if num_inference_steps:
            generate_kwargs["num_inference_steps"] = num_inference_steps

        if self.model_spec.model_family == "WanAnimate2":
            if n != 1:
                raise ValueError(
                    "Wan-Animate-2 only supports generating one video per request"
                )

            video = generate_kwargs.pop("video", None)
            if video is None:
                raise ValueError("`video` is required for Wan-Animate-2")

            fps = generate_kwargs.get("fps", 24)
            temp_video_path = None
            output: Any
            try:
                if isinstance(video, bytes):
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                        temp_video_path = f.name
                        f.write(video)
                    video = temp_video_path
                elif isinstance(video, str):
                    if not os.path.isfile(video):
                        raise FileNotFoundError(
                            f"Video path does not exist or is not a file: {video}"
                        )
                else:
                    raise TypeError("`video` must be video bytes or a local path")

                self._process_progressor(generate_kwargs)
                output = self._model(
                    image=image,
                    driving_video=video,
                    prompt=prompt,
                    **generate_kwargs,
                )
            finally:
                if temp_video_path and os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

            return self._output_to_video(output, fps, response_format)

        generate_kwargs["num_videos_per_prompt"] = n
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
        if self._is_minimax_h3(self._model_spec):
            return self._minimax_h3_generate(
                prompt=prompt,
                image=first_frame,
                last_image=last_frame,
                n=n,
                num_inference_steps=num_inference_steps,
                response_format=response_format,
                **kwargs,
            )
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

    def _minimax_h3_generate(
        self,
        prompt: str,
        n: int,
        num_inference_steps: Optional[int],
        response_format: str,
        image: Optional[PIL.Image.Image] = None,
        last_image: Optional[PIL.Image.Image] = None,
        **kwargs,
    ) -> VideoList:
        if n < 1:
            raise ValueError("n must be at least 1")

        progressor = kwargs.pop("progressor", None)
        kwargs.pop("request_id", None)
        # H3 is guidance-distilled and uses a fixed 24 fps output format.
        kwargs.pop("negative_prompt", None)
        kwargs.pop("guidance_scale", None)
        kwargs.pop("num_videos_per_prompt", None)
        kwargs.pop("fps", None)

        generate_kwargs = (self._model_spec.default_generate_config or {}).copy()
        generate_kwargs.update(kwargs)
        if num_inference_steps is not None:
            generate_kwargs["num_inference_steps"] = num_inference_steps
        else:
            generate_kwargs.setdefault("num_inference_steps", 50)

        pipeline = self._model
        assert callable(pipeline)
        videos = []
        for video_index in range(n):
            call_kwargs = generate_kwargs.copy()
            if image is not None:
                call_kwargs["image"] = image
            if last_image is not None:
                call_kwargs["last_image"] = last_image
            with self._track_minimax_h3_progress(pipeline, progressor, video_index, n):
                output = pipeline(
                    prompt=prompt,
                    output=["videos", "audio", "sampling_rate"],
                    **call_kwargs,
                )
            videos.extend(self._encode_minimax_h3_output(output))

        if progressor and progressor.request_id:
            progressor.set_progress(1.0)
        return self._video_urls_to_response(videos, response_format)

    @staticmethod
    @contextmanager
    def _track_minimax_h3_progress(pipeline, progressor, video_index: int, n: int):
        if not progressor or not progressor.request_id:
            yield
            return

        class ProgressBarWrapper:
            def __init__(self, progress_bar):
                self._progress_bar = progress_bar

            def __enter__(self):
                self._progress_bar.__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return self._progress_bar.__exit__(exc_type, exc_val, exc_tb)

            def __getattr__(self, name):
                return getattr(self._progress_bar, name)

            def update(self, count=1):
                result = self._progress_bar.update(count)
                if self._progress_bar.total:
                    step_progress = min(
                        self._progress_bar.n / self._progress_bar.total, 1.0
                    )
                    progressor.set_progress((video_index + step_progress) / n)
                return result

        def iter_blocks(block):
            yield block
            sub_blocks = getattr(block, "sub_blocks", None)
            if sub_blocks:
                for sub_block in sub_blocks.values():
                    yield from iter_blocks(sub_block)

        missing = object()
        patched_blocks = []
        blocks = getattr(pipeline, "_blocks", None)
        if blocks is None:
            yield
            return

        for block in iter_blocks(blocks):
            if getattr(block, "model_name", None) != "minimax-h3" or not hasattr(
                block, "loop_step"
            ):
                continue
            previous = block.__dict__.get("progress_bar", missing)
            original = block.progress_bar

            def tracked_progress_bar(*args, _original=original, **kwargs):
                return ProgressBarWrapper(_original(*args, **kwargs))

            block.progress_bar = tracked_progress_bar
            patched_blocks.append((block, previous))

        try:
            yield
        finally:
            for block, previous in patched_blocks:
                if previous is missing:
                    del block.progress_bar
                else:
                    block.progress_bar = previous

    @staticmethod
    def _encode_minimax_h3_output(output: dict) -> List[str]:
        from diffusers.utils import encode_video

        os.makedirs(XINFERENCE_VIDEO_DIR, exist_ok=True)
        video_outputs = output["videos"]
        if hasattr(video_outputs, "ndim"):
            video_outputs = (
                list(video_outputs) if video_outputs.ndim == 5 else [video_outputs]
            )
        elif video_outputs and (
            isinstance(video_outputs[0], (PIL.Image.Image, np.ndarray))
            or getattr(video_outputs[0], "ndim", 0) == 3
        ):
            video_outputs = [video_outputs]

        audio_outputs = output.get("audio")
        if audio_outputs is None:
            audio_outputs = [None] * len(video_outputs)
        elif hasattr(audio_outputs, "ndim"):
            audio_outputs = (
                list(audio_outputs) if audio_outputs.ndim == 3 else [audio_outputs]
            )
        elif not isinstance(audio_outputs, list):
            audio_outputs = [audio_outputs]

        sampling_rate = output.get("sampling_rate")
        urls = []
        for index, video in enumerate(video_outputs):
            path = os.path.join(XINFERENCE_VIDEO_DIR, uuid.uuid4().hex + ".mp4")
            audio = audio_outputs[index] if index < len(audio_outputs) else None
            encode_video(
                video,
                fps=24,
                output_path=path,
                audio=audio,
                audio_sample_rate=sampling_rate,
            )
            urls.append(path)
        return urls

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
        return self._video_urls_to_response(urls, response_format)

    @staticmethod
    def _video_urls_to_response(urls: List[str], response_format: str):
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
