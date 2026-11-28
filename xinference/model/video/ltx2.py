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
import gc
import logging
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import PIL.Image

from ...constants import XINFERENCE_VIDEO_DIR
from ...core.exceptions import InvalidModelRequestError
from ...device_utils import empty_cache, get_available_device
from ...types import Video, VideoList
from ..lightning import resolve_request_lightning
from ..utils import resolve_media_seed
from .diffusers import DiffusersVideoModel
from .engine_family import VideoEngineModel
from .ltx25_checkpoint import LTX25_MODEL_SOURCES, load_phai_m1_components

if TYPE_CHECKING:
    from ...core.progress_tracker import Progressor
    from .core import VideoModelFamilyV2


logger = logging.getLogger(__name__)


class LTX2VideoModel(DiffusersVideoModel, VideoEngineModel):
    """LTX-2.5 and Phai-M1 audio-video models backed by Diffusers."""

    _model: Any
    replica_autoscaling = "serialized"
    engine_model_format = "diffusers"
    engine_quantization = "none"
    required_libs = ("diffusers",)

    _PHAI_M1_MODEL_SOURCES = {
        "huggingface": ("PhaiModelLab/Phai-M1", "main"),
        "modelscope": ("PhaiModelLab/Phai-M1", "master"),
    }
    _DISTILLED = "distilled"
    _DEV = "dev"
    _SUPPORTED_VARIANTS = {_DISTILLED, _DEV}
    _LTX25_NEGATIVE_PROMPT_PREFIX = (
        "has_subtitles",
        "has_blurbox",
        "transition from black",
        "transition to black",
        "speech_ending_short",
    )

    @classmethod
    def match(cls, model_family: "VideoModelFamilyV2") -> bool:
        return cls.is_model_family_supported(model_family) and (
            not model_family.engine or model_family.engine.lower() == "diffusers"
        )

    @classmethod
    def is_model_family_supported(cls, model_family: "VideoModelFamilyV2") -> bool:
        return model_family.model_family in {"LTX2", "Phai-M1"}

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "VideoModelFamilyV2",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_uid, model_path, model_spec, **kwargs)
        self._is_phai_m1 = model_spec.model_family == "Phai-M1"
        self._variant = self._variant_from_spec(model_spec)
        self._device = "cpu"
        self._torch_dtype: Any = None
        self._latent_upsample_pipeline: Any = None

    @staticmethod
    def _option_enabled(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "false",
            "none",
            "off",
        }:
            return False
        return bool(value)

    @classmethod
    def validate_model_spec(cls, model_spec: "VideoModelFamilyV2") -> None:
        """Reject non-official LTX specs before a model download can start."""
        if model_spec.model_family == "Phai-M1":
            model_label = "Phai-M1"
            sources = cls._PHAI_M1_MODEL_SOURCES
        else:
            model_label = "LTX-2.5"
            sources = LTX25_MODEL_SOURCES
        official_source = sources.get(model_spec.model_hub)
        if official_source is None:
            raise ValueError(
                f"{model_label} only supports the official Hugging Face and ModelScope "
                "sources"
            )
        official_model_id, official_model_revision = official_source
        if model_spec.model_id != official_model_id:
            raise ValueError(
                f"{model_label} must use the official {official_model_id} model repository"
            )
        if model_spec.model_revision != official_model_revision:
            raise ValueError(
                f"{model_label} must use the tested official model revision "
                f"{official_model_revision}"
            )
        cls._variant_from_spec(model_spec)

    @classmethod
    def _variant_from_spec(cls, model_spec: "VideoModelFamilyV2") -> str:
        variant = (model_spec.default_model_config or {}).get(
            "model_variant", cls._DISTILLED
        )
        if not isinstance(variant, str) or variant not in cls._SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported LTX-2.5 variant {variant!r}; expected one of "
                f"{sorted(cls._SUPPORTED_VARIANTS)}"
            )
        return variant

    def _validate_official_model(self) -> None:
        self.validate_model_spec(self._model_spec)

    @staticmethod
    def validate_phai_m1_device_map(device_map: Any) -> None:
        if device_map:
            raise ValueError(
                "Phai-M1 does not support device_map with its preloaded transformer "
                "and connectors; use CPU offload or a single accelerator"
            )

    def _reject_quantization(self, kwargs: Dict[str, Any]) -> None:
        quantization_options = (
            "quantization",
            "quantization_config",
            "gguf_quantization",
            "load_in_4bit",
            "load_in_8bit",
            "layerwise_cast",
        )
        enabled = [
            key
            for key in quantization_options
            if key in kwargs and self._option_enabled(kwargs.pop(key))
        ]
        diffusers_variant = kwargs.pop("variant", None)
        if self._option_enabled(diffusers_variant):
            enabled.append("variant")
        if self._gguf_model_path:
            enabled.append("gguf_model_path")
        if enabled:
            raise ValueError(
                "LTX-2.5 in Xinference only supports the official BF16 weights; "
                f"quantized or alternate weight options are not allowed: {enabled}"
            )

    def load(self) -> None:
        import torch
        from diffusers import (
            FlowMatchEulerDiscreteScheduler,
            LTX2ConditionPipeline,
            LTX2VideoTransformer3DModel,
        )
        from packaging.version import Version

        self._validate_official_model()

        torch_version = getattr(torch, "__version__", None)
        if torch_version is not None:
            parsed_torch_version = Version(str(torch_version).split("+")[0])
            if parsed_torch_version < Version("2.7"):
                raise RuntimeError(
                    "Official LTX-2.5 requires PyTorch 2.7 or newer; "
                    f"found {torch_version}"
                )

        kwargs = (self._model_spec.default_model_config or {}).copy()
        kwargs.update(self._kwargs)
        self._reject_quantization(kwargs)

        requested_variant = kwargs.pop("model_variant", self._variant)
        if requested_variant != self._variant:
            raise ValueError(
                f"Model {self._model_spec.model_name} is fixed to the "
                f"{self._variant!r} variant, not {requested_variant!r}"
            )

        dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", "bfloat16"))
        if isinstance(dtype, str):
            try:
                dtype = getattr(torch, dtype)
            except AttributeError as exc:
                raise ValueError(
                    f"Unsupported torch dtype for LTX-2.5: {dtype}"
                ) from exc
        if dtype is not torch.bfloat16:
            raise ValueError("Official LTX-2.5 weights must be loaded with bfloat16")
        self._torch_dtype = dtype

        cpu_offload = self._option_enabled(kwargs.pop("cpu_offload", False))
        sequential_cpu_offload = self._option_enabled(
            kwargs.pop("sequential_cpu_offload", False)
        )
        vae_tiling = self._option_enabled(kwargs.pop("vae_tiling", False))
        compile_graph = kwargs.pop("compile_graph", False)
        if not self._option_enabled(compile_graph):
            compile_graph = False
        group_offload = self._option_enabled(kwargs.pop("group_offload", False))
        kwargs.pop("use_stream", None)
        kwargs.pop("enable_virtual_env", None)
        kwargs.pop("n_gpu", None)

        use_safetensors = kwargs.pop("use_safetensors", True)
        if not self._option_enabled(use_safetensors):
            raise ValueError("Official LTX-2.5 must be loaded from safetensors")
        local_files_only = kwargs.pop("local_files_only", True)
        if not self._option_enabled(local_files_only):
            raise ValueError(
                "LTX-2.5 is downloaded by Xinference before loading; "
                "local_files_only must remain enabled"
            )

        if group_offload:
            raise ValueError(
                "LTX-2.5 group offload is not supported because its audio-video "
                "component chain requires model-aware placement; use cpu_offload or "
                "sequential_cpu_offload"
            )
        if sequential_cpu_offload and cpu_offload:
            logger.info(
                "LTX-2.5 sequential CPU offload was requested; disabling model "
                "CPU offload"
            )
            cpu_offload = False

        requested_device = kwargs.pop("device", None)
        device = requested_device
        if not device or str(device).lower() == "auto":
            device = get_available_device()
        self._device = str(device)

        device_map = kwargs.get("device_map")
        if self._is_phai_m1:
            self.validate_phai_m1_device_map(kwargs.pop("device_map", None))
        if device_map:
            if self._variant == self._DEV:
                raise ValueError(
                    "LTX-2.5-Dev does not support pipeline device_map because "
                    "transformer_full is loaded as an explicit component; use "
                    "model CPU offload instead"
                )
            if (
                self._option_enabled(requested_device)
                and str(requested_device).lower() != "auto"
            ):
                raise ValueError(
                    "LTX-2.5 device and device_map are mutually exclusive; "
                    "let device_map control placement"
                )
            if cpu_offload or sequential_cpu_offload:
                logger.info(
                    "LTX-2.5 device_map is set; disabling explicit CPU offload hooks"
                )
            cpu_offload = sequential_cpu_offload = False
            self._device = "device_map"

        load_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "local_files_only": True,
            "use_safetensors": True,
        }
        load_kwargs.update(kwargs)
        # The ModelScope snapshot declares this optional, unused component.
        load_kwargs["prompt_enhancer"] = None

        pipeline_path = self._model_path
        if self._is_phai_m1:
            transformer_filename = load_kwargs.pop("transformer_filename")
            base_model_path = load_kwargs.pop("base_model_path")
            pipeline_path, transformer, connectors = load_phai_m1_components(
                self._model_path,
                transformer_filename,
                base_model_path,
                dtype,
            )
            load_kwargs.update(
                transformer=transformer, connectors=connectors, processor=None
            )
        elif self._variant == self._DEV:
            transformer_kwargs = {
                key: value
                for key, value in load_kwargs.items()
                if key in {"dtype", "local_files_only", "use_safetensors"}
            }
            load_kwargs["transformer"] = LTX2VideoTransformer3DModel.from_pretrained(
                self._model_path,
                subfolder="transformer_full",
                **transformer_kwargs,
            )
        pipeline = LTX2ConditionPipeline.from_pretrained(pipeline_path, **load_kwargs)
        if self._variant == self._DEV:
            pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                pipeline.scheduler.config,
                use_dynamic_shifting=True,
                shift_terminal=0.1,
            )

        if compile_graph:
            compile_mode = (
                "max-autotune" if compile_graph is True else str(compile_graph)
            )
            pipeline.transformer = torch.compile(
                pipeline.transformer, mode=compile_mode, fullgraph=True
            )

        if vae_tiling:
            pipeline.vae.enable_tiling()

        if self._device == "cpu":
            pipeline.to("cpu")
        elif sequential_cpu_offload:
            pipeline.enable_sequential_cpu_offload(device=self._device)
        elif cpu_offload:
            pipeline.enable_model_cpu_offload(device=self._device)
        elif not device_map:
            pipeline.to(self._device)

        self._model = pipeline

    @staticmethod
    def _validate_dimensions_and_frames(
        generate_kwargs: Dict[str, Any], two_stage: bool
    ) -> None:
        width = generate_kwargs.get("width")
        height = generate_kwargs.get("height")
        num_frames = generate_kwargs.get("num_frames")
        frame_rate = generate_kwargs.get("frame_rate")

        for name, value in (("width", width), ("height", height)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"LTX-2.5 {name} must be a positive integer")
            if value % 32 != 0:
                raise ValueError(f"LTX-2.5 {name} must be divisible by 32")
            if two_stage and value % 64 != 0:
                raise ValueError(
                    f"Two-stage LTX-2.5 {name} must be divisible by 64 because "
                    "stage 1 runs at half the final resolution"
                )

        if num_frames is not None:
            if (
                not isinstance(num_frames, int)
                or isinstance(num_frames, bool)
                or num_frames <= 0
            ):
                raise ValueError(
                    "LTX-2.5 num_frames must be a positive integer or None"
                )
            if num_frames % 8 != 1:
                raise ValueError("LTX-2.5 num_frames must satisfy num_frames % 8 == 1")

        if (
            not isinstance(frame_rate, (int, float))
            or isinstance(frame_rate, bool)
            or not math.isfinite(frame_rate)
            or frame_rate < 1
        ):
            raise ValueError("LTX-2.5 fps/frame_rate must be at least 1")
        if not float(frame_rate).is_integer():
            raise ValueError(
                "LTX-2.5 fps/frame_rate must be an integer so generated audio "
                "and encoded video stay synchronized"
            )

        min_seconds = generate_kwargs.get("min_seconds", 1.0)
        max_seconds = generate_kwargs.get("max_seconds", 20.0)
        for name, value in (
            ("min_seconds", min_seconds),
            ("max_seconds", max_seconds),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"LTX-2.5 {name} must be positive")
        if num_frames is None and min_seconds >= max_seconds:
            raise ValueError(
                "LTX-2.5 max_seconds must be greater than min_seconds when "
                "automatic duration is used"
            )

    @staticmethod
    def _validate_prompt_batch(
        prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> None:
        if not isinstance(prompt, (str, list)):
            raise TypeError("LTX-2.5 prompt must be a string or a list of strings")
        if isinstance(prompt, list):
            if not prompt or not all(isinstance(value, str) for value in prompt):
                raise ValueError("LTX-2.5 prompt list must contain strings")
        if not isinstance(negative_prompt, (str, list)):
            raise TypeError(
                "LTX-2.5 negative_prompt must be a string or a list of strings"
            )
        if isinstance(negative_prompt, list):
            if not all(isinstance(value, str) for value in negative_prompt):
                raise ValueError("LTX-2.5 negative_prompt list must contain strings")
            prompt_count = len(prompt) if isinstance(prompt, list) else 1
            if len(negative_prompt) != prompt_count:
                raise ValueError(
                    "LTX-2.5 negative_prompt list must have the same length as prompt"
                )

    @staticmethod
    def _validate_condition_strength(name: str, value: float) -> None:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"LTX-2.5 {name} must be between 0 and 1")

    @staticmethod
    def _validate_image_crf(value: Optional[int]) -> None:
        if value is None:
            return
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 51
        ):
            raise ValueError("LTX-2.5 image_crf must be an integer between 0 and 51")

    @staticmethod
    def _set_progress_callback(
        generate_kwargs: Dict[str, Any],
        progressor: Optional["Progressor"],
        start: float = 0.0,
        span: float = 1.0,
    ) -> None:
        if not progressor or not progressor.request_id:
            return

        def report_status_callback(
            pipe: Any,
            step: int,
            timestep: int,
            callback_kwargs: Dict[str, Any],
        ) -> Dict[str, Any]:
            del timestep
            total = max(int(pipe.num_timesteps), 1)
            progressor.set_progress(start + span * (step + 1) / total)
            return callback_kwargs

        generate_kwargs["callback_on_step_end"] = report_status_callback

    def _prepare_generate_kwargs(
        self,
        n: int,
        num_inference_steps: Optional[int],
        kwargs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float, bool, Optional["Progressor"]]:
        import torch
        from diffusers.pipelines.ltx2.utils import (
            DEFAULT_NEGATIVE_PROMPT,
            DISTILLED_SIGMA_VALUES,
        )

        if self._is_phai_m1:
            if n < 1:
                raise ValueError("n must be at least 1")
        elif not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 10:
            raise ValueError("n must be an integer between 1 and 10")

        generate_kwargs = (self._model_spec.default_generate_config or {}).copy()
        generate_kwargs.update(kwargs)

        progressor = generate_kwargs.pop("progressor", None)
        two_stage = self._option_enabled(generate_kwargs.pop("two_stage", False))
        if self._is_phai_m1:
            if two_stage:
                raise ValueError("Phai-M1 supports single-stage generation")
            resolve_request_lightning(generate_kwargs.pop("use_lightning", None), False)
            frames = generate_kwargs.get("num_frames")
            if isinstance(frames, int) and frames > 0:
                # Keep the requested audio duration aligned with decodable frames.
                generate_kwargs["num_frames"] = 1 + 8 * ((frames + 3) // 8)

        fps = generate_kwargs.pop("fps", None)
        frame_rate = generate_kwargs.pop("frame_rate", fps if fps is not None else 24.0)
        generate_kwargs["frame_rate"] = frame_rate
        generate_kwargs["num_videos_per_prompt"] = n

        if num_inference_steps is not None:
            generate_kwargs["num_inference_steps"] = num_inference_steps

        seed = generate_kwargs.pop("seed", None)
        if self._is_phai_m1:
            seed = resolve_media_seed(seed)
        if seed is not None:
            if "generator" in generate_kwargs:
                raise ValueError("Pass either seed or generator, not both")
            if isinstance(seed, bool):
                raise ValueError("LTX-2.5 seed must be an integer")
            try:
                normalized_seed = int(seed)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("LTX-2.5 seed must be an integer") from exc
            if isinstance(seed, float) and not seed.is_integer():
                raise ValueError("LTX-2.5 seed must be an integer")
            execution_device = getattr(self._model, "_execution_device", self._device)
            generate_kwargs["generator"] = torch.Generator(
                device=execution_device
            ).manual_seed(normalized_seed)

        if generate_kwargs.get("enable_prompt_enhancement"):
            raise ValueError(
                "The official LTX-2.5 Diffusers repository does not ship the "
                "separate prompt-enhancer model; prompt enhancement is disabled"
            )

        negative_prompt = generate_kwargs.get("negative_prompt")
        if negative_prompt is None:
            missing_prefixes = (
                []
                if self._is_phai_m1
                else [
                    value
                    for value in self._LTX25_NEGATIVE_PROMPT_PREFIX
                    if value not in DEFAULT_NEGATIVE_PROMPT
                ]
            )
            generate_kwargs["negative_prompt"] = ", ".join(
                [*missing_prefixes, DEFAULT_NEGATIVE_PROMPT]
            )

        generate_kwargs["output_type"] = "np"
        generate_kwargs["return_dict"] = True

        if self._variant == self._DISTILLED:
            steps = generate_kwargs.pop("num_inference_steps", None)
            timesteps = generate_kwargs.pop("timesteps", None)
            supplied_sigmas = generate_kwargs.pop("sigmas", None)
            if self._is_phai_m1 and steps is not None and steps < 1:
                raise ValueError("num_inference_steps must be positive")
            if not self._is_phai_m1 and steps is not None:
                raise ValueError(
                    "LTX-2.5 Distilled uses the official fixed sigma schedule; "
                    "num_inference_steps must not be supplied"
                )
            if not self._is_phai_m1 and timesteps is not None:
                raise ValueError(
                    "LTX-2.5 Distilled uses sigmas and does not accept timesteps"
                )
            if not self._is_phai_m1 and supplied_sigmas is not None:
                try:
                    normalized_sigmas = [float(value) for value in supplied_sigmas]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "LTX-2.5 Distilled sigmas must use the official schedule"
                    ) from exc
                if normalized_sigmas != list(DISTILLED_SIGMA_VALUES):
                    raise ValueError(
                        "LTX-2.5 Distilled sigmas must use the official fixed schedule"
                    )
            generate_kwargs["sigmas"] = list(DISTILLED_SIGMA_VALUES)
            distilled_guidance = {
                "guidance_scale": 1.0,
                "audio_guidance_scale": 1.0,
                "stg_scale": 0.0,
                "audio_stg_scale": 0.0,
                "modality_scale": 1.0,
                "audio_modality_scale": 1.0,
                "guidance_rescale": 0.0,
                "audio_guidance_rescale": 0.0,
            }
            for name, expected in distilled_guidance.items():
                actual = (
                    expected
                    if self._is_phai_m1
                    else generate_kwargs.get(name, expected)
                )
                if float(actual) != expected:
                    raise ValueError(
                        f"LTX-2.5 Distilled requires {name}={expected}, got {actual}"
                    )
                generate_kwargs[name] = expected
        elif two_stage:
            raise ValueError(
                "Two-stage generation is only supported by LTX-2.5 Distilled"
            )
        else:
            num_steps = generate_kwargs.get("num_inference_steps")
            if (
                not isinstance(num_steps, int)
                or isinstance(num_steps, bool)
                or num_steps <= 0
            ):
                raise ValueError(
                    "LTX-2.5-Dev num_inference_steps must be a positive integer"
                )

            scale_names = (
                "guidance_scale",
                "audio_guidance_scale",
                "stg_scale",
                "audio_stg_scale",
                "modality_scale",
                "audio_modality_scale",
                "guidance_rescale",
                "audio_guidance_rescale",
            )
            for name in scale_names:
                value = generate_kwargs.get(name)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"LTX-2.5-Dev {name} must be non-negative")
            for name in ("guidance_rescale", "audio_guidance_rescale"):
                if generate_kwargs[name] > 1:
                    raise ValueError(f"LTX-2.5-Dev {name} must not exceed 1")

            # Pinned Diffusers uses ``audio_value or video_value`` for these
            # pairs. Reject the one ambiguous case instead of silently changing
            # an explicitly requested audio zero into a non-zero video value.
            fallback_pairs = (
                ("audio_guidance_scale", "guidance_scale"),
                ("audio_stg_scale", "stg_scale"),
                ("audio_modality_scale", "modality_scale"),
                ("audio_guidance_rescale", "guidance_rescale"),
            )
            for audio_name, video_name in fallback_pairs:
                if (
                    generate_kwargs[audio_name] == 0
                    and generate_kwargs[video_name] != 0
                ):
                    raise ValueError(
                        f"Pinned Diffusers cannot preserve {audio_name}=0 while "
                        f"{video_name} is non-zero; use matching zero values"
                    )

        self._validate_dimensions_and_frames(generate_kwargs, two_stage)
        return generate_kwargs, float(frame_rate), two_stage, progressor

    def _get_latent_upsample_pipeline(self) -> Any:
        if self._latent_upsample_pipeline is None:
            from diffusers import LTX2LatentUpsamplePipeline
            from diffusers.pipelines.ltx2.latent_upsampler import (
                LTX2LatentUpsamplerModel,
            )

            latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                self._model_path,
                subfolder="latent_upsampler",
                dtype=self._torch_dtype,
                local_files_only=True,
                use_safetensors=True,
            )
            self._latent_upsample_pipeline = LTX2LatentUpsamplePipeline(
                vae=self._model.vae,
                latent_upsampler=latent_upsampler,
            )
        return self._latent_upsample_pipeline

    def _run_two_stage(
        self,
        conditions: Optional[List[Any]],
        generate_kwargs: Dict[str, Any],
        progressor: Optional["Progressor"],
    ) -> Any:
        from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES

        num_frames = generate_kwargs.get("num_frames")
        if num_frames is None:
            raise ValueError(
                "Two-stage LTX-2.5 generation requires an explicit num_frames"
            )
        if self._device == "device_map":
            raise ValueError(
                "Two-stage LTX-2.5 generation does not support device_map; "
                "use model CPU offload or a single accelerator"
            )

        # The second stage decodes at the final 2x resolution. Official
        # Diffusers recipes enable VAE tiling before this decode to avoid a
        # prohibitively large peak allocation.
        self._model.vae.enable_tiling()

        target_height = generate_kwargs["height"]
        target_width = generate_kwargs["width"]

        stage_1_kwargs = generate_kwargs.copy()
        stage_1_kwargs["height"] = target_height // 2
        stage_1_kwargs["width"] = target_width // 2
        stage_1_kwargs["output_type"] = "latent"
        stage_1_kwargs["return_dict"] = False
        self._set_progress_callback(stage_1_kwargs, progressor, 0.0, 0.7)
        video_latents, audio_latents = self._model(
            conditions=conditions,
            **stage_1_kwargs,
        )

        upsample_pipeline = self._get_latent_upsample_pipeline()
        upsampler = upsample_pipeline.latent_upsampler
        if self._device != "cpu":
            upsampler.to(self._device)
        try:
            upsampled_latents = upsample_pipeline(
                latents=video_latents,
                height=target_height // 2,
                width=target_width // 2,
                num_frames=num_frames,
                output_type="latent",
                return_dict=False,
            )[0]
        finally:
            if self._device != "cpu":
                upsampler.to("cpu")
                empty_cache()

        stage_2_kwargs = generate_kwargs.copy()
        stage_2_kwargs.update(
            {
                "sigmas": list(STAGE_2_DISTILLED_SIGMA_VALUES),
                "latents": upsampled_latents,
                "audio_latents": audio_latents,
                "noise_scale": STAGE_2_DISTILLED_SIGMA_VALUES[0],
                "output_type": "np",
                "return_dict": True,
            }
        )
        self._set_progress_callback(stage_2_kwargs, progressor, 0.7, 0.3)
        return self._model(conditions=conditions, **stage_2_kwargs)

    def _generate(
        self,
        prompt: Union[str, List[str]],
        n: int,
        num_inference_steps: Optional[int],
        response_format: str,
        conditions: Optional[List[Any]],
        **kwargs: Any,
    ) -> VideoList:
        if self._model is None:
            raise RuntimeError("LTX-2.5 model is not loaded")

        try:
            if response_format not in {"url", "b64_json"}:
                raise ValueError(f"Unsupported response format: {response_format}")
            generate_kwargs, frame_rate, two_stage, progressor = (
                self._prepare_generate_kwargs(n, num_inference_steps, kwargs)
            )
        except InvalidModelRequestError:
            raise
        except ValueError as exc:
            if self._is_phai_m1:
                raise InvalidModelRequestError(str(exc)) from exc
            raise
        negative_prompt = generate_kwargs["negative_prompt"]
        self._validate_prompt_batch(prompt, negative_prompt)
        if generate_kwargs.get("num_frames") is None and isinstance(prompt, list):
            raise ValueError(
                "LTX-2.5 automatic duration only supports one prompt at a time; "
                "set num_frames for prompt batches"
            )
        generate_kwargs["prompt"] = prompt

        logger.debug("LTX-2.5 generation args: %s", generate_kwargs)
        if two_stage:
            output = self._run_two_stage(conditions, generate_kwargs, progressor)
        else:
            self._set_progress_callback(generate_kwargs, progressor)
            output = self._model(conditions=conditions, **generate_kwargs)

        return self._output_to_audio_video(output, frame_rate, response_format)

    def text_to_video(
        self,
        prompt: Union[str, List[str]],
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> VideoList:
        return self._generate(
            prompt,
            n,
            num_inference_steps,
            response_format,
            conditions=None,
            **kwargs,
        )

    def image_to_video(
        self,
        image: PIL.Image.Image,
        prompt: Optional[Union[str, List[str]]],
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> VideoList:
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

        if not isinstance(image, PIL.Image.Image):
            raise TypeError("LTX-2.5 image must be a PIL image")
        strength = float(kwargs.pop("condition_strength", 1.0))
        self._validate_condition_strength("condition_strength", strength)
        image_crf = kwargs.pop("image_crf", None)
        self._validate_image_crf(image_crf)
        condition = LTX2VideoCondition(
            frames=image.convert("RGB"),
            index=0,
            strength=strength,
            crf=image_crf,
        )
        return self._generate(
            prompt if prompt is not None else "",
            n,
            num_inference_steps,
            response_format,
            conditions=[condition],
            **kwargs,
        )

    def firstlastframe_to_video(
        self,
        first_frame: PIL.Image.Image,
        last_frame: PIL.Image.Image,
        prompt: Optional[Union[str, List[str]]],
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> VideoList:
        if self._is_phai_m1:
            raise InvalidModelRequestError(
                "Phai-M1 does not support first/last-frame generation"
            )
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

        if not isinstance(first_frame, PIL.Image.Image) or not isinstance(
            last_frame, PIL.Image.Image
        ):
            raise TypeError("LTX-2.5 first and last frames must be PIL images")
        image_crf = kwargs.pop("image_crf", None)
        self._validate_image_crf(image_crf)
        first_strength = float(kwargs.pop("first_frame_strength", 1.0))
        last_strength = float(kwargs.pop("last_frame_strength", 1.0))
        self._validate_condition_strength("first_frame_strength", first_strength)
        self._validate_condition_strength("last_frame_strength", last_strength)
        conditions = [
            LTX2VideoCondition(
                frames=first_frame.convert("RGB"),
                index=0,
                strength=first_strength,
                crf=image_crf,
            ),
            LTX2VideoCondition(
                frames=last_frame.convert("RGB"),
                index=-1,
                strength=last_strength,
                crf=image_crf,
            ),
        ]
        return self._generate(
            prompt if prompt is not None else "",
            n,
            num_inference_steps,
            response_format,
            conditions=conditions,
            **kwargs,
        )

    @staticmethod
    def _paths_to_video_list(paths: List[str], response_format: str) -> VideoList:
        if response_format == "url":
            return VideoList(
                created=int(time.time()),
                data=[Video(url=path, b64_json=None) for path in paths],
            )

        def encode_and_remove(path: str) -> str:
            try:
                with open(path, "rb") as video_file:
                    return base64.b64encode(video_file.read()).decode()
            finally:
                if os.path.exists(path):
                    os.remove(path)

        with ThreadPoolExecutor() as executor:
            encoded = list(executor.map(encode_and_remove, paths))
        return VideoList(
            created=int(time.time()),
            data=[Video(url=None, b64_json=value) for value in encoded],
        )

    def _output_to_audio_video(
        self, output: Any, frame_rate: float, response_format: str
    ) -> VideoList:
        from diffusers.utils import encode_video

        frames = output.frames
        audio = output.audio
        if frames is None or audio is None:
            raise RuntimeError("LTX-2.5 must return both video frames and audio")
        if len(frames) != len(audio):
            raise RuntimeError(
                "LTX-2.5 returned different video and audio batch sizes: "
                f"{len(frames)} != {len(audio)}"
            )

        gc.collect()
        empty_cache()
        os.makedirs(XINFERENCE_VIDEO_DIR, exist_ok=True)

        audio_sample_rate = self._model.vocoder.config.output_sampling_rate
        if (
            not isinstance(audio_sample_rate, int)
            or isinstance(audio_sample_rate, bool)
            or audio_sample_rate <= 0
        ):
            raise RuntimeError(
                "LTX-2.5 vocoder returned an invalid output sampling rate: "
                f"{audio_sample_rate!r}"
            )
        paths: List[str] = []
        try:
            for video_frames, waveform in zip(frames, audio):
                path = os.path.join(XINFERENCE_VIDEO_DIR, uuid.uuid4().hex + ".mp4")
                paths.append(path)
                encode_video(
                    video_frames,
                    fps=int(frame_rate),
                    output_path=path,
                    audio=waveform.float().cpu(),
                    audio_sample_rate=audio_sample_rate,
                )
            return self._paths_to_video_list(paths, response_format)
        except Exception:
            for path in paths:
                if os.path.exists(path):
                    os.remove(path)
            raise
