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

import os
import shutil
import sys
import tempfile
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional

from ...device_utils import get_available_device

if TYPE_CHECKING:
    from .core import AudioModelFamilyV2


_ACE_STEP_VENDOR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../thirdparty/ace_step_1_5")
)
_NANO_VLLM_VENDOR_ROOT = os.path.join(
    _ACE_STEP_VENDOR_ROOT, "acestep", "third_parts", "nano-vllm"
)


def _ensure_vendored_source_paths() -> None:
    """Expose ACE-Step and its bundled nano-vllm as top-level packages."""

    for source_path in (_ACE_STEP_VENDOR_ROOT, _NANO_VLLM_VENDOR_ROOT):
        if not os.path.isdir(source_path):
            raise RuntimeError(
                f"ACE-Step 1.5 vendored source directory is missing: {source_path}"
            )
        if source_path in sys.path:
            sys.path.remove(source_path)
        sys.path.insert(0, source_path)


def is_ace_step_python_supported() -> bool:
    """ACE-Step 1.5 currently publishes dependencies for Python 3.11-3.12."""

    return sys.version_info[:2] in ((3, 11), (3, 12))


@contextmanager
def _ace_step_environment(checkpoint_dir: str, project_root: str) -> Iterator[None]:
    values = {
        "ACESTEP_CHECKPOINTS_DIR": checkpoint_dir,
        "ACESTEP_PROJECT_ROOT": project_root,
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class AceStepModel:
    """ACE-Step 1.5 text-to-music model backed by the upstream Python API."""

    _environment_lock = threading.Lock()
    _bundled_dit_model = "acestep-v15-turbo"
    _bundled_lm_model = "acestep-5Hz-lm-1.7B"
    _lm_backends = {"mlx", "pt", "vllm"}
    _response_formats = {"aac", "flac", "mp3", "opus", "wav", "wav32"}
    _generation_options = {
        "bpm",
        "cfg_interval_end",
        "cfg_interval_start",
        "dcw_enabled",
        "dcw_high_scaler",
        "dcw_mode",
        "dcw_scaler",
        "dcw_wavelet",
        "enable_normalization",
        "fade_in_duration",
        "fade_out_duration",
        "guidance_scale",
        "infer_method",
        "inference_steps",
        "instrumental",
        "keyscale",
        "latent_rescale",
        "latent_shift",
        "lm_cfg_scale",
        "lm_negative_prompt",
        "lm_temperature",
        "lm_top_k",
        "lm_top_p",
        "normalization_db",
        "sampler_mode",
        "shift",
        "timesignature",
        "timesteps",
        "use_adg",
        "use_constrained_decoding",
        "use_cot_caption",
        "use_cot_language",
        "use_cot_metas",
        "velocity_ema_factor",
        "velocity_norm_threshold",
        "vocal_language",
    }
    _initialize_options = {
        "compile_model",
        "offload_dit_to_cpu",
        "offload_to_cpu",
        "quantization",
        "use_flash_attention",
        "use_mlx_dit",
    }

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "AudioModelFamilyV2",
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._device = device
        self._kwargs = kwargs
        self._model = None
        self._llm_handler = None
        self._generation_params_cls = None
        self._generation_config_cls = None
        self._generate_music = None
        self._runtime_workspace: Optional[tempfile.TemporaryDirectory] = None

    @property
    def model_spec(self) -> "AudioModelFamilyV2":
        return self._model_spec

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    def _load_config(self) -> Dict[str, Any]:
        config = (self._model_spec.default_model_config or {}).copy()
        config.update(self._kwargs)
        return config

    @staticmethod
    def _link_checkpoint_file(source: str, destination: str) -> str:
        """Build a cheap writable checkpoint view without copying model weights."""

        resolved_source = os.path.realpath(source)
        if source.endswith(".py"):
            # ACE-Step synchronizes model-side Python files during initialization.
            # Copy these files so that synchronization cannot modify a Hub cache.
            return shutil.copy2(resolved_source, destination)
        try:
            os.link(resolved_source, destination)
        except OSError:
            try:
                os.symlink(resolved_source, destination)
            except OSError:
                return shutil.copy2(resolved_source, destination)
        return destination

    def _prepare_runtime_checkpoint(self) -> tuple[tempfile.TemporaryDirectory, str]:
        """Isolate mutable model code while reusing immutable checkpoint weights."""

        model_path = os.path.realpath(self._model_path)
        if not os.path.isdir(model_path):
            raise ValueError(
                f"ACE-Step 1.5 model path is not a directory: {self._model_path!r}."
            )

        preferred_parent = os.path.dirname(os.path.abspath(self._model_path))
        try:
            workspace = tempfile.TemporaryDirectory(
                prefix="xinference-ace-step-runtime-", dir=preferred_parent
            )
        except OSError:
            workspace = tempfile.TemporaryDirectory(
                prefix="xinference-ace-step-runtime-"
            )
        checkpoint_dir = os.path.join(workspace.name, "checkpoints")
        try:
            shutil.copytree(
                model_path,
                checkpoint_dir,
                copy_function=self._link_checkpoint_file,
            )
        except BaseException:
            workspace.cleanup()
            raise
        return workspace, checkpoint_dir

    def load(self) -> None:
        if not is_ace_step_python_supported():
            raise RuntimeError("ACE-Step 1.5 requires Python 3.11 or 3.12.")

        _ensure_vendored_source_paths()
        try:
            from acestep.handler import AceStepHandler
            from acestep.inference import (
                GenerationConfig,
                GenerationParams,
                generate_music,
            )
        except ImportError as e:
            raise ImportError(
                "ACE-Step 1.5 vendored runtime failed to import. Install the "
                "dependencies declared in its built-in model specification."
            ) from e

        config = self._load_config()
        config_path = config.pop("config_path", self._bundled_dit_model)
        lm_model_path = config.pop("lm_model_path", None)
        lm_backend = config.pop("lm_backend", "pt")
        lm_offload_to_cpu = config.pop("lm_offload_to_cpu", False)
        vae_checkpoint = config.pop("vae_checkpoint", "official")
        unknown = sorted(set(config) - self._initialize_options)
        if unknown:
            raise ValueError(
                "Unsupported ACE-Step 1.5 load option(s): " + ", ".join(unknown)
            )

        if config_path != self._bundled_dit_model:
            raise ValueError(
                "ACE-Step1.5 currently supports only the bundled "
                f"`config_path={self._bundled_dit_model}`."
            )
        if lm_model_path == "":
            lm_model_path = None
        if lm_model_path not in (None, self._bundled_lm_model):
            raise ValueError(
                "ACE-Step1.5 currently supports only the bundled "
                f"`lm_model_path={self._bundled_lm_model}` or no LM."
            )
        if vae_checkpoint in (None, ""):
            vae_checkpoint = "official"
        if vae_checkpoint != "official":
            raise ValueError(
                "ACE-Step1.5 currently supports only the bundled "
                "`vae_checkpoint=official`."
            )
        if (
            not isinstance(lm_backend, str)
            or lm_backend.lower() not in self._lm_backends
        ):
            backends = ", ".join(sorted(self._lm_backends))
            raise ValueError(f"ACE-Step 1.5 `lm_backend` must be one of: {backends}.")
        lm_backend = lm_backend.lower()
        if not isinstance(lm_offload_to_cpu, bool):
            raise ValueError("ACE-Step 1.5 `lm_offload_to_cpu` must be a boolean.")

        device = str(self._device or get_available_device()).split(":", 1)[0]
        if device == "rocm":
            # PyTorch exposes ROCm accelerators through the CUDA device API,
            # which is also the device name expected by ACE-Step.
            device = "cuda"
        if device not in {"auto", "cpu", "cuda", "mps", "xpu"}:
            raise ValueError(f"ACE-Step 1.5 does not support device {device!r}.")

        workspace, checkpoint_dir = self._prepare_runtime_checkpoint()
        project_root = workspace.name
        try:
            with (
                self._environment_lock,
                _ace_step_environment(checkpoint_dir, project_root),
            ):
                handler = AceStepHandler()
                status, success = handler.initialize_service(
                    project_root=project_root,
                    config_path=config_path,
                    device=device,
                    prefer_source=self._model_spec.model_hub,
                    vae_checkpoint=vae_checkpoint,
                    **config,
                )
                if not success:
                    raise RuntimeError(f"Failed to initialize ACE-Step 1.5: {status}")

                llm_handler = None
                if lm_model_path:
                    from acestep.llm_inference import LLMHandler

                    llm_handler = LLMHandler()
                    lm_status, lm_success = llm_handler.initialize(
                        checkpoint_dir=checkpoint_dir,
                        lm_model_path=lm_model_path,
                        backend=lm_backend,
                        device=device,
                        offload_to_cpu=lm_offload_to_cpu,
                    )
                    if not lm_success:
                        raise RuntimeError(
                            f"Failed to initialize ACE-Step 1.5 LM: {lm_status}"
                        )
        except BaseException:
            workspace.cleanup()
            raise

        if self._runtime_workspace is not None:
            self._runtime_workspace.cleanup()
        self._runtime_workspace = workspace
        self._model = handler
        self._llm_handler = llm_handler
        self._generation_params_cls = GenerationParams
        self._generation_config_cls = GenerationConfig
        self._generate_music = generate_music

    @staticmethod
    def _validate_request(
        input: str,
        instruct: Any,
        voice: Optional[str],
        response_format: Optional[str],
        speed: Optional[float],
        stream: Optional[bool],
        seed: Any,
        duration: Any,
    ) -> str:
        if not isinstance(input, str) or not input.strip():
            raise ValueError("ACE-Step 1.5 requires non-empty lyrics in `input`.")
        if len(input) > 4096:
            raise ValueError("ACE-Step 1.5 `input` must not exceed 4096 characters.")
        if not isinstance(instruct, str) or not instruct.strip():
            raise ValueError(
                "ACE-Step 1.5 requires a non-empty music description in `instruct`."
            )
        if len(instruct) > 512:
            raise ValueError("ACE-Step 1.5 `instruct` must not exceed 512 characters.")
        if voice not in (None, "", "default"):
            raise ValueError(
                "ACE-Step 1.5 only accepts `voice` as null, an empty string, "
                "or 'default'."
            )
        audio_format = (response_format or "").lower()
        if audio_format not in AceStepModel._response_formats:
            formats = ", ".join(sorted(AceStepModel._response_formats))
            raise ValueError(
                f"ACE-Step 1.5 supports these response formats: {formats}."
            )
        if speed != 1.0:
            raise ValueError("ACE-Step 1.5 only supports `speed=1.0`.")
        if stream is not False:
            raise ValueError("ACE-Step 1.5 only supports non-streaming generation.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < -1:
            raise ValueError(
                "ACE-Step 1.5 `seed` must be -1 or a non-negative integer."
            )
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError(
                "ACE-Step 1.5 `duration` must be -1 or a number from 10 to 600 seconds."
            )
        if duration != -1 and not 10 <= duration <= 600:
            raise ValueError(
                "ACE-Step 1.5 `duration` must be -1 or a number from 10 to 600 seconds."
            )
        return audio_format

    def speech(
        self,
        input: str,
        voice: Optional[str] = None,
        response_format: Optional[str] = "mp3",
        speed: Optional[float] = 1.0,
        stream: Optional[bool] = False,
        **kwargs: Any,
    ) -> bytes:
        assert self._model is not None
        assert self._generation_params_cls is not None
        assert self._generation_config_cls is not None
        assert self._generate_music is not None

        instruct = kwargs.pop("instruct", None)
        seed = kwargs.pop("seed", -1)
        duration = kwargs.pop("duration", 60.0)
        thinking = kwargs.pop("thinking", self._llm_handler is not None)
        if not isinstance(thinking, bool):
            raise ValueError("ACE-Step 1.5 `thinking` must be a boolean.")
        cot_options = ("use_cot_caption", "use_cot_language", "use_cot_metas")
        uses_lm = thinking or any(kwargs.get(name, False) for name in cot_options)
        if uses_lm and self._llm_handler is None:
            raise ValueError(
                "ACE-Step 1.5 LM generation options require launching with "
                f"`lm_model_path={self._bundled_lm_model}`."
            )

        audio_format = self._validate_request(
            input,
            instruct,
            voice,
            response_format,
            speed,
            stream,
            seed,
            duration,
        )

        unknown = sorted(set(kwargs) - self._generation_options)
        if unknown:
            raise ValueError(
                "ACE-Step 1.5 does not support speech parameter(s): "
                + ", ".join(unknown)
            )

        generation_kwargs = {name: kwargs[name] for name in kwargs}
        generation_kwargs.update(
            {
                "task_type": "text2music",
                "caption": instruct.strip(),
                "lyrics": input,
                "duration": float(duration),
                "seed": seed,
                "thinking": thinking,
            }
        )
        for name in cot_options:
            generation_kwargs.setdefault(name, thinking)

        params = self._generation_params_cls(**generation_kwargs)
        generation_config = self._generation_config_cls(
            batch_size=1,
            use_random_seed=seed == -1,
            seeds=None if seed == -1 else [seed],
            audio_format=audio_format,
        )

        with tempfile.TemporaryDirectory(prefix="xinference-ace-step-") as output_dir:
            result = self._generate_music(
                self._model,
                self._llm_handler,
                params,
                generation_config,
                save_dir=output_dir,
            )
            if not result.success:
                raise RuntimeError(
                    f"ACE-Step 1.5 generation failed: {result.error or result.status_message}"
                )
            if len(result.audios) != 1 or not result.audios[0].get("path"):
                raise RuntimeError("ACE-Step 1.5 returned no saved audio output.")
            audio_path = result.audios[0]["path"]
            with open(audio_path, "rb") as audio_file:
                return audio_file.read()
