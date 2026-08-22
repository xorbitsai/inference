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
import binascii
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional
from urllib.parse import urlparse

from ...constants import XINFERENCE_CACHE_DIR, XINFERENCE_WORLD_DIR
from ...types import Video, VideoList

if TYPE_CHECKING:
    from ...core.progress_tracker import Progressor
    from .core import WorldModelFamilyV1

logger = logging.getLogger(__name__)

_MAX_INPUT_BYTES = 512 * 1024 * 1024


def _tail(path: str, limit: int = 64 * 1024) -> str:
    try:
        with open(path, "rb") as fd:
            fd.seek(0, os.SEEK_END)
            size = fd.tell()
            fd.seek(max(size - limit, 0))
            return fd.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _validate_checkout(path: Path, revision: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return result.stdout.strip() == revision


def _ensure_source_checkout(
    source_url: str,
    source_revision: str,
    source_subdir: Optional[str],
    source_path: Optional[str] = None,
) -> str:
    if source_path is not None:
        checkout = Path(source_path).expanduser().resolve()
        if not checkout.is_dir():
            raise ValueError(f"World model source path does not exist: {checkout}")
    else:
        from filelock import FileLock

        cache_root = Path(XINFERENCE_CACHE_DIR) / "world_code"
        cache_root.mkdir(parents=True, exist_ok=True)
        checkout = cache_root / source_revision
        with FileLock(str(checkout) + ".lock"):
            if not _validate_checkout(checkout, source_revision):
                temp_checkout = Path(
                    tempfile.mkdtemp(prefix="world-code-", dir=str(cache_root))
                )
                try:
                    subprocess.run(
                        ["git", "init", str(temp_checkout)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(temp_checkout),
                            "remote",
                            "add",
                            "origin",
                            source_url,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(temp_checkout),
                            "fetch",
                            "--depth",
                            "1",
                            "origin",
                            source_revision,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            str(temp_checkout),
                            "checkout",
                            "--detach",
                            "FETCH_HEAD",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    if checkout.exists():
                        shutil.rmtree(checkout)
                    os.replace(temp_checkout, checkout)
                except BaseException:
                    shutil.rmtree(temp_checkout, ignore_errors=True)
                    raise

    code_path = checkout / source_subdir if source_subdir else checkout
    code_path = code_path.resolve()
    if not code_path.is_dir():
        raise RuntimeError(f"World model source directory is missing: {code_path}")
    return str(code_path)


@contextmanager
def _materialize_reference(
    reference: Optional[str], suffix: str
) -> Iterator[Optional[str]]:
    if reference is None:
        yield None
        return
    if os.path.isfile(reference):
        yield os.path.realpath(reference)
        return

    data: bytes
    parsed = urlparse(reference)
    if parsed.scheme in ("http", "https"):
        import requests

        with requests.get(reference, stream=True, timeout=30) as response:
            response.raise_for_status()
            chunks = []
            size = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                size += len(chunk)
                if size > _MAX_INPUT_BYTES:
                    raise ValueError("World generation input exceeds 512 MiB")
                chunks.append(chunk)
            data = b"".join(chunks)
    else:
        encoded = reference
        if reference.startswith("data:"):
            try:
                header, encoded = reference.split(",", 1)
            except ValueError:
                raise ValueError(
                    "Invalid data URL for world generation input"
                ) from None
            if ";base64" not in header:
                raise ValueError("World generation data URLs must use base64")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError(
                "World generation inputs must be a local path, HTTP(S) URL, "
                "or base64-encoded data"
            ) from None
        if len(data) > _MAX_INPUT_BYTES:
            raise ValueError("World generation input exceeds 512 MiB")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(data)
        temp_path = temp_file.name
    try:
        yield temp_path
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


class WorldModel:
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "WorldModelFamilyV1",
        **kwargs,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._source_path = kwargs.pop("model_code_path", None)
        self._kwargs = kwargs
        self._code_path: Optional[str] = None

    @property
    def model_ability(self):
        return self._model_spec.model_ability

    @property
    def model_family(self):
        return self._model_spec

    def load(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{self._model_spec.model_name} requires an NVIDIA CUDA GPU"
            )
        self._code_path = _ensure_source_checkout(
            self._model_spec.source_url,
            self._model_spec.source_revision,
            self._model_spec.source_subdir,
            self._source_path,
        )

    def _download_auxiliary_model(
        self, allow_patterns: Optional[List[str]] = None
    ) -> str:
        download_kwargs: Dict[str, Any] = {
            "revision": self._model_spec.auxiliary_model_revision,
        }
        if allow_patterns:
            download_kwargs["allow_patterns"] = allow_patterns

        if self._model_spec.model_hub == "modelscope":
            from modelscope.hub.snapshot_download import snapshot_download
        else:
            from huggingface_hub import snapshot_download

        return snapshot_download(
            self._model_spec.auxiliary_model_id,
            **download_kwargs,
        )

    def _gpu_count(self) -> int:
        import torch

        return max(torch.cuda.device_count(), 1)

    def _torchrun_command(self, script: str) -> List[str]:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={self._gpu_count()}",
            script,
        ]

    def _run_command(
        self,
        command: List[str],
        cwd: str,
        env: Dict[str, str],
        log_path: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        env = env.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                if progress_callback is not None:
                    progress_callback(line)
            returncode = process.wait()
        if returncode != 0:
            output = _tail(log_path)
            raise RuntimeError(
                f"World generation runner exited with code {returncode}.\n" f"{output}"
            )

    @staticmethod
    def _make_response(video_path: str, response_format: str) -> VideoList:
        if response_format == "url":
            os.makedirs(XINFERENCE_WORLD_DIR, exist_ok=True)
            output_path = os.path.join(XINFERENCE_WORLD_DIR, uuid.uuid4().hex + ".mp4")
            shutil.move(video_path, output_path)
            video = Video(url=output_path, b64_json=None)
        elif response_format == "b64_json":
            with open(video_path, "rb") as video_file:
                encoded = base64.b64encode(video_file.read()).decode("ascii")
            video = Video(url=None, b64_json=encoded)
        else:
            raise ValueError(
                "Unsupported response_format for world generation: "
                f"{response_format}"
            )
        return VideoList(created=int(time.time()), data=[video])

    @staticmethod
    def _merge_configs(
        defaults: Optional[Dict[str, Any]],
        generation_config: Optional[Dict[str, Any]],
        model_kwargs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        generation_config = generation_config or {}
        model_kwargs = model_kwargs or {}
        conflicts = set(generation_config).intersection(model_kwargs)
        if conflicts:
            raise ValueError(
                "generation_config and extra_body contain duplicate keys: "
                + ", ".join(sorted(conflicts))
            )
        config = (defaults or {}).copy()
        config.update(generation_config)
        config.update(model_kwargs)
        return config

    def world_generate(
        self,
        prompt: str,
        image: Optional[str] = None,
        video: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        progressor: Optional["Progressor"] = None,
    ) -> VideoList:
        raise NotImplementedError


class MatrixGameModel(WorldModel):
    _SUPPORTED_CONFIG = {
        "compile_vae",
        "convert_model_dtype",
        "dit_fsdp",
        "fa_version",
        "lightvae_pruning_rate",
        "num_frames",
        "num_inference_steps",
        "num_iterations",
        "response_format",
        "sample_guide_scale",
        "sample_shift",
        "seed",
        "size",
        "async_vae_warmup_iters",
        "t5_cpu",
        "t5_fsdp",
        "ulysses_size",
        "use_async_vae",
        "use_base_model",
        "use_int8",
        "vae_type",
        "verify_quant",
    }

    def world_generate(
        self,
        prompt: str,
        image: Optional[str] = None,
        video: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        progressor: Optional["Progressor"] = None,
    ) -> VideoList:
        if self._code_path is None:
            raise RuntimeError("World model is not loaded")
        if not image:
            raise ValueError("Matrix-Game-3.0-5B requires an input image")
        if video is not None:
            raise ValueError("Matrix-Game-3.0-5B does not support video input")

        explicit_config = {
            **(generation_config or {}),
            **(model_kwargs or {}),
        }
        if (
            "guidance_scale" in explicit_config
            and "sample_guide_scale" in explicit_config
        ):
            raise ValueError("Use only one of guidance_scale and sample_guide_scale")
        config = self._merge_configs(
            self._model_spec.default_generate_config,
            generation_config,
            model_kwargs,
        )
        if "sample_guide_scale" in explicit_config:
            config.pop("guidance_scale", None)
        elif "guidance_scale" in config:
            config["sample_guide_scale"] = config.pop("guidance_scale")
        if "num_frames" in config:
            num_frames = int(config.pop("num_frames"))
            if num_frames < 57 or (num_frames - 57) % 40:
                raise ValueError(
                    "Matrix-Game num_frames must equal 57 + 40 * k for k >= 0"
                )
            config["num_iterations"] = (num_frames - 57) // 40 + 1
        unknown = set(config).difference(self._SUPPORTED_CONFIG)
        if unknown:
            raise ValueError(
                "Unsupported Matrix-Game generation options: "
                + ", ".join(sorted(unknown))
            )

        response_format = str(config.pop("response_format", "url"))
        with tempfile.TemporaryDirectory(prefix="xinference-world-") as output_dir:
            with _materialize_reference(image, ".png") as image_path:
                command = self._torchrun_command("generate.py")
                command.extend(
                    [
                        "--ckpt_dir",
                        self._model_path,
                        "--prompt",
                        prompt,
                        "--image",
                        str(image_path),
                        "--output_dir",
                        output_dir,
                        "--save_name",
                        "world",
                    ]
                )
                value_options = {
                    "size": "--size",
                    "ulysses_size": "--ulysses_size",
                    "num_iterations": "--num_iterations",
                    "num_inference_steps": "--num_inference_steps",
                    "sample_guide_scale": "--sample_guide_scale",
                    "sample_shift": "--sample_shift",
                    "seed": "--seed",
                    "lightvae_pruning_rate": "--lightvae_pruning_rate",
                    "vae_type": "--vae_type",
                    "fa_version": "--fa_version",
                    "async_vae_warmup_iters": "--async_vae_warmup_iters",
                }
                for key, option in value_options.items():
                    if config.get(key) is not None:
                        command.extend([option, str(config[key])])
                flag_options = {
                    "compile_vae": "--compile_vae",
                    "convert_model_dtype": "--convert_model_dtype",
                    "dit_fsdp": "--dit_fsdp",
                    "t5_cpu": "--t5_cpu",
                    "t5_fsdp": "--t5_fsdp",
                    "use_async_vae": "--use_async_vae",
                    "use_base_model": "--use_base_model",
                    "use_int8": "--use_int8",
                    "verify_quant": "--verify_quant",
                }
                if self._gpu_count() > 1:
                    config.setdefault("dit_fsdp", True)
                    config.setdefault("t5_fsdp", True)
                for key, option in flag_options.items():
                    if config.get(key):
                        command.append(option)

                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(
                    [self._code_path, env.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep)
                if progressor:
                    progressor.set_progress(0.02, "Starting Matrix-Game runner")
                self._run_command(
                    command,
                    self._code_path,
                    env,
                    os.path.join(output_dir, "runner.log"),
                )
                video_path = os.path.join(output_dir, "world.mp4")
                if not os.path.isfile(video_path):
                    raise RuntimeError("Matrix-Game runner did not produce a video")
                return self._make_response(video_path, response_format)


class HYWorldPlayModel(WorldModel):
    _SUPPORTED_CONFIG = {
        "negative_prompt",
        "num_chunk",
        "num_frames",
        "num_inference_steps",
        "pose",
        "response_format",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_model_path: Optional[str] = None

    def load(self):
        super().load()
        if not self._model_spec.auxiliary_model_id:
            raise RuntimeError("HY-WorldPlay is missing its WAN base model")
        self._base_model_path = self._download_auxiliary_model()

    def world_generate(
        self,
        prompt: str,
        image: Optional[str] = None,
        video: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        progressor: Optional["Progressor"] = None,
    ) -> VideoList:
        if self._code_path is None or self._base_model_path is None:
            raise RuntimeError("World model is not loaded")
        if video is not None:
            raise ValueError("HY-WorldPlay-5B does not support video input")

        config = self._merge_configs(
            self._model_spec.default_generate_config,
            generation_config,
            model_kwargs,
        )
        unknown = set(config).difference(self._SUPPORTED_CONFIG)
        if unknown:
            raise ValueError(
                "Unsupported HY-WorldPlay generation options: "
                + ", ".join(sorted(unknown))
            )
        response_format = str(config.pop("response_format", "url"))

        with tempfile.TemporaryDirectory(prefix="xinference-world-") as output_dir:
            with _materialize_reference(image, ".png") as image_path:
                command = self._torchrun_command("wan/generate.py")
                command.extend(
                    [
                        "--input",
                        prompt,
                        "--model_id",
                        self._base_model_path,
                        "--ar_model_path",
                        os.path.join(self._model_path, "wan_transformer"),
                        "--ckpt_path",
                        os.path.join(
                            self._model_path, "wan_distilled_model", "model.pt"
                        ),
                        "--out",
                        output_dir,
                    ]
                )
                if image_path:
                    command.extend(["--image_path", image_path])
                value_options = {
                    "negative_prompt": "--negative_prompt",
                    "num_chunk": "--num_chunk",
                    "num_frames": "--num_frames",
                    "num_inference_steps": "--num_inference_steps",
                    "pose": "--pose",
                }
                for key, option in value_options.items():
                    if config.get(key) is not None:
                        command.extend([option, str(config[key])])

                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(
                    [
                        self._code_path,
                        os.path.join(self._code_path, "wan"),
                        env.get("PYTHONPATH", ""),
                    ]
                ).rstrip(os.pathsep)
                if progressor:
                    progressor.set_progress(0.02, "Starting HY-WorldPlay runner")
                self._run_command(
                    command,
                    self._code_path,
                    env,
                    os.path.join(output_dir, "runner.log"),
                )
                videos = list(Path(output_dir).glob("*.mp4"))
                if len(videos) != 1:
                    raise RuntimeError(
                        "HY-WorldPlay runner did not produce exactly one video"
                    )
                return self._make_response(str(videos[0]), response_format)


class AstraModel(WorldModel):
    _SUPPORTED_CONFIG = {
        "add_icons",
        "cam_type",
        "camera_guidance_scale",
        "frames_per_generation",
        "initial_condition_frames",
        "max_history_frames",
        "modality_type",
        "moe_hidden_dim",
        "moe_num_experts",
        "moe_top_k",
        "response_format",
        "start_frame",
        "text_guidance_scale",
        "total_frames_to_generate",
        "use_camera_cfg",
        "use_gt_prompt",
    }
    _MODALITY_TYPES = {"sekai", "nuscenes", "openx"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_model_path: Optional[str] = None

    def load(self):
        super().load()
        if not self._model_spec.auxiliary_model_id:
            raise RuntimeError("Astra is missing its Wan2.1 base model")
        self._base_model_path = self._download_auxiliary_model(
            self._model_spec.auxiliary_model_allow_patterns,
        )
        checkpoint_path = self._checkpoint_path()
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(f"Astra checkpoint is missing: {checkpoint_path}")

    def _checkpoint_path(self) -> str:
        return os.path.join(
            self._model_path,
            "models",
            "Astra",
            "checkpoints",
            "diffusion_pytorch_model.ckpt",
        )

    def world_generate(
        self,
        prompt: str,
        image: Optional[str] = None,
        video: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        progressor: Optional["Progressor"] = None,
    ) -> VideoList:
        if self._code_path is None or self._base_model_path is None:
            raise RuntimeError("World model is not loaded")
        if not image:
            raise ValueError("Astra requires an input image")
        if video is not None:
            raise ValueError("Astra does not support video input")

        config = self._merge_configs(
            self._model_spec.default_generate_config,
            generation_config,
            model_kwargs,
        )
        unknown = set(config).difference(self._SUPPORTED_CONFIG)
        if unknown:
            raise ValueError(
                "Unsupported Astra generation options: " + ", ".join(sorted(unknown))
            )

        cam_type = int(config.get("cam_type", 1))
        if cam_type not in range(1, 8):
            raise ValueError("Astra cam_type must be between 1 and 7")
        modality_type = str(config.get("modality_type", "sekai"))
        if modality_type not in self._MODALITY_TYPES:
            raise ValueError(
                "Astra modality_type must be one of: "
                + ", ".join(sorted(self._MODALITY_TYPES))
            )
        positive_options = (
            "initial_condition_frames",
            "frames_per_generation",
            "total_frames_to_generate",
            "moe_num_experts",
            "moe_top_k",
        )
        for key in positive_options:
            if int(config[key]) <= 0:
                raise ValueError(f"Astra {key} must be greater than 0")
        if int(config["start_frame"]) < 0:
            raise ValueError("Astra start_frame must not be negative")
        if int(config["max_history_frames"]) < 2:
            raise ValueError("Astra max_history_frames must be at least 2")

        response_format = str(config.pop("response_format", "url"))
        with tempfile.TemporaryDirectory(prefix="xinference-world-") as output_dir:
            with _materialize_reference(image, ".png") as image_path:
                output_path = os.path.join(output_dir, "world.mp4")
                command = [sys.executable, "scripts/infer_demo.py"]
                command.extend(
                    [
                        "--condition_image",
                        str(image_path),
                        "--dit_path",
                        self._checkpoint_path(),
                        "--wan_model_path",
                        self._base_model_path,
                        "--output_path",
                        output_path,
                        "--prompt",
                        prompt,
                    ]
                )
                value_options = {
                    "start_frame": "--start_frame",
                    "initial_condition_frames": "--initial_condition_frames",
                    "frames_per_generation": "--frames_per_generation",
                    "total_frames_to_generate": "--total_frames_to_generate",
                    "max_history_frames": "--max_history_frames",
                    "modality_type": "--modality_type",
                    "camera_guidance_scale": "--camera_guidance_scale",
                    "text_guidance_scale": "--text_guidance_scale",
                    "moe_num_experts": "--moe_num_experts",
                    "moe_top_k": "--moe_top_k",
                    "moe_hidden_dim": "--moe_hidden_dim",
                    "cam_type": "--cam_type",
                }
                for key, option in value_options.items():
                    if config.get(key) is not None:
                        value = cam_type if key == "cam_type" else config[key]
                        command.extend([option, str(value)])
                if config.get("use_camera_cfg"):
                    # The upstream argument expects a value rather than being
                    # a conventional store_true flag.
                    command.extend(["--use_camera_cfg", "true"])
                for key, option in {
                    "add_icons": "--add_icons",
                    "use_gt_prompt": "--use_gt_prompt",
                }.items():
                    if config.get(key):
                        command.append(option)

                env = os.environ.copy()
                env["PYTHONPATH"] = os.pathsep.join(
                    [self._code_path, env.get("PYTHONPATH", "")]
                ).rstrip(os.pathsep)
                generation_count = (
                    int(config["total_frames_to_generate"])
                    + int(config["frames_per_generation"])
                    - 1
                ) // int(config["frames_per_generation"])
                current_generation = 0

                def update_progress(line: str) -> None:
                    nonlocal current_generation
                    if progressor is None:
                        return
                    if "Starting MoE FramePack" in line:
                        progressor.set_progress(0.03, "Loading Astra weights")
                    elif "Loading initial condition frames" in line:
                        progressor.set_progress(0.10, "Encoding input image")
                    elif match := re.search(r"Generation step (\d+)", line):
                        current_generation = int(match.group(1))
                        progressor.set_progress(
                            0.15 + 0.75 * (current_generation - 1) / generation_count,
                            f"Generating chunk {current_generation}/{generation_count}",
                        )
                    elif match := re.search(r"Denoising step (\d+)/(\d+)", line):
                        step, total_steps = map(int, match.groups())
                        completed = (current_generation - 1) + step / total_steps
                        progressor.set_progress(
                            0.15 + 0.75 * completed / generation_count,
                            f"Denoising chunk {current_generation}/{generation_count}: "
                            f"step {step}/{total_steps}",
                        )
                    elif "Decoding generated video" in line:
                        progressor.set_progress(0.92, "Decoding video")
                    elif "Saving video to" in line:
                        progressor.set_progress(0.98, "Saving video")

                if progressor:
                    progressor.set_progress(0.01, "Starting Astra runner")
                self._run_command(
                    command,
                    self._code_path,
                    env,
                    os.path.join(output_dir, "runner.log"),
                    progress_callback=update_progress,
                )
                if not os.path.isfile(output_path):
                    raise RuntimeError("Astra runner did not produce a video")
                return self._make_response(output_path, response_format)
