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
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from packaging.requirements import Requirement

from ...constants import XINFERENCE_VIDEO_DIR
from ...types import Video, VideoList

if TYPE_CHECKING:
    import PIL.Image

    from ...core.progress_tracker import Progressor
    from .core import VideoModelFamilyV2

logger = logging.getLogger(__name__)

_WAN_CONVERSION_DTYPE = "bfloat16"
_WAN_CONVERSION_MANIFEST = ".xinference-mlx-video-conversion.json"
_WAN_CONVERSION_SCHEMA = 1


class MLXVideoModel:
    """Run ``Blaizzy/mlx-video`` models behind Xinference's Video API.

    The upstream package exposes generation functions rather than persistent
    pipeline objects. Model preparation therefore happens at ``load()``, while
    each request delegates generation to the corresponding upstream function.
    All MLX work is pinned to one thread because MLX GPU streams are
    thread-local.
    """

    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: "VideoModelFamilyV2",
        **kwargs: Any,
    ) -> None:
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []
        self._enable_virtual_env = bool(
            kwargs.pop("_xinference_enable_virtual_env", False)
        )
        self._virtual_env_packages: List[str] = list(
            kwargs.pop("_xinference_virtual_env_packages", None) or []
        )
        self._kwargs = kwargs
        self._runtime_model_path = model_path
        self._text_encoder_path: Optional[str] = None
        self._mlx_executor: Optional[ThreadPoolExecutor] = None

    @property
    def model_spec(self) -> "VideoModelFamilyV2":
        return self._model_spec

    @property
    def model_ability(self) -> List[str]:
        return self._abilities

    def _run_on_mlx_thread(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if self._mlx_executor is None:
            self._mlx_executor = ThreadPoolExecutor(max_workers=1)
        return self._mlx_executor.submit(fn, *args, **kwargs).result()

    @staticmethod
    def _is_wan_mlx_model_dir(path: Path) -> bool:
        shared_files = (
            path / "config.json",
            path / "t5_encoder.safetensors",
            path / "vae.safetensors",
        )
        if not all(file.is_file() for file in shared_files):
            return False
        return (path / "model.safetensors").is_file() or all(
            (path / filename).is_file()
            for filename in (
                "high_noise_model.safetensors",
                "low_noise_model.safetensors",
            )
        )

    @classmethod
    def _wan_source_fingerprint(cls, source_path: Path) -> Dict[str, Dict[str, int]]:
        fingerprint: Dict[str, Dict[str, int]] = {}
        for path in source_path.rglob("*"):
            try:
                if not path.is_file():
                    continue
                stat = path.stat()
            except OSError:
                continue
            fingerprint[path.relative_to(source_path).as_posix()] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        return fingerprint

    @staticmethod
    def _converter_requirement(packages: List[str]) -> Optional[str]:
        for package in packages:
            requirement_text = package.partition(";")[0].strip()
            try:
                if Requirement(requirement_text).name.lower() == "mlx-video":
                    return requirement_text
            except Exception:
                if requirement_text.lower().startswith("mlx-video"):
                    return requirement_text
        return None

    def _converter_runtime_identity(self) -> Dict[str, Any]:
        """Describe the converter that is actually imported in this runtime."""

        runtime: Dict[str, Any] = {
            "environment": "virtualenv" if self._enable_virtual_env else "host"
        }
        try:
            distribution = importlib.metadata.distribution("mlx-video")
            runtime["distribution_version"] = distribution.version
            direct_url = distribution.read_text("direct_url.json")
            if direct_url:
                runtime["direct_url"] = json.loads(direct_url)
        except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
            module = importlib.import_module("mlx_video")
            module_path = getattr(module, "__file__", None)
            if module_path:
                try:
                    stat = Path(module_path).stat()
                    runtime["module"] = {
                        "path": str(Path(module_path).resolve()),
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                    }
                except OSError:
                    runtime["module"] = {"path": module_path}
        return runtime

    @classmethod
    def _convert_wan_model(
        cls,
        source_path: Path,
        conversion_identity: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Convert an official Wan checkpoint into mlx-video's native layout."""

        from filelock import FileLock

        converted_path = Path(f"{source_path}.mlx-video")
        lock_path = Path(f"{converted_path}.lock")
        with FileLock(str(lock_path), preserve_lock_file=True):
            expected_manifest = {
                "schema_version": _WAN_CONVERSION_SCHEMA,
                "source": conversion_identity or {},
                "source_files": cls._wan_source_fingerprint(source_path),
                "dtype": _WAN_CONVERSION_DTYPE,
                "output_schema": "mlx-video-wan-native-v1",
            }
            manifest_path = converted_path / _WAN_CONVERSION_MANIFEST
            if cls._is_wan_mlx_model_dir(converted_path):
                try:
                    if json.loads(manifest_path.read_text()) == expected_manifest:
                        return converted_path
                except (OSError, ValueError, TypeError):
                    pass

            if converted_path.exists() and not manifest_path.is_file():
                raise FileExistsError(
                    f"Refusing to overwrite unmanaged Wan conversion target "
                    f"{converted_path}. Move or remove it explicitly before retrying."
                )
            if converted_path.is_dir():
                shutil.rmtree(converted_path)
            elif converted_path.exists():
                converted_path.unlink()

            converted_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = Path(
                tempfile.mkdtemp(
                    prefix=f".{converted_path.name}-",
                    dir=str(converted_path.parent),
                )
            )
            try:
                convert_module = importlib.import_module(
                    "mlx_video.models.wan_2.convert"
                )
                convert_module.convert_wan_checkpoint(
                    str(source_path), str(temp_path), dtype=_WAN_CONVERSION_DTYPE
                )
                if not cls._is_wan_mlx_model_dir(temp_path):
                    raise RuntimeError(
                        f"mlx-video produced an incomplete Wan model in {temp_path}"
                    )
                (temp_path / _WAN_CONVERSION_MANIFEST).write_text(
                    json.dumps(expected_manifest, sort_keys=True)
                )
                os.replace(temp_path, converted_path)
            finally:
                if temp_path.exists():
                    shutil.rmtree(temp_path)
        return converted_path

    def _prepare_wan_model(self) -> None:
        model_path = Path(self._model_path)
        if self._is_wan_mlx_model_dir(model_path):
            self._runtime_model_path = str(model_path)
        else:
            virtualenv = getattr(self._model_spec, "virtualenv", None)
            spec_packages = getattr(virtualenv, "packages", None) or []
            converter_requirement = self._converter_requirement(
                self._virtual_env_packages
            ) or self._converter_requirement(spec_packages)
            converter_identity = {
                "requirement": converter_requirement,
                "runtime": self._converter_runtime_identity(),
            }
            self._runtime_model_path = str(
                self._convert_wan_model(
                    model_path,
                    {
                        "model_id": getattr(self._model_spec, "model_id", None),
                        "model_revision": getattr(
                            self._model_spec, "model_revision", None
                        ),
                        "converter": converter_identity,
                    },
                )
            )

    def _prepare_ltx_model(self) -> None:
        """Prepare an LTX model and its optional external text encoder."""

        model_path = Path(self._model_path)
        text_encoder_model_id = getattr(self._model_spec, "text_encoder_model_id", None)
        if text_encoder_model_id is None:
            self._runtime_model_path = str(model_path)
            self._text_encoder_path = None
            return

        if self._model_spec.model_hub == "modelscope":
            from modelscope.hub.snapshot_download import snapshot_download
        else:
            from huggingface_hub import snapshot_download

        text_encoder_path = Path(
            snapshot_download(
                text_encoder_model_id,
                revision=getattr(self._model_spec, "text_encoder_model_revision", None),
                allow_patterns=["text_encoder/**", "tokenizer/**"],
            )
        )
        self._runtime_model_path = str(model_path)
        self._text_encoder_path = str(text_encoder_path)

    def load(self) -> None:
        self._run_on_mlx_thread(self._load)

    def stop(self) -> None:
        if self._mlx_executor is not None:
            self._mlx_executor.shutdown(wait=True)
            self._mlx_executor = None

    def _load(self) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("The MLX video engine requires Apple Silicon")
        if sys.version_info < (3, 11):
            raise RuntimeError("Blaizzy/mlx-video requires Python 3.11 or newer")
        importlib.import_module("mlx_video")
        if self._model_spec.model_family == "Wan":
            self._prepare_wan_model()
        elif self._model_spec.model_family in ("LTX-2", "LTX-2.3"):
            self._prepare_ltx_model()
        else:
            raise ValueError(
                f"Unsupported mlx-video model family: {self._model_spec.model_family}"
            )

    @staticmethod
    def _save_conditioning_image(image: "PIL.Image.Image") -> str:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            image.save(path, format="PNG")
        except BaseException:
            os.remove(path)
            raise
        return path

    def _generation_kwargs(self, request_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        result = (self._model_spec.default_generate_config or {}).copy()
        # ``self._kwargs`` contains model-launch options (for example the Web
        # UI always sends ``cpu_offload``).  They are not generation options
        # and must not leak into mlx-video's strict ``generate_video``
        # signatures.
        result.update(request_kwargs)
        return {key: value for key, value in result.items() if value is not None}

    @staticmethod
    def _normalize_prompts(
        prompt: Optional[Union[str, List[str]]], field_name: str
    ) -> List[str]:
        if prompt is None:
            return [""]
        if isinstance(prompt, str):
            return [prompt]
        if not isinstance(prompt, list):
            raise TypeError(f"{field_name} must be a string or a list of strings")
        if not prompt:
            raise ValueError(f"{field_name} list must not be empty")
        if not all(isinstance(value, str) for value in prompt):
            raise TypeError(f"{field_name} must contain only strings")
        return prompt

    def _generate_wan(
        self,
        prompt: str,
        output_path: str,
        image_path: Optional[str],
        num_inference_steps: Optional[int],
        request_kwargs: Dict[str, Any],
    ) -> None:
        generate_kwargs = self._generation_kwargs(request_kwargs)
        generate_kwargs.pop("output_path", None)
        generate_kwargs.pop("model_dir", None)
        generate_kwargs.pop("fps", None)
        generate_kwargs.pop("pipeline", None)
        if "guidance_scale" in generate_kwargs:
            generate_kwargs["guide_scale"] = generate_kwargs.pop("guidance_scale")
        if "cfg_scale" in generate_kwargs:
            generate_kwargs["guide_scale"] = generate_kwargs.pop("cfg_scale")
        if num_inference_steps is not None:
            generate_kwargs["steps"] = num_inference_steps

        generate_module = importlib.import_module("mlx_video.models.wan_2.generate")
        generate_module.generate_video(
            model_dir=self._runtime_model_path,
            prompt=prompt,
            image=image_path,
            output_path=output_path,
            **generate_kwargs,
        )

    def _generate_ltx(
        self,
        prompt: str,
        output_path: str,
        image_path: Optional[str],
        last_image_path: Optional[str],
        num_inference_steps: Optional[int],
        request_kwargs: Dict[str, Any],
    ) -> None:
        generate_kwargs = self._generation_kwargs(request_kwargs)
        generate_kwargs.pop("output_path", None)
        generate_kwargs.pop("model_repo", None)
        generate_kwargs.pop("text_encoder_repo", None)
        generate_kwargs["save_frames"] = False
        pipeline_name = generate_kwargs.pop("pipeline", "distilled")
        if "guidance_scale" in generate_kwargs:
            generate_kwargs["cfg_scale"] = generate_kwargs.pop("guidance_scale")
        if "guide_scale" in generate_kwargs:
            generate_kwargs["cfg_scale"] = generate_kwargs.pop("guide_scale")
        if num_inference_steps is not None:
            generate_kwargs["num_inference_steps"] = num_inference_steps

        generate_module = importlib.import_module("mlx_video.models.ltx_2.generate")
        pipeline = (
            pipeline_name
            if isinstance(pipeline_name, generate_module.PipelineType)
            else generate_module.PipelineType(pipeline_name)
        )
        audio_path = str(Path(output_path).with_suffix(".wav"))
        generate_kwargs["output_audio_path"] = audio_path
        try:
            generate_module.generate_video(
                model_repo=self._runtime_model_path,
                text_encoder_repo=self._text_encoder_path,
                prompt=prompt,
                pipeline=pipeline,
                image=image_path,
                end_image=last_image_path,
                output_path=output_path,
                **generate_kwargs,
            )
        finally:
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    def _generate(
        self,
        prompt: Optional[Union[str, List[str]]],
        n: int,
        num_inference_steps: Optional[int],
        response_format: str,
        image: Optional["PIL.Image.Image"],
        last_image: Optional["PIL.Image.Image"],
        request_kwargs: Dict[str, Any],
    ) -> VideoList:
        if n < 1:
            raise ValueError("n must be at least 1")
        if response_format not in ("url", "b64_json"):
            raise ValueError(f"Unsupported response format: {response_format}")
        if last_image is not None:
            if "firstlastframe2video" not in self._abilities:
                raise ValueError(
                    f"{self._model_spec.model_name} does not support "
                    "firstlastframe2video"
                )
        elif image is not None and "image2video" not in self._abilities:
            raise ValueError(
                f"{self._model_spec.model_name} does not support image2video"
            )
        elif image is None and "text2video" not in self._abilities:
            raise ValueError(
                f"{self._model_spec.model_name} does not support text2video"
            )
        prompts = self._normalize_prompts(prompt, "prompt")
        negative_prompt = request_kwargs.pop("negative_prompt", None)
        negative_prompts: List[Optional[str]] = []
        if isinstance(negative_prompt, list):
            negative_prompts.extend(
                self._normalize_prompts(negative_prompt, "negative_prompt")
            )
            if len(negative_prompts) != len(prompts):
                raise ValueError(
                    "negative_prompt list length must match prompt list length"
                )
        elif negative_prompt is None or isinstance(negative_prompt, str):
            negative_prompts = [negative_prompt] * len(prompts)
        else:
            raise TypeError("negative_prompt must be a string or a list of strings")

        progressor: Optional["Progressor"] = request_kwargs.pop("progressor", None)
        request_kwargs.pop("request_id", None)
        request_kwargs.pop("num_videos_per_prompt", None)

        image_paths: List[str] = []
        output_paths: List[str] = []
        try:
            image_path = None
            last_image_path = None
            if image is not None:
                image_path = self._save_conditioning_image(image)
                image_paths.append(image_path)
            if last_image is not None:
                last_image_path = self._save_conditioning_image(last_image)
                image_paths.append(last_image_path)

            os.makedirs(XINFERENCE_VIDEO_DIR, exist_ok=True)
            total_outputs = len(prompts) * n
            completed_outputs = 0
            for prompt_index, prompt_text in enumerate(prompts):
                prompt_kwargs = request_kwargs.copy()
                if negative_prompts[prompt_index] is not None:
                    prompt_kwargs["negative_prompt"] = negative_prompts[prompt_index]
                for _ in range(n):
                    output_path = os.path.join(
                        XINFERENCE_VIDEO_DIR, f"{uuid.uuid4().hex}.mp4"
                    )
                    output_paths.append(output_path)
                    if self._model_spec.model_family == "Wan":
                        self._generate_wan(
                            prompt_text,
                            output_path,
                            image_path,
                            num_inference_steps,
                            prompt_kwargs,
                        )
                    else:
                        self._generate_ltx(
                            prompt_text,
                            output_path,
                            image_path,
                            last_image_path,
                            num_inference_steps,
                            prompt_kwargs,
                        )
                    if not os.path.isfile(output_path):
                        raise RuntimeError(
                            "mlx-video did not create the expected output "
                            f"{output_path}"
                        )
                    completed_outputs += 1
                    if progressor is not None and progressor.request_id:
                        progressor.set_progress(completed_outputs / total_outputs)
            return self._video_urls_to_response(output_paths, response_format)
        except BaseException:
            for output_path in output_paths:
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            raise
        finally:
            for image_path in image_paths:
                try:
                    os.remove(image_path)
                except OSError:
                    pass

    def text_to_video(
        self,
        prompt: Union[str, List[str]],
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> VideoList:
        return self._run_on_mlx_thread(
            self._generate,
            prompt,
            n,
            num_inference_steps,
            response_format,
            None,
            None,
            kwargs,
        )

    def image_to_video(
        self,
        image: "PIL.Image.Image",
        prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> VideoList:
        return self._run_on_mlx_thread(
            self._generate,
            prompt,
            n,
            num_inference_steps,
            response_format,
            image,
            None,
            kwargs,
        )

    def firstlastframe_to_video(
        self,
        first_frame: "PIL.Image.Image",
        last_frame: "PIL.Image.Image",
        prompt: Optional[Union[str, List[str]]] = None,
        n: int = 1,
        num_inference_steps: Optional[int] = None,
        response_format: str = "b64_json",
        **kwargs: Any,
    ) -> VideoList:
        return self._run_on_mlx_thread(
            self._generate,
            prompt,
            n,
            num_inference_steps,
            response_format,
            first_frame,
            last_frame,
            kwargs,
        )

    @staticmethod
    def _video_urls_to_response(urls: List[str], response_format: str) -> VideoList:
        if response_format == "url":
            return VideoList(
                created=int(time.time()),
                data=[Video(url=url, b64_json=None) for url in urls],
            )
        if response_format != "b64_json":
            raise ValueError(f"Unsupported response format: {response_format}")

        data = []
        try:
            for url in urls:
                with open(url, "rb") as video_file:
                    encoded = base64.b64encode(video_file.read()).decode()
                data.append(Video(url=None, b64_json=encoded))
            return VideoList(created=int(time.time()), data=data)
        finally:
            for url in urls:
                try:
                    os.remove(url)
                except OSError:
                    pass
