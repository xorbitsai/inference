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

import asyncio
import concurrent.futures
import dataclasses
import logging
import random
import re
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from ....core.replica_load import EngineLoad
from ....types import LoRA
from ....utils import dependency_import_guard
from ...lightning import resolve_request_lightning
from ..utils import MAX_IMAGE_SEED, handle_image_result, resolve_image_seed_list

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

logger = logging.getLogger(__name__)

# Image models verified against the vllm-omni supported diffusion models:
# https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image
VLLM_SUPPORTED_IMAGE_MODELS = (
    "MiniMax-H3-Image",
    "Qwen-Image",
    "Qwen-Image-2512",
    "Qwen-Image-2.1",
    "Qwen-Image-2.1-Turbo",
    "Z-Image",
    "Z-Image-Turbo",
    "FLUX.1-dev",
    "sd3.5-medium",
)

# Abilities this engine actually implements; the builtin specs of the models
# above also advertise image2image/inpainting, which only the diffusers
# engine provides
VLLM_SUPPORTED_ABILITIES = ("text2image",)


def _filter_kwargs_by_dataclass_fields(
    kwargs: Dict[str, Any], dataclass_type: Any, purpose: str
) -> Dict[str, Any]:
    valid_keys = {f.name for f in dataclasses.fields(dataclass_type)}
    dropped = sorted(set(kwargs) - valid_keys)
    if dropped:
        logger.info("Dropping args unsupported by vLLM-Omni %s: %s", purpose, dropped)
    return {k: v for k, v in kwargs.items() if k in valid_keys}


class _RequestWaiter:
    """Bookkeeping for one in-flight engine request routed by the dispatcher."""

    __slots__ = ("future", "outputs", "queued_since")

    def __init__(self, queued_since: Optional[float] = None):
        self.future: concurrent.futures.Future = concurrent.futures.Future()
        self.outputs: List[Any] = []
        self.queued_since = queued_since


class VLLMDiffusionModel:
    supported_abilities: Tuple[str, ...] = VLLM_SUPPORTED_ABILITIES
    generation_mode = "text-to-image"

    # ModelActor skips its serializing lock when allow_batch is True.
    # Omni.generate itself is NOT safe for concurrent callers (it drains the
    # shared engine output queue and drops messages of other requests), so
    # concurrent requests are submitted through engine.add_request with a
    # single dispatcher thread routing outputs back by request_id.
    allow_batch = True

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        lora_model: Optional[List[LoRA]] = None,
        lora_load_kwargs: Optional[Dict] = None,
        lora_fuse_kwargs: Optional[Dict] = None,
        model_spec: Optional["ImageModelFamilyV2"] = None,
        gguf_model_path: Optional[str] = None,
        lightning_model_path: Optional[str] = None,
        **kwargs,
    ):
        if gguf_model_path:
            raise ValueError(
                "GGUF quantization is not supported by the vLLM diffusion engine, "
                "please use the diffusers engine instead"
            )
        if lightning_model_path:
            raise ValueError(
                "Lightning LoRA acceleration is not supported by the vLLM diffusion "
                "engine, please use the diffusers engine instead"
            )
        if lora_model:
            raise ValueError(
                "LoRA is not supported by the vLLM diffusion engine yet, "
                "please use the diffusers engine instead"
            )
        if kwargs.get("controlnet"):
            raise ValueError(
                "Controlnet is not supported by the vLLM diffusion engine, "
                "please use the diffusers engine instead"
            )
        # only advertise the abilities this engine implements, so the model
        # description does not report endpoints that would fail; copy the
        # family first to keep the caller's (possibly shared builtin) spec
        # untouched
        if model_spec is not None:
            restricted_spec = model_spec.copy()
            restricted_spec.model_ability = [
                ability
                for ability in (restricted_spec.model_ability or [])
                if ability in self.supported_abilities
            ]
            model_spec = restricted_spec
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._model_spec = model_spec
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs
        # concurrent-dispatch state, see class docstring on allow_batch
        self._submit_lock = threading.Lock()
        self._waiters: Dict[str, _RequestWaiter] = {}
        self._dispatcher_thread: Optional[threading.Thread] = None
        # serializes the Omni.generate fallback path when the engine internals
        # needed for concurrent dispatch are unavailable
        self._generate_lock = threading.Lock()
        self._closed = False
        # set on fatal engine errors; subsequent requests fail immediately
        # with this as the cause instead of hanging on a dead backend
        self._fatal_error: Optional[BaseException] = None
        self._is_request_started: Optional[Callable[[Any], bool]] = None

    def __getstate__(self):
        # the instance is pickled when handed to the model actor subprocess
        # (before load); locks and threads cannot cross that boundary
        state = self.__dict__.copy()
        state["_submit_lock"] = None
        state["_generate_lock"] = None
        state["_dispatcher_thread"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._submit_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    @property
    def model_ability(self):
        return self._abilities

    def enable_engine_load_reporting(self) -> None:
        requirement = (
            "vLLM-Omni diffusion replica autoscaling requires request lifecycle "
            "events (vllm-omni>=0.32.0)"
        )
        try:
            from vllm_omni.diffusion.data import is_diffusion_request_started_output
            from vllm_omni.inputs.data import OmniDiffusionSamplingParams
        except ImportError as exc:
            raise ImportError(requirement) from exc
        if not hasattr(OmniDiffusionSamplingParams, "emit_request_lifecycle"):
            raise ImportError(requirement)
        self._is_request_started = is_diffusion_request_started_output

    def get_engine_load(self) -> Optional[EngineLoad]:
        runner = self._model
        if (
            self._is_request_started is None
            or runner is None
            or not self._concurrency_available()
        ):
            return None
        runner.check_health()
        with self._submit_lock:
            if self._closed or self._model is not runner:
                return None
            pending = [
                waiter.queued_since
                for waiter in self._waiters.values()
                if waiter.queued_since is not None
            ]
            age = time.monotonic() - min(pending) if pending else 0.0
            return EngineLoad(len(pending), age, len(self._waiters) - len(pending))

    def load(self):
        with dependency_import_guard(
            "vllm_omni",
            install_hint="Install vllm-omni alongside a matching vLLM release.",
        ):
            from vllm_omni.entrypoints.omni import Omni

        minimum_version = (
            "0.34.0"
            if self._model_spec.model_name == "MiniMax-H3-Image"
            else (
                "0.32.0"
                if self._model_spec.model_name
                in ("Qwen-Image-2.1", "Qwen-Image-2.1-Turbo")
                else None
            )
        )
        if minimum_version is not None:
            from importlib.metadata import version

            from packaging.version import Version

            if Version(version("vllm-omni")) < Version(minimum_version):
                raise ImportError(
                    f"{self._model_spec.model_name} requires vllm-omni>={minimum_version}"
                )
        options = self._kwargs.copy()
        # Built-in image specs also serve Diffusers. Omni consumes its own
        # dtype option and does not accept the Diffusers loader setting.
        dtype = options.pop("torch_dtype", None)
        options.pop("low_cpu_mem_usage", None)
        if dtype is not None:
            options.setdefault("dtype", dtype)
        if self._model_spec.model_name == "MiniMax-H3-Image":
            cpu_offload = options.pop("cpu_offload", None)
            if cpu_offload is not None:
                options.setdefault("enable_cpu_offload", cpu_offload)
        logger.debug(
            "Loading vLLM-Omni diffusion model from %s, kwargs: %s",
            self._model_path,
            options,
        )
        self._model = Omni(
            model=self._model_path,
            **options,
        )
        if self._is_request_started is not None:
            engine = self._model.engine
            if (
                engine.num_stages != 1
                or engine.get_stage_metadata(0).stage_type != "diffusion"
            ):
                self.stop()
                raise ValueError(
                    "vLLM-Omni replica autoscaling requires a single diffusion stage"
                )

    def stop(self):
        self._closed = True
        self._fail_all_waiters(RuntimeError("model is being stopped"))
        if self._model is not None:
            try:
                self._model.close()
            except Exception:
                logger.warning(
                    "Failed to shutdown vLLM-Omni diffusion model", exc_info=True
                )
            self._model = None

    # --- concurrent request dispatch -------------------------------------

    def _concurrency_available(self) -> bool:
        engine = getattr(self._model, "engine", None)
        return (
            engine is not None
            and hasattr(engine, "add_request")
            and hasattr(engine, "try_get_output")
        )

    def _fail_all_waiters(self, error: BaseException) -> None:
        with self._submit_lock:
            waiters = list(self._waiters.values())
            self._waiters.clear()
        for waiter in waiters:
            if not waiter.future.done():
                waiter.future.set_exception(error)

    def _fail_and_close(self, error: BaseException) -> None:
        """Terminal transition for fatal engine errors.

        Atomically marks the model closed and drains the waiters, so both
        in-flight and subsequent requests fail immediately instead of being
        submitted to a dead backend, then shuts the engine down.
        """
        with self._submit_lock:
            self._closed = True
            if self._fatal_error is None:
                self._fatal_error = error
            self._dispatcher_thread = None
            waiters = list(self._waiters.values())
            self._waiters.clear()
            model, self._model = self._model, None
        for waiter in waiters:
            if not waiter.future.done():
                waiter.future.set_exception(error)
        if model is not None:
            try:
                model.close()
            except Exception:
                logger.warning(
                    "Failed to shutdown vLLM-Omni diffusion model after a "
                    "fatal error",
                    exc_info=True,
                )

    def _raise_if_unavailable(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(
                "vLLM-Omni engine is unavailable after a fatal error: "
                f"{self._fatal_error}"
            ) from self._fatal_error
        if self._closed:
            raise RuntimeError("model is stopped")

    def _ensure_dispatcher(self) -> None:
        # caller must hold _submit_lock
        if self._dispatcher_thread is None or not self._dispatcher_thread.is_alive():
            thread = threading.Thread(
                target=self._dispatch_outputs,
                name=f"vllm-omni-dispatch-{self._model_uid}",
                daemon=True,
            )
            self._dispatcher_thread = thread
            thread.start()

    def _dispatch_outputs(self) -> None:
        """Single consumer of the shared engine output queue.

        The queue interleaves messages of all in-flight requests, so exactly
        one thread drains it and routes each message to the waiter registered
        for its request_id. Exits when no request is in flight; the next
        submission restarts it.
        """
        while True:
            with self._submit_lock:
                if self._closed or not self._waiters or self._model is None:
                    self._dispatcher_thread = None
                    return
                model = self._model
            try:
                msg = model.engine.try_get_output(timeout=0.5)
            except Exception as e:
                logger.error(
                    "vLLM-Omni engine output loop failed, failing %d in-flight "
                    "request(s) and closing the model",
                    len(self._waiters),
                    exc_info=True,
                )
                self._fail_and_close(e)
                return
            if msg is None:
                continue
            self._route_message(msg)

    def _route_message(self, msg: Any) -> None:
        from vllm_omni.engine.messages import ErrorMessage, OutputMessage

        if isinstance(msg, ErrorMessage):
            error = RuntimeError(f"vLLM-Omni engine error: {msg.error}")
            request_id = getattr(msg, "request_id", None)
            if getattr(msg, "fatal", False) or (
                request_id is None and self._is_request_started is not None
            ):
                # A replica reporting load must retain ownership or stop the
                # engine if an error cannot be attributed to a request.
                self._fail_and_close(error)
                return
            if request_id is None:
                # unroutable error: fail everyone in flight, but the engine
                # itself may still be usable
                self._fail_all_waiters(error)
                return
            with self._submit_lock:
                waiter = self._waiters.pop(request_id, None)
            if waiter is not None and not waiter.future.done():
                waiter.future.set_exception(error)
            return

        if not isinstance(msg, OutputMessage):
            # e.g. StageMetricsMessage — nothing to route
            return

        engine_outputs = msg.engine_outputs
        with self._submit_lock:
            waiter = self._waiters.get(msg.request_id)
            if waiter is None:
                return
            if self._is_request_started is not None and self._is_request_started(
                engine_outputs
            ):
                waiter.queued_since = None
                return
        error_text = getattr(engine_outputs, "error", None)
        if error_text is not None:
            with self._submit_lock:
                self._waiters.pop(msg.request_id, None)
            if not waiter.future.done():
                waiter.future.set_exception(
                    RuntimeError(f"vLLM-Omni generation failed: {error_text}")
                )
            return
        waiter.outputs.append(engine_outputs)
        if msg.finished:
            with self._submit_lock:
                self._waiters.pop(msg.request_id, None)
            if not waiter.future.done():
                waiter.future.set_result(list(waiter.outputs))

    def _resolve_stage_args(self, prompt_payload: Any, sampling_params: Any) -> Any:
        """Mirror the per-request preparation Omni.generate applies before
        engine.add_request, degrading gracefully across vllm-omni versions."""
        omni = self._model
        sp_list = [sampling_params]
        resolve = getattr(omni, "resolve_sampling_params_list", None)
        if callable(resolve):
            sp_list = resolve(sp_list)
        force_final = getattr(omni, "_maybe_force_final_only_for_llm_stages", None)
        if callable(force_final):
            sp_list = force_final(sp_list)
        modalities = (
            prompt_payload.get("modalities")
            if isinstance(prompt_payload, dict)
            else None
        )
        final_stage_id = 0
        compute_final = getattr(omni, "_compute_final_stage_id", None)
        if callable(compute_final):
            final_stage_id = compute_final(modalities)
        final_output_stage_ids = None
        compute_outputs = getattr(omni, "_compute_final_output_stage_ids", None)
        if callable(compute_outputs):
            final_output_stage_ids = compute_outputs(modalities)
        if not final_output_stage_ids:
            final_output_stage_ids = [final_stage_id]
        return sp_list, final_stage_id, final_output_stage_ids

    async def _submit_and_wait(
        self, prompt_payload: Any, sampling_params: Any
    ) -> List[Any]:
        (
            sp_list,
            final_stage_id,
            final_output_stage_ids,
        ) = self._resolve_stage_args(prompt_payload, sampling_params)
        report_load = self._is_request_started is not None
        if report_load:
            # Enable the event after resolving pipeline sampling constraints.
            sp_list = [dataclasses.replace(sp_list[0], emit_request_lifecycle=True)]
        request_id = f"xinf-{uuid.uuid4()}"
        waiter = _RequestWaiter(time.monotonic() if report_load else None)
        with self._submit_lock:
            self._raise_if_unavailable()
            self._waiters[request_id] = waiter
            self._ensure_dispatcher()
        try:
            self._model.engine.add_request(  # type: ignore
                request_id=request_id,
                prompt=prompt_payload,
                sampling_params_list=sp_list,
                final_stage_id=final_stage_id,
                final_output_stage_ids=final_output_stage_ids,
            )
        except Exception:
            with self._submit_lock:
                self._waiters.pop(request_id, None)
            raise
        return await asyncio.wrap_future(waiter.future)

    def _generate_serial(self, prompt_payload: Any, sampling_params: Any) -> Any:
        # Omni.generate is unsafe for concurrent callers and the actor no
        # longer serializes calls (allow_batch=True), so guard it here.
        with self._generate_lock:
            return self._model.generate(  # type: ignore
                prompt_payload,
                sampling_params_list=[sampling_params],
            )

    def _build_sampling_params(
        self,
        n: int,
        width: int,
        height: int,
        generate_config: Dict[str, Any],
    ) -> Any:
        resolve_request_lightning(generate_config.pop("use_lightning", None), False)
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        seed = generate_config.pop("seed", None)
        if seed is None or seed == -1:
            # keep the default xinference behavior of generating a new image
            # every call instead of a fixed default seed
            seed = random.randint(0, 2**31 - 1)
        params = dict(generate_config)
        params.update(
            width=width,
            height=height,
            num_outputs_per_prompt=n,
            seed=seed,
        )
        params = _filter_kwargs_by_dataclass_fields(
            params, OmniDiffusionSamplingParams, "sampling params"
        )
        return OmniDiffusionSamplingParams(**params)

    def _build_prompt(self, prompt: str, width: int, height: int) -> Any:
        # build_text_to_image_prompt adds model-specific prompt structure
        # (e.g. system prompts); fall back to the raw prompt when the running
        # vllm-omni version does not expose these helpers
        try:
            from vllm_omni.model_extras import (
                build_text_to_image_prompt,
                get_model_class_name,
            )
        except ImportError:
            return prompt
        try:
            model_class_name = get_model_class_name(self._model)
            return build_text_to_image_prompt(
                model_class_name=model_class_name,
                prompt=prompt,
                negative_prompt=None,
                height=height,
                width=width,
            )
        except Exception:
            logger.warning(
                "Failed to build vLLM-Omni prompt payload, using raw prompt",
                exc_info=True,
            )
            return prompt

    @staticmethod
    def _extract_images(outputs: Any) -> List[Any]:
        from PIL import Image

        if not outputs:
            raise RuntimeError(
                "vLLM-Omni image generation failed, see server logs for details"
            )
        images = []
        for output in outputs:
            frames = getattr(output, "images", None)
            if not frames:
                request_output = getattr(output, "request_output", None)
                frames = (
                    getattr(request_output, "images", None)
                    if request_output is not None
                    else None
                )
            if not frames:
                continue
            if not isinstance(frames, (list, tuple)):
                frames = [frames]
            for frame in frames:
                if isinstance(frame, Image.Image):
                    images.append(frame)
                else:
                    raise RuntimeError(
                        f"Unexpected image type from vLLM-Omni: {type(frame)}"
                    )
        if not images:
            raise RuntimeError(
                "vLLM-Omni image generation returned no images, "
                "see server logs for details"
            )
        return images

    async def text_to_image(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024*1024",
        response_format: str = "url",
        **kwargs,
    ):
        self._raise_if_unavailable()
        assert self._model is not None
        width, height = map(int, re.split(r"[^\d]+", size))
        generate_config: Dict[str, Any] = (
            self._model_spec.default_generate_config or {}  # type: ignore
        ).copy()
        generate_config.update({k: v for k, v in kwargs.items() if v is not None})
        single_output = self._model_spec.model_name == "MiniMax-H3-Image"
        if single_output:
            generate_config.setdefault("num_inference_steps", 6)
            generate_config.setdefault("guidance_scale", 1.0)
        seeds = resolve_image_seed_list(generate_config.get("seed"), n)
        prompt_payload = self._build_prompt(prompt, width, height)
        images = []
        configs = (
            [{**generate_config, "seed": seed} for seed in seeds]
            if seeds is not None
            else [generate_config]
        )
        outputs_per_config = 1 if seeds is not None else n
        if single_output and seeds is None and n > 1:
            seed = generate_config.get("seed")
            configs = [generate_config.copy() for _ in range(n)]
            if isinstance(seed, int) and seed >= 0:
                for index, config in enumerate(configs):
                    config["seed"] = (seed + index) % (MAX_IMAGE_SEED + 1)
            outputs_per_config = 1
        for config in configs:
            sampling_params = self._build_sampling_params(
                outputs_per_config, width, height, config
            )
            if self._concurrency_available():
                outputs = await self._submit_and_wait(prompt_payload, sampling_params)
            else:
                outputs = await asyncio.to_thread(
                    self._generate_serial, prompt_payload, sampling_params
                )
            images.extend(self._extract_images(outputs))
        return handle_image_result(response_format, images)
