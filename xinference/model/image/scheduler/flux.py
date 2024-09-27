# Copyright 2022-2024 XProbe Inc.
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
import logging
import re
import typing
from collections import deque
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
import xoscar as xo

from ..utils import handle_image_result

if TYPE_CHECKING:
    from ..stable_diffusion.core import DiffusionModel


logger = logging.getLogger(__name__)


class Text2ImageRequest:
    def __init__(
        self,
        future,
        prompt: str,
        n: int,
        size: str,
        response_format: str,
        *args,
        **kwargs,
    ):
        self.future = future
        self._prompt = prompt
        self._n = n
        self._size = size
        self._response_format = response_format
        self._args = args
        self._kwargs = kwargs
        self._width = -1
        self._height = -1
        self._generate_kwargs: Dict[str, Any] = {}
        self._set_width_and_height()
        self.is_encode = True
        self.done_steps = 0
        self.total_steps = 0
        self.static_tensors: Dict[str, torch.Tensor] = {}
        self.timesteps = None
        self.dtype = None
        self.output = None

    def _set_width_and_height(self):
        self._width, self._height = map(int, re.split(r"[^\d]+", self._size))

    def set_generate_kwargs(self, generate_kwargs: Dict):
        self._generate_kwargs = {k: v for k, v in generate_kwargs.items()}

    @property
    def prompt(self):
        return self._prompt

    @property
    def n(self):
        return self._n

    @property
    def size(self):
        return self._size

    @property
    def response_format(self):
        return self._response_format

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def generate_kwargs(self):
        return self._generate_kwargs


class FluxBatchSchedulerActor(xo.StatelessActor):
    @classmethod
    def gen_uid(cls, model_uid: str):
        return f"{model_uid}-scheduler-actor"

    def __init__(self):
        from ....device_utils import get_available_device

        super().__init__()
        self._waiting_queue: deque[Text2ImageRequest] = deque()  # type: ignore
        self._running_queue: deque[Text2ImageRequest] = deque()  # type: ignore
        self._model = None
        self._available_device = get_available_device()

    def set_model(self, model):
        """
        Must use `set_model`. Otherwise, the model will be copied once.
        """
        self._model = model

    async def __post_create__(self):
        from ....isolation import Isolation

        self._isolation = Isolation(
            asyncio.new_event_loop(), threaded=True, daemon=True
        )
        self._isolation.start()
        asyncio.run_coroutine_threadsafe(self.run(), loop=self._isolation.loop)

    async def add_request(self, future, *args, **kwargs):
        req = Text2ImageRequest(future, *args, **kwargs)
        self._waiting_queue.append(req)

    def _handle_request(self) -> Optional[List[Text2ImageRequest]]:
        if self._model is None:
            return None
        # TODO: consider each request's n
        max_num_seqs = self._model.get_max_num_seqs_for_batching()
        # currently, FCFS strategy
        running_list: List[Text2ImageRequest] = []
        while len(self._running_queue) > 0:
            if len(running_list) == max_num_seqs:
                break
            req = self._running_queue.popleft()
            running_list.append(req)

        waiting_list: List[Text2ImageRequest] = []
        if len(running_list) < max_num_seqs:
            while len(self._waiting_queue) > 0:
                req = self._waiting_queue.popleft()
                waiting_list.append(req)
                if len(running_list) + len(waiting_list) == max_num_seqs:
                    break
        return waiting_list + running_list

    @staticmethod
    def _empty_cache():
        from ....device_utils import empty_cache

        empty_cache()

    async def step(self):
        req_list = self._handle_request()
        if not req_list:
            return
        _batch_text_to_image(self._model, req_list, self._available_device)
        # handle results
        for r in req_list:
            if r.output is not None:
                r.future.set_result(
                    handle_image_result(r.response_format, r.output.images)
                )
            else:
                self._running_queue.append(r)
        self._empty_cache()

    async def run(self):
        try:
            while True:
                # wait 10ms
                await asyncio.sleep(0.01)
                await self.step()
        except Exception as e:
            logger.exception(
                f"Scheduler actor uid: {self.uid}, address: {self.address} run with error: {e}"
            )


def _cat_tensors(infos: List[Dict]) -> Dict:
    keys = infos[0].keys()
    res = {}
    for k in keys:
        tmp = [info[k] for info in infos]
        res[k] = torch.cat(tmp)
    return res


@typing.no_type_check
@torch.inference_mode()
def _batch_text_to_image(
    model_cls: "DiffusionModel",
    req_list: List[Text2ImageRequest],
    available_device: str,
):
    from diffusers.pipelines.flux.pipeline_flux import (
        calculate_shift,
        retrieve_timesteps,
    )
    from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

    device = model_cls._model._execution_device
    height, width = req_list[0].height, req_list[0].width
    for r in req_list:
        if r.is_encode:
            generate_kwargs = model_cls._model_spec.default_generate_config.copy()
            generate_kwargs.update({k: v for k, v in r.kwargs.items() if v is not None})
            model_cls._filter_kwargs(model_cls._model, generate_kwargs)
            r.set_generate_kwargs(generate_kwargs)

            num_images_per_prompt = r.n
            callback_on_step_end_tensor_inputs = r.generate_kwargs.get(
                "callback_on_step_end_tensor_inputs", ["latents"]
            )
            max_sequence_length = r.generate_kwargs.get("max_sequence_length", 512)
            num_inference_steps = r.generate_kwargs.get("num_inference_steps", 28)
            guidance_scale = r.generate_kwargs.get("guidance_scale", 7.0)
            generator = None
            seed = r.generate_kwargs.get("seed", None)
            if seed is not None:
                generator = torch.Generator(device=available_device)  # type: ignore
                if seed != -1:
                    generator = generator.manual_seed(seed)
            latents = None
            timesteps = None

            # check inputs
            model_cls._model.check_inputs(
                r.prompt,
                None,
                height,
                width,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

            # handle prompt
            (
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
            ) = model_cls._model.encode_prompt(
                prompt=r.prompt,
                prompt_2=None,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=None,
            )

            # Prepare latent variables
            num_channels_latents = model_cls._model.transformer.config.in_channels // 4
            latents, latent_image_ids = model_cls._model.prepare_latents(
                num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

            # Prepare timesteps
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = latents.shape[1]
            mu = calculate_shift(
                image_seq_len,
                model_cls._model.scheduler.config.base_image_seq_len,
                model_cls._model.scheduler.config.max_image_seq_len,
                model_cls._model.scheduler.config.base_shift,
                model_cls._model.scheduler.config.max_shift,
            )
            timesteps, num_inference_steps = retrieve_timesteps(
                model_cls._model.scheduler,
                num_inference_steps,
                device,
                timesteps,
                sigmas,
                mu=mu,
            )

            # handle guidance
            if model_cls._model.transformer.config.guidance_embeds:
                guidance = torch.full(
                    [1], guidance_scale, device=device, dtype=torch.float32
                )
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None

            r.static_tensors["latents"] = latents
            r.static_tensors["guidance"] = guidance
            r.static_tensors["pooled_prompt_embeds"] = pooled_prompt_embeds
            r.static_tensors["prompt_embeds"] = prompt_embeds
            r.static_tensors["text_ids"] = text_ids
            r.static_tensors["latent_image_ids"] = latent_image_ids
            r.timesteps = timesteps
            r.dtype = latents.dtype
            r.total_steps = len(timesteps)
            r.is_encode = False

    static_tensors = _cat_tensors([r.static_tensors for r in req_list])

    # Do a step
    timestep_tmp = []
    for r in req_list:
        timestep_tmp.append(r.timesteps[r.done_steps].expand(r.n).to(r.dtype))
        r.done_steps += 1
    timestep = torch.cat(timestep_tmp)
    noise_pred = model_cls._model.transformer(
        hidden_states=static_tensors["latents"],
        # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
        timestep=timestep / 1000,
        guidance=static_tensors["guidance"],
        pooled_projections=static_tensors["pooled_prompt_embeds"],
        encoder_hidden_states=static_tensors["prompt_embeds"],
        txt_ids=static_tensors["text_ids"],
        img_ids=static_tensors["latent_image_ids"],
        joint_attention_kwargs=None,
        return_dict=False,
    )[0]
    # update latents
    start_idx = 0
    for r in req_list:
        n = r.n
        # handle diffusion scheduler step
        with model_cls._reset_when_done(
            model_cls._model, r.generate_kwargs.get("sampler_name", None)
        ):
            _noise_pred = noise_pred[start_idx : start_idx + n, ::]
            _timestep = timestep[start_idx]
            latents_out = model_cls._model.scheduler.step(
                _noise_pred, _timestep, r.static_tensors["latents"], return_dict=False
            )[0]
            r.static_tensors["latents"] = latents_out
        start_idx += n
        print(f"=====Done one step")

        # process result
        if r.done_steps == r.total_steps:
            output_type = r.generate_kwargs.get("output_type", "pil")
            _latents = r.static_tensors["latents"]
            if output_type == "latent":
                image = _latents
            else:
                _latents = model_cls._model._unpack_latents(
                    _latents, height, width, model_cls._model.vae_scale_factor
                )
                _latents = (
                    _latents / model_cls._model.vae.config.scaling_factor
                ) + model_cls._model.vae.config.shift_factor
                image = model_cls._model.vae.decode(_latents, return_dict=False)[0]
                image = model_cls._model.image_processor.postprocess(
                    image, output_type=output_type
                )

            is_padded = r.generate_kwargs.get("is_padded", None)
            origin_size = r.generate_kwargs.get("origin_size", None)

            if is_padded and origin_size:
                new_images = []
                x, y = origin_size
                for img in image:
                    new_images.append(img.crop((0, 0, x, y)))
                image = new_images

            r.output = FluxPipelineOutput(images=image)
