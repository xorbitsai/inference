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

"""SD WebUI request orchestration on top of the community image pipelines."""

import asyncio
import inspect
import math
import time
from contextlib import nullcontext

from PIL import ImageOps


class SDAPIToDiffusersConverter:
    txt2img_identical_args = {
        "prompt",
        "negative_prompt",
        "seed",
        "subseed",
        "subseed_strength",
        "seed_resize_from_h",
        "seed_resize_from_w",
        "width",
        "height",
        "sampler_name",
        "scheduler",
        "enable_hr",
        "hr_scale",
        "hr_upscaler",
        "hr_second_pass_steps",
        "progressor",
        "request_id",
    }
    txt2img_arg_mapping = {
        "steps": "num_inference_steps",
        "cfg_scale": "guidance_scale",
        "denoising_strength": "strength",
    }
    img2img_identical_args = {
        "prompt",
        "negative_prompt",
        "seed",
        "subseed",
        "subseed_strength",
        "seed_resize_from_h",
        "seed_resize_from_w",
        "width",
        "height",
        "sampler_name",
        "scheduler",
        "resize_mode",
        "mask_blur",
        "inpainting_mask_invert",
        "progressor",
        "request_id",
    }
    img2img_arg_mapping = {
        "init_images": "image",
        "mask": "mask_image",
        "steps": "num_inference_steps",
        "cfg_scale": "guidance_scale",
        "denoising_strength": "strength",
        "inpaint_full_res_padding": "padding_mask_crop",
    }

    @staticmethod
    def convert_to_diffusers(sd_type: str, params: dict) -> dict:
        diffusers_params = {}

        identical_args = getattr(SDAPIToDiffusersConverter, f"{sd_type}_identical_args")
        mapping_args = getattr(SDAPIToDiffusersConverter, f"{sd_type}_arg_mapping")
        for param, value in params.items():
            if param in identical_args:
                diffusers_params[param] = value
            elif param in mapping_args:
                diffusers_params[mapping_args[param]] = value
            else:
                raise ValueError(f"Unknown arg: {param}")

        return diffusers_params

    @staticmethod
    def get_available_args(sd_type: str) -> set:
        identical_args = getattr(SDAPIToDiffusersConverter, f"{sd_type}_identical_args")
        mapping_args = getattr(SDAPIToDiffusersConverter, f"{sd_type}_arg_mapping")
        return identical_args.union(mapping_args)


class _NoProgress(nullcontext):
    request_id = None

    def split_stages(self, *args, **kwargs):
        pass

    def set_progress(self, *args, **kwargs):
        pass


class SDAPIDiffusionModelMixin:
    @staticmethod
    def _decode_b64_img(value):
        from .utils import decode_base64_to_image

        return decode_base64_to_image(value)

    @staticmethod
    def _check_kwargs(sd_type: str, kwargs: dict):
        available = SDAPIToDiffusersConverter.get_available_args(sd_type)
        converted = SDAPIToDiffusersConverter.convert_to_diffusers(
            sd_type,
            {k: v for k, v in kwargs.items() if k in available and v is not None},
        )
        if sd_type == "img2img":
            # Diffusers enables mask cropping even when padding is zero.
            if kwargs.get("mask") and kwargs.get("inpaint_full_res"):
                converted["padding_mask_crop"] = kwargs.get(
                    "inpaint_full_res_padding", 0
                )
            else:
                converted.pop("padding_mask_crop", None)
        width = converted.pop("width", 512)
        height = converted.pop("height", 512)
        if min(width, height) < 8:
            raise ValueError("Image dimensions must be at least 8 pixels")
        converted["size"] = f"{width // 8 * 8}*{height // 8 * 8}"
        scheduler = (converted.get("scheduler") or "automatic").lower()
        if scheduler == "automatic":
            scheduler = {
                "DPM++ 2M": "karras",
                "DPM++ SDE": "karras",
                "DPM++ 2M SDE": "exponential",
                "DPM2": "karras",
                "DPM2 a": "karras",
            }.get(converted.get("sampler_name") or "", "automatic")
        converted["scheduler"] = scheduler
        return converted

    @staticmethod
    async def _sdapi_call(fn, **kwargs):
        import threading

        cancelled = kwargs["_cancel_event"] = threading.Event()
        operation = (
            fn(**kwargs)
            if inspect.iscoroutinefunction(fn)
            else asyncio.to_thread(fn, **kwargs)
        )
        task = asyncio.create_task(operation)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # Keep the actor's serialization lock until the pipeline has stopped.
            try:
                await task
            except Exception:
                pass
            raise

    async def txt2img(self, **kwargs):
        return await self._sdapi_generate("txt2img", kwargs)

    async def img2img(self, **kwargs):
        return await self._sdapi_generate("img2img", kwargs)

    async def _basic_sdapi_generate(self, sd_type, params):
        # Preserve SDAPI access to other community image engines, including MLX.
        if (
            params.get("enable_hr")
            or params.get("alwayson_scripts")
            or params.get("subseed_strength")
        ):
            raise ValueError(
                "SD WebUI extensions require a Stable Diffusion or SDXL model"
            )
        kwargs = self._check_kwargs(sd_type, params)
        for key in (
            "subseed",
            "subseed_strength",
            "seed_resize_from_h",
            "seed_resize_from_w",
            "enable_hr",
            "hr_scale",
            "hr_upscaler",
            "hr_second_pass_steps",
        ):
            kwargs.pop(key, None)
        kwargs["progressor"] = params.get("progressor") or _NoProgress()
        if sd_type == "txt2img":
            method = self.text_to_image
        else:
            images = [
                self._decode_b64_img(value) for value in params.get("init_images", [])
            ]
            if not images:
                raise ValueError("init_images must contain at least one image")
            kwargs["image"] = images[0] if len(images) == 1 else images
            if params.get("mask"):
                kwargs["mask_image"] = self._decode_b64_img(params["mask"]).convert("L")
                if params.get("inpainting_mask_invert"):
                    kwargs["mask_image"] = ImageOps.invert(kwargs["mask_image"])
                method = self.inpainting
            else:
                method = self.image_to_image
        result = await self._sdapi_call(method, response_format="b64_json", **kwargs)
        return {
            "images": [item["b64_json"] for item in result["data"]],
            "info": {"created": result["created"]},
            "parameters": {},
        }

    async def _sdapi_generate(self, sd_type, params):
        if getattr(self._model_spec, "model_base", None) not in (
            "SD 1.5",
            "SD 2.0",
            "SD 2.1",
            "SDXL",
        ):
            return await self._basic_sdapi_generate(sd_type, params)

        from .lora import process_loras
        from .prompt_converter import gen_prompt_embeds
        from .rng import create_generator
        from .utils import (
            create_binary_mask,
            encode_pil_to_base64,
            get_fixed_seed,
            resize_image,
        )

        defaults = dict(
            prompt="",
            negative_prompt="",
            width=512,
            height=512,
            seed=-1,
            subseed=-1,
            subseed_strength=0.0,
            seed_resize_from_h=0,
            seed_resize_from_w=0,
            batch_size=1,
            n_iter=1,
            cfg_scale=7.0,
            enable_hr=False,
            hr_scale=2.0,
            hr_upscaler="Latent",
            hr_second_pass_steps=0,
            resize_mode=0,
            mask_blur=0,
            inpainting_mask_invert=0,
        )
        defaults.update({k: v for k, v in params.items() if v is not None})
        params = defaults
        batch_size, n_iter = params["batch_size"], params["n_iter"]
        if batch_size < 1 or n_iter < 1:
            raise ValueError("batch_size and n_iter must be positive")
        count = batch_size * n_iter
        seed, subseed = get_fixed_seed(params["seed"]), get_fixed_seed(
            params["subseed"]
        )
        seeds = [
            seed + (i if params["subseed_strength"] == 0 else 0) for i in range(count)
        ]
        subseeds = [subseed + i for i in range(count)]
        settings = params.get("override_settings") or {}
        scripts = params.get("alwayson_scripts") or {}
        converted = self._check_kwargs(sd_type, params)
        for key in ("enable_hr", "hr_scale", "hr_upscaler", "hr_second_pass_steps"):
            converted.pop(key, None)
        converted["num_inference_steps"] = params.get("steps") or (
            getattr(self._model_spec, "default_generate_config", None) or {}
        ).get("num_inference_steps", 20)
        converted["strength"] = params.get("denoising_strength", 0.75)
        if not 0 < converted["strength"] <= 1:
            raise ValueError("denoising_strength must be greater than 0 and at most 1")
        converted["clip_skip"] = settings.get("clip_skip", 1)
        converted["gen_prompt_embeds"] = gen_prompt_embeds
        lora_specs = params.pop("_sdapi_lora_specs", None)
        if lora_specs is not None:
            from .custom import CustomImageModelFamilyV2

            lora_specs = [
                CustomImageModelFamilyV2.parse_obj(spec) for spec in lora_specs
            ]
        process_loras(converted, strict=settings.get("strict", True), specs=lora_specs)
        # An empty adapter list resets adapters enabled by the preceding SDAPI request.
        converted.setdefault("loras", [])
        progressor = converted["progressor"] = params.get("progressor") or _NoProgress()
        progressor.split_stages(n_iter)
        width, height = map(int, converted["size"].split("*"))
        mask = None
        if sd_type == "img2img":
            inputs = params.get("init_images") or []
            if not inputs:
                raise ValueError("init_images must contain at least one image")
            inputs = [self._decode_b64_img(value) for value in inputs]
            if len(inputs) not in (1, batch_size):
                raise ValueError(
                    "init_images must contain one image or batch_size images"
                )
            inputs = [
                resize_image(params["resize_mode"], image, width, height)
                for image in inputs
            ]
            converted["image"] = inputs[0] if len(inputs) == 1 else inputs
            if params.get("mask"):
                mask = create_binary_mask(self._decode_b64_img(params["mask"]))
                if params["inpainting_mask_invert"]:
                    mask = ImageOps.invert(mask)
                converted["mask_image"] = resize_image(
                    params["resize_mode"], mask, width, height
                )
            converted["num_inference_steps"] = math.ceil(
                converted["num_inference_steps"] / converted["strength"]
            )
        has_adetailer = bool(
            ((scripts.get("ADetailer") or {}).get("args") or [False])[0]
        )
        results = []
        for iteration in range(n_iter):
            with progressor:
                batch = converted.copy()
                batch["seed"] = seeds[
                    iteration * batch_size : (iteration + 1) * batch_size
                ]
                batch["subseed"] = subseeds[
                    iteration * batch_size : (iteration + 1) * batch_size
                ]
                batch["generator"] = create_generator(
                    batch["seed"][0], getattr(self, "_device", None)
                )
                batch["n"] = batch_size
                hires = sd_type == "txt2img" and params["enable_hr"]
                progressor.split_stages(1 + int(hires) + int(has_adetailer))
                first = batch.copy()
                if scripts.get("ControlNet") or scripts.get("controlnet"):
                    from .controlnet import (
                        generate_controlnet_kwargs_for_image2image,
                        generate_controlnet_kwargs_for_text2image,
                    )

                    convert = (
                        generate_controlnet_kwargs_for_text2image
                        if sd_type == "txt2img"
                        else generate_controlnet_kwargs_for_image2image
                    )
                    await asyncio.to_thread(convert, self._model_spec, first, scripts)
                if hires:
                    from .upscaler import HiResUpscaler, upscaler_preprocess

                    hr_upscaler = HiResUpscaler(params["hr_upscaler"])
                    upscaler_preprocess(hr_upscaler, first)
                method = (
                    self.text_to_image
                    if sd_type == "txt2img"
                    else (self.inpainting if mask is not None else self.image_to_image)
                )
                with progressor:
                    images = await self._sdapi_call(
                        method, _return_images=True, **first
                    )
                if hires:
                    from .upscaler import upscale

                    scale = params["hr_scale"]
                    if scale < 1:
                        raise ValueError("hr_scale must be at least 1")
                    dest = (int(width * scale) // 8 * 8, int(height * scale) // 8 * 8)
                    with progressor:
                        if hr_upscaler == HiResUpscaler.none:
                            from ._compat import LANCZOS

                            images = [image.resize(dest, LANCZOS) for image in images]
                        else:
                            images = await asyncio.to_thread(
                                upscale,
                                hr_upscaler,
                                images,
                                scale,
                                dest,
                                downsample_factor=getattr(
                                    getattr(self, "_model", None), "vae_scale_factor", 8
                                ),
                            )
                        second = batch.copy()
                        second["size"] = f"{dest[0]}*{dest[1]}"
                        second["image"] = images
                        second["num_inference_steps"] = math.ceil(
                            (
                                params["hr_second_pass_steps"]
                                or batch["num_inference_steps"]
                            )
                            / batch["strength"]
                        )
                        if scripts.get("ControlNet") or scripts.get("controlnet"):
                            from .controlnet import (
                                generate_controlnet_kwargs_for_image2image,
                            )

                            second["_sdapi_hires"] = True
                            await asyncio.to_thread(
                                generate_controlnet_kwargs_for_image2image,
                                self._model_spec,
                                second,
                                scripts,
                            )
                        images = await self._sdapi_call(
                            self.image_to_image, _return_images=True, **second
                        )
                if has_adetailer:
                    from .adetailer import process_adetailer

                    with progressor:
                        progressor.split_stages(len(images))
                        enhanced = []
                        for index, image in enumerate(images):
                            detail = batch.copy()
                            detail["num_inference_steps"] = params.get("steps") or 20
                            detail.update(
                                seed=[batch["seed"][index]],
                                subseed=[batch["subseed"][index]],
                                size=f"{image.width}*{image.height}",
                            )
                            with progressor:
                                enhanced.append(
                                    await self._sdapi_call(
                                        process_adetailer,
                                        model=self,
                                        sd_type=sd_type,
                                        image=image,
                                        kwargs=detail,
                                        alwayson_scripts=scripts,
                                    )
                                )
                        images = enhanced
                results.extend(encode_pil_to_base64(image) for image in images)
        return {
            "images": results,
            "parameters": {k: v for k, v in params.items() if k != "progressor"},
            "info": {
                "created": int(time.time()),
                "seed": seeds[0],
                "all_seeds": seeds,
                "subseed": subseeds[0],
                "all_subseeds": subseeds,
                "subseed_strength": params["subseed_strength"],
                "width": width,
                "height": height,
                "batch_size": batch_size,
                "prompt": params["prompt"],
                "negative_prompt": params["negative_prompt"],
                "all_prompts": [params["prompt"]] * count,
                "all_negative_prompts": [params["negative_prompt"]] * count,
            },
        }

    def controlnet_detect(self, *args, **kwargs):
        from .controlnet import detect

        return detect(*args, **kwargs)

    def controlnet_model_list(self):
        from .controlnet import list_models

        return list_models(self._model_spec)  # type: ignore

    @staticmethod
    def controlnet_module_list():
        from .controlnet import list_modules

        return list_modules()

    def controlnet_control_types(self):
        from .controlnet import control_types

        return control_types(self._model_spec)  # type: ignore
