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
import base64
import io
import warnings

from PIL import Image


class SDAPIToDiffusersConverter:
    txt2img_identical_args = {
        "prompt",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "sampler_name",
    }
    txt2img_arg_mapping = {
        "steps": "num_inference_steps",
        "cfg_scale": "guidance_scale",
        # "denoising_strength": "strength",
    }
    img2img_identical_args = {
        "prompt",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "sampler_name",
    }
    img2img_arg_mapping = {
        "init_images": "image",
        "steps": "num_inference_steps",
        "cfg_scale": "guidance_scale",
        "denoising_strength": "strength",
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


class SDAPIDiffusionModelMixin:
    @staticmethod
    def _check_kwargs(sd_type: str, kwargs: dict):
        available_args = SDAPIToDiffusersConverter.get_available_args(sd_type)
        unknown_args = []
        available_kwargs = {}
        for arg, value in kwargs.items():
            if arg in available_args:
                available_kwargs[arg] = value
            else:
                unknown_args.append(arg)
        if unknown_args:
            warnings.warn(
                f"Some args are not supported for now and will be ignored: {unknown_args}"
            )

        converted_kwargs = SDAPIToDiffusersConverter.convert_to_diffusers(
            sd_type, available_kwargs
        )

        width, height = converted_kwargs.pop("width", None), converted_kwargs.pop(
            "height", None
        )
        if width and height:
            converted_kwargs["size"] = f"{width}*{height}"

        return converted_kwargs

    def txt2img(self, **kwargs):
        converted_kwargs = self._check_kwargs("txt2img", kwargs)
        result = self.text_to_image(response_format="b64_json", **converted_kwargs)  # type: ignore

        # convert to SD API result
        return {
            "images": [r["b64_json"] for r in result["data"]],
            "info": {"created": result["created"]},
            "parameters": {},
        }

    @staticmethod
    def _decode_b64_img(img_str: str) -> Image:
        # img_str in a format: "data:image/png;base64," + raw_b64_img(image)
        f, data = img_str.split(",", 1)
        f, encode_type = f.split(";", 1)
        assert encode_type == "base64"
        f = f.split("/", 1)[1]
        b = base64.b64decode(data)
        return Image.open(io.BytesIO(b), formats=[f])

    def img2img(self, **kwargs):
        init_images = kwargs.pop("init_images", [])
        kwargs["init_images"] = [self._decode_b64_img(i) for i in init_images]
        clip_skip = kwargs.get("override_settings", {}).get("clip_skip")
        converted_kwargs = self._check_kwargs("img2img", kwargs)
        if clip_skip:
            converted_kwargs["clip_skip"] = clip_skip
        result = self.image_to_image(response_format="b64_json", **converted_kwargs)  # type: ignore

        # convert to SD API result
        return {
            "images": [r["b64_json"] for r in result["data"]],
            "info": {"created": result["created"]},
            "parameters": {},
        }
