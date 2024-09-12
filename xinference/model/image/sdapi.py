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

import warnings


class SDAPIToDiffusersConverter:
    txt2img_identical_args = [
        "prompt",
        "negative_prompt",
        "seed",
        "width",
        "height",
        "sampler_name",
    ]
    txt2img_arg_mapping = {
        "steps": "num_inference_steps",
        "cfg_scale": "guidance_scale",
    }

    @staticmethod
    def convert_txt2img_to_diffusers(params: dict) -> dict:
        diffusers_params = {}

        identical_args = set(SDAPIToDiffusersConverter.txt2img_identical_args)
        mapping_args = SDAPIToDiffusersConverter.txt2img_arg_mapping
        for param, value in params.items():
            if param in identical_args:
                diffusers_params[param] = value
            elif param in mapping_args:
                diffusers_params[mapping_args[param]] = value
            else:
                raise ValueError(f"Unknown arg: {param}")

        return diffusers_params


class SDAPIDiffusionModelMixin:
    def txt2img(self, **kwargs):
        available_args = set(
            SDAPIToDiffusersConverter.txt2img_identical_args
            + list(SDAPIToDiffusersConverter.txt2img_arg_mapping)
        )
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

        converted_kwargs = SDAPIToDiffusersConverter.convert_txt2img_to_diffusers(
            available_kwargs
        )
        width, height = converted_kwargs.pop("width", None), converted_kwargs.pop(
            "height", None
        )
        if width and height:
            converted_kwargs["size"] = f"{width}*{height}"
        result = self.text_to_image(response_format="b64_json", **converted_kwargs)  # type: ignore

        # convert to SD API result
        return {
            "images": [r["b64_json"] for r in result["data"]],
            "info": {"created": result["created"]},
            "parameters": {},
        }
