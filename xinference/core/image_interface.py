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
import logging
import os
from typing import Dict, List, Optional, Union

import gradio as gr
import PIL.Image
from gradio import Markdown

from ..client.restful.restful_client import RESTfulImageModelHandle

logger = logging.getLogger(__name__)


class ImageInterface:
    def __init__(
        self,
        endpoint: str,
        model_uid: str,
        model_family: str,
        model_name: str,
        model_id: str,
        model_revision: str,
        model_ability: List[str],
        controlnet: Union[None, List[Dict[str, Union[str, None]]]],
        access_token: Optional[str],
    ):
        self.endpoint = endpoint
        self.model_uid = model_uid
        self.model_family = model_family
        self.model_name = model_name
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_ability = model_ability
        self.controlnet = controlnet
        self.access_token = (
            access_token.replace("Bearer ", "") if access_token is not None else None
        )

    def build(self) -> gr.Blocks:
        assert "stable_diffusion" in self.model_family

        interface = self.build_main_interface()
        interface.queue()
        # Gradio initiates the queue during a startup event, but since the app has already been
        # started, that event will not run, so manually invoke the startup events.
        # See: https://github.com/gradio-app/gradio/issues/5228
        interface.startup_events()
        favicon_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.pardir,
            "web",
            "ui",
            "public",
            "favicon.svg",
        )
        interface.favicon_path = favicon_path
        return interface

    def text2image_interface(self) -> "gr.Blocks":
        from ..model.image.stable_diffusion.core import SAMPLING_METHODS

        def text_generate_image(
            prompt: str,
            n: int,
            size_width: int,
            size_height: int,
            guidance_scale: int,
            num_inference_steps: int,
            negative_prompt: Optional[str] = None,
            sampler_name: Optional[str] = None,
        ) -> PIL.Image.Image:
            from ..client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulImageModelHandle)

            size = f"{int(size_width)}*{int(size_height)}"
            guidance_scale = None if guidance_scale == -1 else guidance_scale  # type: ignore
            num_inference_steps = (
                None if num_inference_steps == -1 else num_inference_steps  # type: ignore
            )
            sampler_name = None if sampler_name == "default" else sampler_name

            response = model.text_to_image(
                prompt=prompt,
                n=n,
                size=size,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                sampler_name=sampler_name,
                response_format="b64_json",
            )

            images = []
            for image_dict in response["data"]:
                assert image_dict["b64_json"] is not None
                image_data = base64.b64decode(image_dict["b64_json"])
                image = PIL.Image.open(io.BytesIO(image_data))
                images.append(image)

            return images

        with gr.Blocks() as text2image_vl_interface:
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=10):
                        prompt = gr.Textbox(
                            label="Prompt",
                            show_label=True,
                            placeholder="Enter prompt here...",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative prompt",
                            show_label=True,
                            placeholder="Enter negative prompt here...",
                        )
                    with gr.Column(scale=1):
                        generate_button = gr.Button("Generate")

                with gr.Row():
                    n = gr.Number(label="Number of Images", value=1)
                    size_width = gr.Number(label="Width", value=1024)
                    size_height = gr.Number(label="Height", value=1024)
                with gr.Row():
                    guidance_scale = gr.Number(label="Guidance scale", value=-1)
                    num_inference_steps = gr.Number(
                        label="Inference Step Number", value=-1
                    )
                    sampler_name = gr.Dropdown(
                        choices=SAMPLING_METHODS,
                        value="default",
                        label="Sampling method",
                    )

                with gr.Column():
                    image_output = gr.Gallery()

            generate_button.click(
                text_generate_image,
                inputs=[
                    prompt,
                    n,
                    size_width,
                    size_height,
                    guidance_scale,
                    num_inference_steps,
                    negative_prompt,
                    sampler_name,
                ],
                outputs=image_output,
            )

        return text2image_vl_interface

    def image2image_interface(self) -> "gr.Blocks":
        from ..model.image.stable_diffusion.core import SAMPLING_METHODS

        def image_generate_image(
            prompt: str,
            negative_prompt: str,
            image: PIL.Image.Image,
            n: int,
            size_width: int,
            size_height: int,
            num_inference_steps: int,
            padding_image_to_multiple: int,
            sampler_name: Optional[str] = None,
        ) -> PIL.Image.Image:
            from ..client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulImageModelHandle)

            if size_width > 0 and size_height > 0:
                size = f"{int(size_width)}*{int(size_height)}"
            else:
                size = None
            num_inference_steps = (
                None if num_inference_steps == -1 else num_inference_steps  # type: ignore
            )
            padding_image_to_multiple = None if padding_image_to_multiple == -1 else padding_image_to_multiple  # type: ignore
            sampler_name = None if sampler_name == "default" else sampler_name

            bio = io.BytesIO()
            image.save(bio, format="png")

            response = model.image_to_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                n=n,
                image=bio.getvalue(),
                size=size,
                response_format="b64_json",
                num_inference_steps=num_inference_steps,
                padding_image_to_multiple=padding_image_to_multiple,
                sampler_name=sampler_name,
            )

            images = []
            for image_dict in response["data"]:
                assert image_dict["b64_json"] is not None
                image_data = base64.b64decode(image_dict["b64_json"])
                image = PIL.Image.open(io.BytesIO(image_data))
                images.append(image)

            return images

        with gr.Blocks() as image2image_inteface:
            with gr.Column():
                with gr.Row():
                    with gr.Column(scale=10):
                        prompt = gr.Textbox(
                            label="Prompt",
                            show_label=True,
                            placeholder="Enter prompt here...",
                        )
                        negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            show_label=True,
                            placeholder="Enter negative prompt here...",
                        )
                    with gr.Column(scale=1):
                        generate_button = gr.Button("Generate")

                with gr.Row():
                    n = gr.Number(label="Number of image", value=1)
                    size_width = gr.Number(label="Width", value=-1)
                    size_height = gr.Number(label="Height", value=-1)

                with gr.Row():
                    num_inference_steps = gr.Number(
                        label="Inference Step Number", value=-1
                    )
                    padding_image_to_multiple = gr.Number(
                        label="Padding image to multiple", value=-1
                    )
                    sampler_name = gr.Dropdown(
                        choices=SAMPLING_METHODS,
                        value="default",
                        label="Sampling method",
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        uploaded_image = gr.Image(type="pil", label="Upload Image")
                    with gr.Column(scale=1):
                        output_gallery = gr.Gallery()

            generate_button.click(
                image_generate_image,
                inputs=[
                    prompt,
                    negative_prompt,
                    uploaded_image,
                    n,
                    size_width,
                    size_height,
                    num_inference_steps,
                    padding_image_to_multiple,
                    sampler_name,
                ],
                outputs=output_gallery,
            )
        return image2image_inteface

    def build_main_interface(self) -> "gr.Blocks":
        with gr.Blocks(
            title=f"🎨 Xinference Stable Diffusion: {self.model_name} 🎨",
            css="""
                    .center{
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        padding: 0px;
                        color: #9ea4b0 !important;
                    }
                    """,
            analytics_enabled=False,
        ) as app:
            Markdown(
                f"""
                    <h1 class="center" style='text-align: center; margin-bottom: 1rem'>🎨 Xinference Stable Diffusion: {self.model_name} 🎨</h1>
                    """
            )
            Markdown(
                f"""
                    <div class="center">
                    Model ID: {self.model_uid}
                    </div>
                    """
            )
            if "text2image" in self.model_ability:
                with gr.Tab("Text to Image"):
                    self.text2image_interface()
            if "image2image" in self.model_ability:
                with gr.Tab("Image to Image"):
                    self.image2image_interface()

        return app
