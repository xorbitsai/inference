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

import logging
import os
from typing import Dict, List, Optional, Union

import gradio as gr
import PIL.Image

from ..client.restful.restful_client import RESTfulImageModelHandle

logger = logging.getLogger(__name__)


class Text2ImageInterface:
    def __init__(
        self,
        endpoint: str,
        model_uid: str,
        model_family: str,
        model_name: str,
        model_id: str,
        model_revision: str,
        controlnet: Union[None, List[Dict[str, Union[str, None]]]],
        access_token: Optional[str],
    ):
        self.endpoint = endpoint
        self.model_uid = model_uid
        self.model_family = model_family
        self.model_name = model_name
        self.model_id = model_id
        self.model_revision = model_revision
        self.controlnet = controlnet
        self.access_token = (
            access_token.replace("Bearer ", "") if access_token is not None else None
        )

    def build(self) -> gr.Blocks:
        assert "stable_diffusion" in self.model_family

        interface = self.build_stable_diffusion_interface()
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

    def build_stable_diffusion_interface(self) -> gr.Blocks:
        def generate_image(
            prompt: str,
            num_inference_steps: int,
            guidance_scale: float,
            seed: int,
        ) -> PIL.Image.Image:
            from ..client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulImageModelHandle)

            image_urls = model.text_to_image(
                prompt=prompt,
                n=1,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                **self.controlnet if self.controlnet else {},
            )
            logger.info(f"Image URLs: {image_urls}")
            image_path = image_urls["data"][0]["url"]

            img = PIL.Image.open(image_path)

            return img

        return gr.Interface(
            fn=generate_image,
            inputs=[
                gr.Textbox(label="Prompt"),
                gr.Slider(minimum=1, maximum=50, label="Num Inference Steps"),
                gr.Slider(minimum=1, maximum=20, label="Guidance Scale"),
                gr.Number(label="Seed"),
            ],
            outputs=gr.Image(type="pil"),
            title=f"🎨 Xinference Stable Diffusion: {self.model_name} 🎨",
            description="""
            Generate images from textual descriptions using Stable Diffusion.
            """,
            analytics_enabled=False,
        )
