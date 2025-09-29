# Copyright 2022-2025 XProbe Inc.
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
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import PIL.Image
from gradio import Markdown

from ...client.restful.restful_client import (
    RESTfulAudioModelHandle,
    RESTfulImageModelHandle,
    RESTfulVideoModelHandle,
)

logger = logging.getLogger(__name__)


class MediaInterface:
    def __init__(
        self,
        endpoint: str,
        model_uid: str,
        model_family: str,
        model_name: str,
        model_id: str,
        model_revision: str,
        model_ability: List[str],
        model_type: str,
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
        self.model_type = model_type
        self.controlnet = controlnet
        self.access_token = (
            access_token.replace("Bearer ", "") if access_token is not None else None
        )

    def build(self) -> gr.Blocks:
        if self.model_type == "image":
            assert "stable_diffusion" in self.model_family

        interface = self.build_main_interface()
        interface.queue()
        # Gradio initiates the queue during a startup event, but since the app has already been
        # started, that event will not run, so manually invoke the startup events.
        # See: https://github.com/gradio-app/gradio/issues/5228
        try:
            interface.run_startup_events()
        except AttributeError:
            # compatibility
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
        from ...model.image.stable_diffusion.core import SAMPLING_METHODS

        def text_generate_image(
            prompt: str,
            n: int,
            size_width: int,
            size_height: int,
            guidance_scale: int,
            num_inference_steps: int,
            negative_prompt: Optional[str] = None,
            sampler_name: Optional[str] = None,
            progress=gr.Progress(),
        ) -> PIL.Image.Image:
            from ...client import RESTfulClient

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

            response = None
            exc = None
            request_id = str(uuid.uuid4())

            def run_in_thread():
                nonlocal exc, response
                try:
                    response = model.text_to_image(
                        request_id=request_id,
                        prompt=prompt,
                        n=n,
                        size=size,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        negative_prompt=negative_prompt,
                        sampler_name=sampler_name,
                        response_format="b64_json",
                    )
                except Exception as e:
                    exc = e

            t = threading.Thread(target=run_in_thread)
            t.start()
            while t.is_alive():
                try:
                    cur_progress = client.get_progress(request_id)["progress"]
                except (KeyError, RuntimeError):
                    cur_progress = 0.0

                progress(cur_progress, desc="Generating images")
                time.sleep(1)

            if exc:
                raise exc

            images = []
            for image_dict in response["data"]:  # type: ignore
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
        from ...model.image.stable_diffusion.core import SAMPLING_METHODS

        def image_generate_image(
            prompt: str,
            negative_prompt: str,
            images: Optional[List[PIL.Image.Image]],
            n: int,
            size_width: int,
            size_height: int,
            guidance_scale: int,
            num_inference_steps: int,
            padding_image_to_multiple: int,
            strength: float,
            sampler_name: Optional[str] = None,
            progress=gr.Progress(),
        ) -> PIL.Image.Image:
            from ...client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulImageModelHandle)

            if size_width > 0 and size_height > 0:
                size = f"{int(size_width)}*{int(size_height)}"
            else:
                size = None
            guidance_scale = None if guidance_scale == -1 else guidance_scale  # type: ignore
            num_inference_steps = (
                None if num_inference_steps == -1 else num_inference_steps  # type: ignore
            )
            padding_image_to_multiple = None if padding_image_to_multiple == -1 else padding_image_to_multiple  # type: ignore
            # Initialize kwargs and handle strength parameter
            kwargs = {}
            if strength is not None:
                kwargs["strength"] = strength
            sampler_name = None if sampler_name == "default" else sampler_name

            # Handle single image or multiple images
            if images is None:
                raise ValueError("Please upload at least one image")

            # Process uploaded files to get PIL images
            processed_images = process_uploaded_files(images)
            if processed_images is None:
                raise ValueError("Please upload at least one image")

            # Convert all images to bytes
            image_bytes_list = []
            for img in processed_images:
                bio = io.BytesIO()
                img.save(bio, format="png")
                image_bytes_list.append(bio.getvalue())

            response = None
            exc = None
            request_id = str(uuid.uuid4())

            def run_in_thread():
                nonlocal exc, response
                try:
                    response = model.image_to_image(
                        request_id=request_id,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        n=n,
                        image=image_bytes_list,
                        size=size,
                        response_format="b64_json",
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        padding_image_to_multiple=padding_image_to_multiple,
                        sampler_name=sampler_name,
                        **kwargs,
                    )
                except Exception as e:
                    exc = e

            t = threading.Thread(target=run_in_thread)
            t.start()
            while t.is_alive():
                try:
                    cur_progress = client.get_progress(request_id)["progress"]
                except (KeyError, RuntimeError):
                    cur_progress = 0.0

                progress(cur_progress, desc="Generating images")
                time.sleep(1)

            if exc:
                raise exc

            images = []
            for image_dict in response["data"]:  # type: ignore
                assert image_dict["b64_json"] is not None
                image_data = base64.b64decode(image_dict["b64_json"])
                image = PIL.Image.open(io.BytesIO(image_data))
                images.append(image)

            return images

        with gr.Blocks() as image2image_interface:
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
                    guidance_scale = gr.Number(label="Guidance scale", value=-1)
                    num_inference_steps = gr.Number(
                        label="Inference Step Number", value=-1
                    )
                    padding_image_to_multiple = gr.Number(
                        label="Padding image to multiple", value=-1
                    )
                    strength = gr.Slider(
                        label="Strength", value=0.6, step=0.1, minimum=0.0, maximum=1.0
                    )
                    sampler_name = gr.Dropdown(
                        choices=SAMPLING_METHODS,
                        value="default",
                        label="Sampling method",
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Upload Images")
                        gr.Markdown(
                            "*Multiple images supported for image-to-image generation*"
                        )
                        uploaded_images = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="Upload Images",
                        )
                        image_preview = gr.Gallery(label="Image Preview", height=300)
                    with gr.Column(scale=1):
                        output_gallery = gr.Gallery()

            # Function to handle file uploads and convert to PIL images
            def process_uploaded_files(files):
                if files is None:
                    return None

                images = []
                for file_info in files:
                    if isinstance(file_info, dict) and "name" in file_info:
                        # Handle file info format from gradio
                        file_path = file_info["name"]
                        try:
                            img = PIL.Image.open(file_path)
                            images.append(img)
                        except Exception as e:
                            logger.warning(f"Failed to load image {file_path}: {e}")
                    elif hasattr(file_info, "name"):
                        # Handle file object
                        try:
                            img = PIL.Image.open(file_info.name)
                            images.append(img)
                        except Exception as e:
                            logger.warning(
                                f"Failed to load image {file_info.name}: {e}"
                            )

                return images if images else None

            # Update gallery when files are uploaded
            def update_gallery(files):
                images = process_uploaded_files(files)
                return images if images else []

            uploaded_images.change(
                update_gallery, inputs=[uploaded_images], outputs=[image_preview]
            )

            generate_button.click(
                image_generate_image,
                inputs=[
                    prompt,
                    negative_prompt,
                    uploaded_images,
                    n,
                    size_width,
                    size_height,
                    guidance_scale,
                    num_inference_steps,
                    padding_image_to_multiple,
                    strength,
                    sampler_name,
                ],
                outputs=output_gallery,
            )
        return image2image_interface

    def inpainting_interface(self) -> "gr.Blocks":
        from ...model.image.stable_diffusion.core import SAMPLING_METHODS

        def preview_mask(
            image_editor_output: Dict[str, Any],
        ) -> PIL.Image.Image:
            """Preview the generated mask without submitting inpainting task"""
            # Extract original image and mask from ImageEditor output
            if not image_editor_output or "background" not in image_editor_output:
                return PIL.Image.new(
                    "L", (512, 512), 0
                )  # Return black image if no input

            # Get the original image (background)
            original_image = image_editor_output["background"]

            # Get the composite image which contains the edits
            composite_image = image_editor_output.get("composite", original_image)

            # Create mask from the differences between original and composite
            # White areas in composite indicate regions to inpaint
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")
            if composite_image.mode != "RGB":
                composite_image = composite_image.convert("RGB")

            # Create mask by finding differences (white drawn areas)
            mask_image = PIL.Image.new("L", original_image.size, 0)
            orig_data = original_image.load()
            comp_data = composite_image.load()
            mask_data = mask_image.load()

            for y in range(original_image.size[1]):
                for x in range(original_image.size[0]):
                    orig_pixel = orig_data[x, y]
                    comp_pixel = comp_data[x, y]
                    # If pixels are different, assume it's a drawn area (white for inpainting)
                    if orig_pixel != comp_pixel:
                        mask_data[x, y] = 255  # White for inpainting

            return mask_image

        def process_inpainting(
            prompt: str,
            negative_prompt: str,
            image_editor_output: Dict[str, Any],
            uploaded_mask: Optional[PIL.Image.Image],
            n: int,
            size_width: int,
            size_height: int,
            guidance_scale: int,
            num_inference_steps: int,
            padding_image_to_multiple: int,
            strength: float,
            sampler_name: Optional[str] = None,
            progress=gr.Progress(),
        ) -> List[PIL.Image.Image]:
            from ...client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulImageModelHandle)

            if size_width > 0 and size_height > 0:
                size = f"{int(size_width)}*{int(size_height)}"
            else:
                size = None
            guidance_scale = None if guidance_scale == -1 else guidance_scale  # type: ignore
            num_inference_steps = (
                None if num_inference_steps == -1 else num_inference_steps  # type: ignore
            )
            padding_image_to_multiple = None if padding_image_to_multiple == -1 else padding_image_to_multiple  # type: ignore
            # Initialize kwargs and handle strength parameter
            kwargs = {}
            if strength is not None:
                kwargs["strength"] = strength
            sampler_name = None if sampler_name == "default" else sampler_name

            # Get the original image for inpainting
            if not image_editor_output or "background" not in image_editor_output:
                raise ValueError("Please upload and edit an image first")
            original_image = image_editor_output["background"]

            # Convert original image to RGB if needed
            if original_image.mode == "RGBA":
                # Create a white background and paste the RGBA image onto it
                rgb_image = PIL.Image.new("RGB", original_image.size, (255, 255, 255))
                rgb_image.paste(
                    original_image, mask=original_image.split()[3]
                )  # Use alpha channel as mask
                original_image = rgb_image
            elif original_image.mode != "RGB":
                original_image = original_image.convert("RGB")

            # Assert that original image is RGB format
            assert (
                original_image.mode == "RGB"
            ), f"Expected RGB image, got {original_image.mode}"

            # Use uploaded mask if provided, otherwise generate from editor
            if uploaded_mask is not None:
                mask_image = uploaded_mask

                # Convert RGBA to RGB if needed
                if mask_image.mode == "RGBA":
                    # Create a white background and paste the RGBA image onto it
                    rgb_mask = PIL.Image.new("RGB", mask_image.size, (255, 255, 255))
                    rgb_mask.paste(
                        mask_image, mask=(mask_image.split()[3])
                    )  # Use alpha channel as mask
                    mask_image = rgb_mask
                elif mask_image.mode != "RGB":
                    mask_image = mask_image.convert("RGB")

                # Ensure mask is the same size as original image
                if mask_image.size != original_image.size:
                    mask_image = mask_image.resize(original_image.size)

                # Assert that mask image is RGB format
                assert (
                    mask_image.mode == "RGB"
                ), f"Expected RGB mask, got {mask_image.mode}"
            else:
                # Generate mask using the preview function
                mask_image = preview_mask(image_editor_output)
                # Assert that generated mask is L format (grayscale)
                assert mask_image.mode == "L", f"Expected L mask, got {mask_image.mode}"

            bio = io.BytesIO()
            original_image.save(bio, format="png")

            mask_bio = io.BytesIO()
            mask_image.save(mask_bio, format="png")

            response = None
            exc = None
            request_id = str(uuid.uuid4())

            def run_in_thread():
                nonlocal exc, response
                try:
                    response = model.inpainting(
                        request_id=request_id,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        n=n,
                        image=bio.getvalue(),
                        mask_image=mask_bio.getvalue(),
                        size=size,
                        response_format="b64_json",
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        padding_image_to_multiple=padding_image_to_multiple,
                        sampler_name=sampler_name,
                        **kwargs,
                    )
                except Exception as e:
                    exc = e

            t = threading.Thread(target=run_in_thread)
            t.start()
            while t.is_alive():
                try:
                    cur_progress = client.get_progress(request_id)["progress"]
                except (KeyError, RuntimeError):
                    cur_progress = 0.0

                progress(cur_progress, desc="Inpainting images")
                time.sleep(1)

            if exc:
                raise exc

            images = []
            for image_dict in response["data"]:  # type: ignore
                assert image_dict["b64_json"] is not None
                image_data = base64.b64decode(image_dict["b64_json"])
                image = PIL.Image.open(io.BytesIO(image_data))
                images.append(image)

            return images

        with gr.Blocks() as inpainting_interface:
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
                    guidance_scale = gr.Number(label="Guidance scale", value=-1)
                    num_inference_steps = gr.Number(
                        label="Inference Step Number", value=-1
                    )
                    padding_image_to_multiple = gr.Number(
                        label="Padding image to multiple", value=-1
                    )
                    strength = gr.Slider(
                        label="Strength", value=0.6, step=0.1, minimum=0.0, maximum=1.0
                    )
                    sampler_name = gr.Dropdown(
                        choices=SAMPLING_METHODS,
                        value="default",
                        label="Sampling method",
                    )

                with gr.Row():
                    with gr.Column(scale=2):
                        image_editor = gr.ImageEditor(
                            type="pil",
                            label="Edit Image and Create Mask (Draw white areas to inpaint)",
                            interactive=True,
                            height=400,
                        )

                        # Mask controls below the editor
                        with gr.Row():
                            preview_button = gr.Button("Preview Mask", size="sm")
                            upload_mask = gr.Image(
                                type="pil",
                                label="Or upload mask image directly",
                                interactive=True,
                            )
                        with gr.Row():
                            mask_output = gr.Image(
                                label="Current Mask Preview",
                                interactive=False,
                                height=200,
                            )

                    with gr.Column(scale=1):
                        gr.Markdown("### Inpainting Results")
                        output_gallery = gr.Gallery()

            preview_button.click(
                preview_mask,
                inputs=[image_editor],
                outputs=[mask_output],
            )

            # When user uploads a mask, display it
            def process_uploaded_mask(
                mask: Optional[PIL.Image.Image],
            ) -> PIL.Image.Image:
                if mask is None:
                    return PIL.Image.new("L", (512, 512), 0)

                # Convert RGBA to grayscale for preview
                if mask.mode == "RGBA":
                    # Use alpha channel for mask preview
                    alpha = mask.split()[3]
                    mask = alpha.convert("L")
                elif mask.mode != "L":
                    # Convert to grayscale
                    mask = mask.convert("L")

                return mask

            upload_mask.change(
                process_uploaded_mask, inputs=[upload_mask], outputs=[mask_output]
            )

            generate_button.click(
                process_inpainting,
                inputs=[
                    prompt,
                    negative_prompt,
                    image_editor,
                    upload_mask,
                    n,
                    size_width,
                    size_height,
                    guidance_scale,
                    num_inference_steps,
                    padding_image_to_multiple,
                    strength,
                    sampler_name,
                ],
                outputs=[output_gallery],
            )
        return inpainting_interface

    def text2video_interface(self) -> "gr.Blocks":
        def text_generate_video(
            prompt: str,
            negative_prompt: str,
            num_frames: int,
            fps: int,
            num_inference_steps: int,
            guidance_scale: float,
            width: int,
            height: int,
            progress=gr.Progress(),
        ) -> List[Tuple[str, str]]:
            from ...client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulVideoModelHandle)

            request_id = str(uuid.uuid4())
            response = None
            exc = None

            # Run generation in a separate thread to allow progress tracking
            def run_in_thread():
                nonlocal exc, response
                try:
                    response = model.text_to_video(
                        request_id=request_id,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_frames=num_frames,
                        fps=fps,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        response_format="b64_json",
                    )
                except Exception as e:
                    exc = e

            t = threading.Thread(target=run_in_thread)
            t.start()

            # Update progress bar during generation
            while t.is_alive():
                try:
                    cur_progress = client.get_progress(request_id)["progress"]
                except Exception:
                    cur_progress = 0.0
                progress(cur_progress, desc="Generating video")
                time.sleep(1)

            if exc:
                raise exc

            # Decode and return the generated video
            videos = []
            for video_dict in response["data"]:  # type: ignore
                video_data = base64.b64decode(video_dict["b64_json"])
                video_path = f"/tmp/{uuid.uuid4()}.mp4"
                with open(video_path, "wb") as f:
                    f.write(video_data)
                videos.append((video_path, "Generated Video"))

            return videos

        # Gradio UI definition
        with gr.Blocks() as text2video_ui:
            # Prompt & Negative Prompt (stacked vertically)
            prompt = gr.Textbox(label="Prompt", placeholder="Enter video prompt")
            negative_prompt = gr.Textbox(
                label="Negative Prompt", placeholder="Enter negative prompt"
            )

            # Parameters (2-column layout)
            with gr.Row():
                with gr.Column():
                    width = gr.Number(label="Width", value=512)
                    num_frames = gr.Number(label="Frames", value=16)
                    steps = gr.Number(label="Inference Steps", value=25)
                with gr.Column():
                    height = gr.Number(label="Height", value=512)
                    fps = gr.Number(label="FPS", value=8)
                    guidance_scale = gr.Slider(
                        label="Guidance Scale", minimum=1, maximum=20, value=7.5
                    )

            # Generate button
            generate = gr.Button("Generate")

            # Output gallery
            gallery = gr.Gallery(label="Generated Videos", columns=2)

            # Button click logic
            generate.click(
                fn=text_generate_video,
                inputs=[
                    prompt,
                    negative_prompt,
                    num_frames,
                    fps,
                    steps,
                    guidance_scale,
                    width,
                    height,
                ],
                outputs=gallery,
            )

        return text2video_ui

    def image2video_interface(self) -> "gr.Blocks":
        def image_generate_video(
            image: "PIL.Image.Image",
            prompt: str,
            negative_prompt: str,
            num_frames: int,
            fps: int,
            num_inference_steps: int,
            guidance_scale: float,
            width: int,
            height: int,
            progress=gr.Progress(),
        ) -> List[Tuple[str, str]]:
            from ...client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulVideoModelHandle)

            request_id = str(uuid.uuid4())
            response = None
            exc = None

            # Convert uploaded image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")

            # Run generation in a separate thread
            def run_in_thread():
                nonlocal exc, response
                try:
                    response = model.image_to_video(
                        request_id=request_id,
                        image=buffered.getvalue(),
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_frames=num_frames,
                        fps=fps,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        response_format="b64_json",
                    )
                except Exception as e:
                    exc = e

            t = threading.Thread(target=run_in_thread)
            t.start()

            # Progress loop
            while t.is_alive():
                try:
                    cur_progress = client.get_progress(request_id)["progress"]
                except Exception:
                    cur_progress = 0.0
                progress(cur_progress, desc="Generating video from image")
                time.sleep(1)

            if exc:
                raise exc

            # Decode and return video files
            videos = []
            for video_dict in response["data"]:  # type: ignore
                video_data = base64.b64decode(video_dict["b64_json"])
                video_path = f"/tmp/{uuid.uuid4()}.mp4"
                with open(video_path, "wb") as f:
                    f.write(video_data)
                videos.append((video_path, "Generated Video"))

            return videos

        # Gradio UI
        with gr.Blocks() as image2video_ui:
            image = gr.Image(label="Input Image", type="pil")

            prompt = gr.Textbox(label="Prompt", placeholder="Enter video prompt")
            negative_prompt = gr.Textbox(
                label="Negative Prompt", placeholder="Enter negative prompt"
            )

            with gr.Row():
                with gr.Column():
                    width = gr.Number(label="Width", value=512)
                    num_frames = gr.Number(label="Frames", value=16)
                    steps = gr.Number(label="Inference Steps", value=25)
                with gr.Column():
                    height = gr.Number(label="Height", value=512)
                    fps = gr.Number(label="FPS", value=8)
                    guidance_scale = gr.Slider(
                        label="Guidance Scale", minimum=1, maximum=20, value=7.5
                    )

            generate = gr.Button("Generate")
            gallery = gr.Gallery(label="Generated Videos", columns=2)

            generate.click(
                fn=image_generate_video,
                inputs=[
                    image,
                    prompt,
                    negative_prompt,
                    num_frames,
                    fps,
                    steps,
                    guidance_scale,
                    width,
                    height,
                ],
                outputs=gallery,
            )

        return image2video_ui

    def flf2video_interface(self) -> "gr.Blocks":
        def generate_video_from_flf(
            first_frame: "PIL.Image.Image",
            last_frame: "PIL.Image.Image",
            prompt: str,
            negative_prompt: str,
            num_frames: int,
            fps: int,
            num_inference_steps: int,
            guidance_scale: float,
            width: int,
            height: int,
            progress=gr.Progress(),
        ) -> List[Tuple[str, str]]:
            from ...client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert hasattr(model, "flf_to_video")

            request_id = str(uuid.uuid4())
            response = None
            exc = None

            buffer_first = io.BytesIO()
            buffer_last = io.BytesIO()
            first_frame.save(buffer_first, format="PNG")
            last_frame.save(buffer_last, format="PNG")

            def run_in_thread():
                nonlocal exc, response
                try:
                    response = model.flf_to_video(
                        first_frame=buffer_first.getvalue(),
                        last_frame=buffer_last.getvalue(),
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        n=1,
                        num_frames=num_frames,
                        fps=fps,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        response_format="b64_json",
                        request_id=request_id,
                    )
                except Exception as e:
                    exc = e

            t = threading.Thread(target=run_in_thread)
            t.start()

            while t.is_alive():
                try:
                    cur_progress = client.get_progress(request_id)["progress"]
                except Exception:
                    cur_progress = 0.0
                progress(cur_progress, desc="Generating video from first/last frames")
                time.sleep(1)

            if exc:
                raise exc

            videos = []
            for video_dict in response["data"]:  # type: ignore
                video_data = base64.b64decode(video_dict["b64_json"])
                video_path = f"/tmp/{uuid.uuid4()}.mp4"
                with open(video_path, "wb") as f:
                    f.write(video_data)
                videos.append((video_path, "Generated Video"))

            return videos

        # Gradio UI
        with gr.Blocks() as flf2video_ui:
            with gr.Row():
                first_frame = gr.Image(label="First Frame", type="pil")
                last_frame = gr.Image(label="Last Frame", type="pil")

            prompt = gr.Textbox(label="Prompt", placeholder="Enter video prompt")
            negative_prompt = gr.Textbox(
                label="Negative Prompt", placeholder="Enter negative prompt"
            )

            with gr.Row():
                with gr.Column():
                    width = gr.Number(label="Width", value=512)
                    num_frames = gr.Number(label="Frames", value=16)
                    steps = gr.Number(label="Inference Steps", value=25)
                with gr.Column():
                    height = gr.Number(label="Height", value=512)
                    fps = gr.Number(label="FPS", value=8)
                    guidance_scale = gr.Slider(
                        label="Guidance Scale", minimum=1, maximum=20, value=7.5
                    )

            generate = gr.Button("Generate")
            gallery = gr.Gallery(label="Generated Videos", columns=2)

            generate.click(
                fn=generate_video_from_flf,
                inputs=[
                    first_frame,
                    last_frame,
                    prompt,
                    negative_prompt,
                    num_frames,
                    fps,
                    steps,
                    guidance_scale,
                    width,
                    height,
                ],
                outputs=gallery,
            )

        return flf2video_ui

    def audio2text_interface(self) -> "gr.Blocks":
        def transcribe_audio(
            audio_path: str,
            language: Optional[str],
            prompt: Optional[str],
            temperature: float,
        ) -> str:
            from ...client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulAudioModelHandle)

            with open(audio_path, "rb") as f:
                audio_data = f.read()

            response = model.transcriptions(
                audio=audio_data,
                language=language or None,
                prompt=prompt or None,
                temperature=temperature,
                response_format="json",
            )

            return response.get("text", "No transcription result.")

        with gr.Blocks() as audio2text_ui:
            with gr.Row():
                audio_input = gr.Audio(
                    type="filepath",
                    label="Upload or Record Audio",
                    sources=["upload", "microphone"],  # ✅ support both
                )
            with gr.Row():
                language = gr.Textbox(
                    label="Language", placeholder="e.g. en or zh", value=""
                )
                prompt = gr.Textbox(
                    label="Prompt (optional)",
                    placeholder="Provide context or vocabulary",
                )
                temperature = gr.Slider(
                    label="Temperature", minimum=0.0, maximum=1.0, value=0.0, step=0.1
                )
            transcribe_btn = gr.Button("Transcribe")
            output_text = gr.Textbox(label="Transcription", lines=5)

            transcribe_btn.click(
                fn=transcribe_audio,
                inputs=[audio_input, language, prompt, temperature],
                outputs=output_text,
            )

        return audio2text_ui

    def text2speech_interface(self) -> "gr.Blocks":
        def tts_generate(
            input_text: str,
            voice: str,
            speed: float,
            prompt_speech_file,
            prompt_text: Optional[str],
        ) -> str:
            from ...client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert hasattr(model, "speech")

            prompt_speech_bytes = None
            if prompt_speech_file is not None:
                with open(prompt_speech_file, "rb") as f:
                    prompt_speech_bytes = f.read()

            kw: Dict[str, Any] = {}
            if prompt_speech_bytes:
                kw["prompt_speech"] = prompt_speech_bytes
            if prompt_text:
                kw["prompt_text"] = prompt_text

            response = model.speech(
                input=input_text, voice=voice, speed=speed, response_format="mp3", **kw
            )

            # Write to a temp .mp3 file and return its path
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp3")
            with open(audio_path, "wb") as f:
                f.write(response)

            return audio_path

        # Determine model abilities
        supports_basic_tts = "text2audio" in self.model_ability
        supports_zero_shot = "text2audio_zero_shot" in self.model_ability
        supports_voice_cloning = "text2audio_voice_cloning" in self.model_ability

        # Show ability info
        ability_info = []
        if supports_basic_tts:
            ability_info.append("✅ Basic TTS (text-to-speech)")
        if supports_zero_shot:
            ability_info.append("✅ Zero-shot TTS (voice selection)")
        if supports_voice_cloning:
            ability_info.append("✅ Voice Cloning (requires reference audio)")

        # Gradio UI
        with gr.Blocks() as tts_ui:
            gr.Markdown(f"**Model Abilities:**\n{chr(10).join(ability_info)}")

            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Text", placeholder="Enter text to synthesize"
                    )
                    voice = gr.Textbox(
                        label="Voice", placeholder="Optional voice ID", value=""
                    )
                    speed = gr.Slider(
                        label="Speed", minimum=0.5, maximum=2.0, value=1.0, step=0.1
                    )

                    # Show voice cloning controls if supported
                    if supports_voice_cloning:
                        gr.Markdown("---\n**Voice Cloning Options**")
                        # Make voice cloning required if model doesn't support zero-shot
                        if supports_zero_shot:
                            prompt_speech = gr.Audio(
                                label="Prompt Speech (for cloning, optional)",
                                type="filepath",
                            )
                            prompt_text = gr.Textbox(
                                label="Prompt Text (for cloning, optional)",
                                placeholder="Text of the prompt speech",
                            )
                        else:
                            prompt_speech = gr.Audio(
                                label="Prompt Speech (for cloning, required)",
                                type="filepath",
                            )
                            prompt_text = gr.Textbox(
                                label="Prompt Text (for cloning, optional)",
                                placeholder="Text of the prompt speech (optional)",
                            )
                    else:
                        # Hidden components for API compatibility
                        prompt_speech = gr.Audio(visible=False)
                        prompt_text = gr.Textbox(visible=False)

                    generate = gr.Button("Generate")

                with gr.Column():
                    audio_output = gr.Audio(label="Generated Audio", type="filepath")

            generate.click(
                fn=tts_generate,
                inputs=[input_text, voice, speed, prompt_speech, prompt_text],
                outputs=audio_output,
            )

        return tts_ui

    def build_main_interface(self) -> "gr.Blocks":
        if self.model_type == "image":
            title = f"🎨 Xinference Stable Diffusion: {self.model_name} 🎨"
        elif self.model_type == "video":
            title = f"🎨 Xinference Video Generation: {self.model_name} 🎨"
        else:
            assert self.model_type == "audio"
            title = f"🎨 Xinference Audio Model: {self.model_name} 🎨"
        with gr.Blocks(
            title=title,
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
                    <h1 class="center" style='text-align: center; margin-bottom: 1rem'>{title}</h1>
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
            if "inpainting" in self.model_ability:
                with gr.Tab("Inpainting"):
                    self.inpainting_interface()
            if "text2video" in self.model_ability:
                with gr.Tab("Text to Video"):
                    self.text2video_interface()
            if "image2video" in self.model_ability:
                with gr.Tab("Image to Video"):
                    self.image2video_interface()
            if "firstlastframe2video" in self.model_ability:
                with gr.Tab("FirstLastFrame to Video"):
                    self.flf2video_interface()
            if "audio2text" in self.model_ability:
                with gr.Tab("Audio to Text"):
                    self.audio2text_interface()
            if "text2audio" in self.model_ability:
                with gr.Tab("Text to Audio"):
                    self.text2speech_interface()
        return app
