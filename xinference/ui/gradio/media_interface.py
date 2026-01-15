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
        # Remove the stable_diffusion restriction to support OCR models
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

    def ocr_interface(self) -> "gr.Blocks":
        def extract_text_from_image(
            image: "PIL.Image.Image",
            ocr_type: str = "ocr",
            model_size: str = "gundam",
            test_compress: bool = False,
            enable_visualization: bool = False,
            save_results: bool = False,
            progress=gr.Progress(),
        ) -> Union[str, Tuple[str, str, str]]:
            from ...client import RESTfulClient

            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert hasattr(model, "ocr")

            # Convert PIL image to bytes
            import io

            buffered = io.BytesIO()
            if image.mode == "RGBA" or image.mode == "CMYK":
                image = image.convert("RGB")
            image.save(buffered, format="PNG")
            image_bytes = buffered.getvalue()

            progress(0.1, desc="Processing image for OCR")

            # Prepare prompt based on OCR type
            if ocr_type == "markdown":
                prompt = "<image>\nConvert this document to clean markdown format. Extract the text content and format it properly using markdown syntax. Do not include any coordinate annotations or special formatting markers."
            elif ocr_type == "format":
                prompt = "<image>\n<|grounding|>Convert the document to markdown with structure annotations. Include coordinate information for text regions and maintain the document structure."
            else:  # ocr
                prompt = "<image>\nFree OCR. Extract all text content from the image."

            try:
                logger.info(
                    f"Starting OCR processing - Type: {ocr_type}, Model Size: {model_size}"
                )
                logger.info(
                    f"Image info: {image.size if image else 'None'}, Mode: {image.mode if image else 'None'}"
                )

                if enable_visualization and hasattr(model, "visualize_ocr"):
                    # Use visualization method
                    logger.info("Using visualization method")
                    response = model.visualize_ocr(
                        image=image_bytes,
                        prompt=prompt,
                        model_size=model_size,
                        save_results=save_results,
                        eval_mode=True,
                    )

                    progress(0.8, desc="Processing visualization")

                    # Debug: Log response type and content
                    logger.info(f"Visualization response type: {type(response)}")
                    logger.info(f"Visualization response: {response}")

                    # Format response - handle both string and dict responses
                    if isinstance(response, dict):
                        if response.get("success"):
                            text_result = response.get("text", "No text extracted")
                        else:
                            error_msg = response.get(
                                "error", "OCR visualization failed"
                            )
                            # Return formatted error message for Markdown
                            error_md = f"**Error**: {error_msg}"
                            return error_md, "", ""
                    elif isinstance(response, str):
                        # Handle string response from original model
                        text_result = response
                    else:
                        text_result = str(response)

                    # Check if the result looks like Markdown and format it properly
                    if ocr_type == "markdown" and isinstance(text_result, str):
                        # Markdown mode - process LaTeX formulas for better rendering
                        try:
                            from .utils.latex import process_ocr_latex

                            if "\\" in text_result and (
                                "\\[" in text_result
                                or "\\(" in text_result
                                or "$" in text_result
                            ):
                                # Process LaTeX formulas for Markdown compatibility
                                text_result = process_ocr_latex(
                                    text_result, output_format="markdown"
                                )
                                logger.info(
                                    "Applied LaTeX processing for Markdown rendering (visualization)"
                                )
                        except ImportError:
                            logger.warning(
                                "LaTeX processing utils not available, using raw text"
                            )
                        pass
                    elif ocr_type == "format" and isinstance(text_result, str):
                        # For format mode, keep annotations but format as code block
                        if "<|ref|>" in text_result:
                            text_result = f"```\n{text_result}\n```"
                    elif ocr_type == "ocr" and isinstance(text_result, str):
                        # For plain text, format as a simple block
                        text_result = text_result  # Keep as plain text, Markdown will render it normally

                        # Add compression info if available
                    if (
                        isinstance(response, dict)
                        and test_compress
                        and "compression_ratio" in response
                    ):
                        compression_info = (
                            f"\n\n--- Compression Ratio Information ---\n"
                        )
                        compression_info += f"Compression Ratio: {response.get('compression_ratio', 'N/A')}\n"
                        compression_info += f"Valid Image Tokens: {response.get('valid_image_tokens', 'N/A')}\n"
                        compression_info += f"Output Text Tokens: {response.get('output_text_tokens', 'N/A')}\n"
                        text_result += compression_info

                    # Add visualization info
                    viz_info = {}
                    if isinstance(response, dict):
                        viz_info = response.get("visualization", {})
                        if viz_info.get("has_annotations"):
                            viz_text = f"\n\n--- Visualization Information ---\n"
                            viz_text += f"Number of Bounding Boxes: {viz_info.get('num_bounding_boxes', 0)}\n"
                            viz_text += f"Number of Extracted Images: {viz_info.get('num_extracted_images', 0)}\n"
                            text_result += viz_text

                        saved_files = response.get("saved_files", {})
                    else:
                        saved_files = {}

                    # Return text and visualization info
                    return text_result, str(viz_info), str(saved_files)
                else:
                    # Standard OCR branch
                    logger.info("Using standard OCR branch (not visualization)")
                    response = model.ocr(
                        image=image_bytes,
                        prompt=prompt,
                        model_size=model_size,
                        test_compress=test_compress,
                        save_results=save_results,
                        eval_mode=True,
                    )

                    progress(0.8, desc="Extracting text")

                    # Debug: Log response type and content
                    logger.info(f"Standard OCR response type: {type(response)}")
                    logger.info(
                        f"Standard OCR response content: {str(response)[:200]}..."
                    )

                    # Format response - handle both string and dict responses
                    if isinstance(response, dict):
                        if response.get("success"):
                            text_result = response.get("text", "No text extracted")

                            # Debug: Check if text is empty
                            if not text_result or not text_result.strip():
                                logger.warning("OCR returned empty text")
                                logger.warning(f"Full response: {response}")
                                # Return a helpful message instead of empty result
                                text_result = """**OCR Recognition Complete, No Text Detected**

**Possible Reasons:**
- Text in image is unclear or insufficient resolution
- Image format not supported
- Model unable to recognize text in image

**Suggestions:**
- Try uploading a clearer image
- Ensure text in image is clear and legible
- Handwritten text may have poor results

**Technical Information:**
- Model Status: Normal
- Image Size: Original {image.size if image else 'Unknown'}, Processed {response.get('image_size', 'Unknown')}
- Processing Mode: {response.get('model_size', 'Unknown')}"""
                        else:
                            error_msg = response.get("error", "OCR failed")
                            error_md = f"**Error**: {error_msg}"
                            return error_md, "", ""
                    elif isinstance(response, str):
                        # Handle string response from original model
                        text_result = response
                    else:
                        text_result = str(response)

                    # Format based on OCR type
                    if ocr_type == "markdown" and isinstance(text_result, str):
                        # Markdown mode - process LaTeX formulas for better rendering
                        try:
                            from .utils.latex import process_ocr_latex

                            if "\\" in text_result and (
                                "\\[" in text_result
                                or "\\(" in text_result
                                or "$" in text_result
                            ):
                                # Process LaTeX formulas for Markdown compatibility
                                text_result = process_ocr_latex(
                                    text_result, output_format="markdown"
                                )
                                logger.info(
                                    "Applied LaTeX processing for Markdown rendering"
                                )
                        except ImportError:
                            logger.warning(
                                "LaTeX processing utils not available, using raw text"
                            )
                        pass
                    elif ocr_type == "format" and isinstance(text_result, str):
                        # Format mode - show annotations in code block
                        if "<|ref|>" in text_result:
                            text_result = f"```text\n{text_result}\n```"
                    elif ocr_type == "ocr" and isinstance(text_result, str):
                        # Plain text mode - keep as plain text
                        text_result = text_result

                    # Add compression info if available
                    if (
                        isinstance(response, dict)
                        and test_compress
                        and "compression_ratio" in response
                    ):
                        compression_info = (
                            f"\n\n--- Compression Ratio Information ---\n"
                        )
                        compression_info += f"Compression Ratio: {response.get('compression_ratio', 'N/A')}\n"
                        compression_info += f"Valid Image Tokens: {response.get('valid_image_tokens', 'N/A')}\n"
                        compression_info += f"Output Text Tokens: {response.get('output_text_tokens', 'N/A')}\n"
                        text_result += compression_info

                    return text_result, "", ""

            except Exception as e:
                logger.error(f"OCR processing error: {e}")
                import traceback

                error_details = traceback.format_exc()
                logger.error(f"Full traceback: {error_details}")
                # Show error in markdown format for better visibility
                error_msg = f"""**OCR Processing Error**

```
{str(e)}
```

**Debug Info:**
- OCR Type: {ocr_type}
- Model Size: {model_size}
- Image Mode: {image.mode if image else 'None'}
- Image Size: {image.size if image else 'None'}
"""
                return error_msg, "", ""

            finally:
                progress(1.0, desc="OCR complete")

        with gr.Blocks() as ocr_interface:
            gr.Markdown(f"### Enhanced OCR Text Extraction with {self.model_name}")

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Upload Image for OCR",
                        interactive=True,
                        height=400,
                    )

                    gr.Markdown(f"**Current OCR Model:** {self.model_name}")

                    # Model configuration options
                    model_size = gr.Dropdown(
                        choices=["tiny", "small", "base", "large", "gundam"],
                        value="gundam",
                        label="Model Size",
                        info="Choose model size configuration",
                    )

                    ocr_type = gr.Dropdown(
                        choices=["ocr", "format", "markdown"],
                        value="ocr",
                        label="Output Format",
                        info="ocr: Plain text extraction, format: Structured document (with annotations), markdown: Standard Markdown format",
                    )

                    enable_visualization = gr.Checkbox(
                        label="Enable Visualization",
                        value=False,
                        info="Generate bounding boxes and annotations (only applicable to format mode)",
                    )

                    test_compress = gr.Checkbox(
                        label="Test Compression Ratio",
                        value=False,
                        info="Analyze image compression performance",
                    )

                    save_results = gr.Checkbox(
                        label="Save Results",
                        value=False,
                        info="Save OCR results to files (if supported)",
                    )

                    extract_btn = gr.Button("Extract Text", variant="primary")

                with gr.Column(scale=1):
                    # Create a bordered container for the output
                    with gr.Group(elem_classes="output-container"):
                        gr.Markdown("### 📄 Extraction Results")

                        text_output = gr.Markdown(
                            value="Extracted text will be displayed here...",
                            elem_classes="output-text",
                            container=False,
                        )

                    # Additional info outputs (hidden by default)
                    viz_info_output = gr.Textbox(
                        label="Visualization Info",
                        lines=5,
                        visible=False,
                        interactive=False,
                    )

                    file_info_output = gr.Textbox(
                        label="File Info",
                        lines=3,
                        visible=False,
                        interactive=False,
                    )

            # Toggle visibility of additional outputs
            def toggle_additional_outputs(enable_viz):
                return {
                    viz_info_output: gr.update(visible=enable_viz),
                    file_info_output: gr.update(visible=enable_viz),
                }

            enable_visualization.change(
                fn=toggle_additional_outputs,
                inputs=[enable_visualization],
                outputs=[viz_info_output, file_info_output],
            )

            # Examples section
            gr.Markdown("### Examples")
            gr.Examples(
                examples=[
                    # You can add example image paths here if needed
                ],
                inputs=[image_input],
                label="Example Images",
            )

            # Extract button click event
            extract_btn.click(
                fn=extract_text_from_image,
                inputs=[
                    image_input,
                    ocr_type,
                    model_size,
                    test_compress,
                    enable_visualization,
                    save_results,
                ],
                outputs=[text_output, viz_info_output, file_info_output],
            )

        return ocr_interface

    def build_main_interface(self) -> "gr.Blocks":
        if self.model_type == "image":
            if "ocr" in self.model_ability:
                title = f"🔍 Xinference OCR: {self.model_name} 🔍"
            else:
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

                    .output-container {
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 16px;
                        background-color: #f8f9fa;
                        margin: 8px 0;
                    }

                    .output-text {
                        background-color: white;
                        border: 1px solid #dee2e6;
                        border-radius: 6px;
                        padding: 16px;
                        min-height: 200px;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                        line-height: 1.6;
                    }

                    .output-text h1, .output-text h2, .output-text h3,
                    .output-text h4, .output-text h5, .output-text h6 {
                        margin-top: 0.5em !important;
                        margin-bottom: 0.5em !important;
                        color: #2d3748 !important;
                    }

                    .output-text p {
                        margin: 0.5em 0 !important;
                    }

                    .output-text pre {
                        background-color: #f6f8fa !important;
                        border: 1px solid #e9ecef !important;
                        border-radius: 4px !important;
                        padding: 12px !important;
                        margin: 8px 0 !important;
                    }

                    .output-text code {
                        background-color: #e9ecef !important;
                        padding: 2px 4px !important;
                        border-radius: 3px !important;
                        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace !important;
                    }

                    .output-text ul, .output-text ol {
                        margin: 0.5em 0 !important;
                        padding-left: 20px !important;
                    }

                    .output-text blockquote {
                        border-left: 4px solid #6c757d !important;
                        padding-left: 16px !important;
                        margin: 0.5em 0 !important;
                        color: #6c757d !important;
                        background-color: #f8f9fa !important;
                    }

                    .output-text table {
                        border-collapse: collapse !important;
                        width: 100% !important;
                        margin: 8px 0 !important;
                    }

                    .output-text th, .output-text td {
                        border: 1px solid #dee2e6 !important;
                        padding: 8px 12px !important;
                        text-align: left !important;
                    }

                    .output-text th {
                        background-color: #f8f9fa !important;
                        font-weight: bold !important;
                    }

                    /* Ensure Markdown displays correctly */
                    .output-text .katex-display {
                        display: block !important;
                        text-align: center !important;
                        margin: 1em 0 !important;
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
            if "ocr" in self.model_ability:
                with gr.Tab("OCR"):
                    self.ocr_interface()
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
