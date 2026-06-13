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

import logging
import os
from typing import Any, Dict, List, Optional

import gradio as gr
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingInterface:
    def __init__(
        self,
        endpoint: str,
        model_uid: str,
        model_name: str,
        model_family: str,
        model_id: str,
        model_revision: Optional[str],
        model_ability: List[str],
        model_type: str,
        access_token: Optional[str],
    ):
        self.endpoint = endpoint
        self.model_uid = model_uid
        self.model_name = model_name
        self.model_family = model_family
        self.model_id = model_id
        self.model_revision = model_revision
        self.model_ability = model_ability
        self.model_type = model_type
        self.access_token = (
            access_token.replace("Bearer ", "") if access_token is not None else None
        )

    def build(self) -> gr.Blocks:
        interface = self.build_main_interface()
        interface.queue()
        try:
            interface.run_startup_events()
        except AttributeError:
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

    def build_main_interface(self) -> "gr.Blocks":
        from ...client.restful.restful_client import RESTfulEmbeddingModelHandle

        def cosine_similarity(a: List[float], b: List[float]) -> float:
            """Calculate cosine similarity between two vectors."""
            a_arr = np.array(a)
            b_arr = np.array(b)
            norm_a = np.linalg.norm(a_arr)
            norm_b = np.linalg.norm(b_arr)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

        def encode_text(
            text: str,
            task: Optional[str],
            dimensions: Optional[int],
            progress=gr.Progress(),
        ) -> Dict[str, Any]:
            """Encode text and return embedding vector + metadata."""
            from ...client import RESTfulClient

            if not text or not text.strip():
                return {"embedding": None, "dim": 0, "error": "Please enter some text"}

            progress(0.1, desc="Connecting to model...")
            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulEmbeddingModelHandle)

            progress(0.5, desc="Encoding text...")
            kwargs: Dict[str, Any] = {}
            if task:
                kwargs["task"] = task
            if dimensions:
                kwargs["dimensions"] = dimensions

            try:
                response = model.create_embedding(input=text, **kwargs)
                progress(1.0, desc="Done")
                embedding = response["data"][0]["embedding"]
                dim = len(embedding)
                return {
                    "embedding": embedding,
                    "dim": dim,
                    "usage": response.get("usage", {}),
                    "error": None,
                }
            except Exception as e:
                logger.error(f"Encoding failed: {e}", exc_info=True)
                return {"embedding": None, "dim": 0, "error": str(e)}

        def encode_multimodal(
            input_type: str,
            text: str,
            image_path: str,
            video_path: str,
            audio_path: str,
            task: Optional[str],
            dimensions: Optional[int],
            progress=gr.Progress(),
        ) -> Dict[str, Any]:
            """Encode multimodal input (text/image/video/audio)."""
            from ...client import RESTfulClient

            progress(0.1, desc="Preparing input...")
            client = RESTfulClient(self.endpoint)
            client._set_token(self.access_token)
            model = client.get_model(self.model_uid)
            assert isinstance(model, RESTfulEmbeddingModelHandle)

            # Build input based on type
            input_data: Any
            if input_type == "text":
                if not text or not text.strip():
                    return {
                        "embedding": None,
                        "dim": 0,
                        "error": "Please enter some text",
                    }
                input_data = text
            elif input_type == "image":
                if not image_path:
                    return {
                        "embedding": None,
                        "dim": 0,
                        "error": "Please provide an image path or URL",
                    }
                input_data = {"image": image_path}
            elif input_type == "video":
                if not video_path:
                    return {
                        "embedding": None,
                        "dim": 0,
                        "error": "Please provide a video path or URL",
                    }
                input_data = {"video": video_path}
            elif input_type == "audio":
                if not audio_path:
                    return {
                        "embedding": None,
                        "dim": 0,
                        "error": "Please provide an audio path or URL",
                    }
                input_data = {"audio": audio_path}
            else:
                return {
                    "embedding": None,
                    "dim": 0,
                    "error": f"Unknown input type: {input_type}",
                }

            progress(0.5, desc=f"Encoding {input_type}...")
            kwargs: Dict[str, Any] = {}
            if task:
                kwargs["task"] = task
            if dimensions:
                kwargs["dimensions"] = dimensions

            try:
                response = model.create_embedding(input=input_data, **kwargs)
                progress(1.0, desc="Done")
                embedding = response["data"][0]["embedding"]
                dim = len(embedding)
                return {
                    "embedding": embedding,
                    "dim": dim,
                    "usage": response.get("usage", {}),
                    "error": None,
                }
            except Exception as e:
                logger.error(f"Encoding failed: {e}", exc_info=True)
                return {"embedding": None, "dim": 0, "error": str(e)}

        def compare_embeddings(
            text_a: str,
            text_b: str,
            task: Optional[str],
            dimensions: Optional[int],
            progress=gr.Progress(),
        ) -> Dict[str, Any]:
            """Encode two texts and compute cosine similarity."""
            # Pass a no-op progress callback to sub-calls so the outer progress
            # bar keeps moving forward smoothly instead of jumping back to 0.1
            # when each ``encode_text`` resets it.
            noop_progress = lambda *args, **kwargs: None  # noqa: E731

            progress(0.1, desc="Encoding text A...")
            result_a = encode_text(text_a, task, dimensions, noop_progress)
            if result_a["error"]:
                return {
                    "similarity": None,
                    "error": f"Text A error: {result_a['error']}",
                }

            progress(0.5, desc="Encoding text B...")
            result_b = encode_text(text_b, task, dimensions, noop_progress)
            if result_b["error"]:
                return {
                    "similarity": None,
                    "error": f"Text B error: {result_b['error']}",
                }

            progress(0.9, desc="Computing similarity...")
            sim = cosine_similarity(result_a["embedding"], result_b["embedding"])
            progress(1.0, desc="Done")
            return {
                "similarity": sim,
                "dim_a": result_a["dim"],
                "dim_b": result_b["dim"],
                "error": None,
            }

        # Build Gradio UI
        with gr.Blocks() as embedding_interface:
            gr.Markdown(
                f"""
                # Embedding Model: {self.model_name}

                **Model Family**: {self.model_family} | **Dimensions**: Check output

                Encode text or multimodal inputs into embedding vectors, or compare similarity between texts.
                """
            )

            with gr.Tabs():
                # Tab 1: Text Encoding
                with gr.Tab("Text Encoding"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            text_input = gr.Textbox(
                                label="Input Text",
                                placeholder="Enter text to encode...",
                                lines=5,
                            )
                            with gr.Row():
                                task_dropdown = gr.Dropdown(
                                    choices=[
                                        "retrieval",
                                        "text-matching",
                                        "classification",
                                        "clustering",
                                    ],
                                    value=None,
                                    label="Task (optional)",
                                    info="For jina-embeddings-v5: retrieval / text-matching / classification / clustering",
                                )
                                dim_input = gr.Number(
                                    label="Dimensions (optional)",
                                    value=None,
                                    precision=0,
                                    info="For Matryoshka models: truncate to this dimension",
                                )
                            encode_btn = gr.Button("Encode", variant="primary")

                        with gr.Column(scale=2):
                            output_dim = gr.Number(
                                label="Embedding Dimension", interactive=False
                            )
                            output_usage = gr.JSON(label="Usage Info")
                            output_error = gr.Textbox(
                                label="Error", interactive=False, visible=False
                            )
                            output_embedding = gr.Code(
                                label="Embedding Vector (first 10 dims)",
                                language="json",
                                lines=10,
                            )

                    def encode_text_ui(text, task, dimensions, progress=gr.Progress()):
                        result = encode_text(
                            text,
                            task,
                            int(dimensions) if dimensions else None,
                            progress,
                        )
                        if result["error"]:
                            return (
                                0,
                                {},
                                gr.update(value=result["error"], visible=True),
                                "",
                            )
                        emb = result["embedding"]
                        preview = emb[:10] if len(emb) > 10 else emb
                        return (
                            result["dim"],
                            result.get("usage", {}),
                            gr.update(visible=False),
                            str(preview),
                        )

                    encode_btn.click(
                        encode_text_ui,
                        inputs=[text_input, task_dropdown, dim_input],
                        outputs=[
                            output_dim,
                            output_usage,
                            output_error,
                            output_embedding,
                        ],
                    )

                # Tab 2: Multimodal Encoding (for omni models)
                with gr.Tab("Multimodal Encoding"):
                    gr.Markdown(
                        """
                        Encode image, video, or audio inputs (requires multimodal model like jina-embeddings-v5-omni-*).
                        """
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            input_type_radio = gr.Radio(
                                choices=["text", "image", "video", "audio"],
                                value="text",
                                label="Input Type",
                            )
                            mm_text_input = gr.Textbox(
                                label="Text Input",
                                placeholder="Enter text...",
                                lines=3,
                                visible=True,
                            )
                            mm_image_input = gr.Textbox(
                                label="Image Path or URL",
                                placeholder="/path/to/image.jpg or https://...",
                                visible=False,
                            )
                            mm_video_input = gr.Textbox(
                                label="Video Path or URL",
                                placeholder="/path/to/video.mp4 or https://...",
                                visible=False,
                            )
                            mm_audio_input = gr.Textbox(
                                label="Audio Path or URL",
                                placeholder="/path/to/audio.wav or https://...",
                                visible=False,
                            )
                            with gr.Row():
                                mm_task_dropdown = gr.Dropdown(
                                    choices=[
                                        "retrieval",
                                        "text-matching",
                                        "classification",
                                        "clustering",
                                    ],
                                    value=None,
                                    label="Task (optional)",
                                )
                                mm_dim_input = gr.Number(
                                    label="Dimensions (optional)",
                                    value=None,
                                    precision=0,
                                )
                            mm_encode_btn = gr.Button("Encode", variant="primary")

                        with gr.Column(scale=2):
                            mm_output_dim = gr.Number(
                                label="Embedding Dimension", interactive=False
                            )
                            mm_output_usage = gr.JSON(label="Usage Info")
                            mm_output_error = gr.Textbox(
                                label="Error", interactive=False, visible=False
                            )
                            mm_output_embedding = gr.Code(
                                label="Embedding Vector (first 10 dims)",
                                language="json",
                                lines=10,
                            )

                    def toggle_input_visibility(input_type):
                        return (
                            gr.update(visible=(input_type == "text")),
                            gr.update(visible=(input_type == "image")),
                            gr.update(visible=(input_type == "video")),
                            gr.update(visible=(input_type == "audio")),
                        )

                    input_type_radio.change(
                        toggle_input_visibility,
                        inputs=[input_type_radio],
                        outputs=[
                            mm_text_input,
                            mm_image_input,
                            mm_video_input,
                            mm_audio_input,
                        ],
                    )

                    def encode_multimodal_ui(
                        input_type,
                        text,
                        image_path,
                        video_path,
                        audio_path,
                        task,
                        dimensions,
                        progress=gr.Progress(),
                    ):
                        result = encode_multimodal(
                            input_type,
                            text,
                            image_path,
                            video_path,
                            audio_path,
                            task,
                            int(dimensions) if dimensions else None,
                            progress,
                        )
                        if result["error"]:
                            return (
                                0,
                                {},
                                gr.update(value=result["error"], visible=True),
                                "",
                            )
                        emb = result["embedding"]
                        preview = emb[:10] if len(emb) > 10 else emb
                        return (
                            result["dim"],
                            result.get("usage", {}),
                            gr.update(visible=False),
                            str(preview),
                        )

                    mm_encode_btn.click(
                        encode_multimodal_ui,
                        inputs=[
                            input_type_radio,
                            mm_text_input,
                            mm_image_input,
                            mm_video_input,
                            mm_audio_input,
                            mm_task_dropdown,
                            mm_dim_input,
                        ],
                        outputs=[
                            mm_output_dim,
                            mm_output_usage,
                            mm_output_error,
                            mm_output_embedding,
                        ],
                    )

                # Tab 3: Similarity Comparison
                with gr.Tab("Similarity Comparison"):
                    gr.Markdown("Compare cosine similarity between two texts.")
                    with gr.Row():
                        with gr.Column():
                            sim_text_a = gr.Textbox(
                                label="Text A",
                                placeholder="Enter first text...",
                                lines=3,
                            )
                        with gr.Column():
                            sim_text_b = gr.Textbox(
                                label="Text B",
                                placeholder="Enter second text...",
                                lines=3,
                            )
                    with gr.Row():
                        sim_task_dropdown = gr.Dropdown(
                            choices=[
                                "retrieval",
                                "text-matching",
                                "classification",
                                "clustering",
                            ],
                            value=None,
                            label="Task (optional)",
                        )
                        sim_dim_input = gr.Number(
                            label="Dimensions (optional)",
                            value=None,
                            precision=0,
                        )
                    compare_btn = gr.Button("Compare", variant="primary")

                    with gr.Row():
                        sim_output = gr.Number(
                            label="Cosine Similarity", interactive=False
                        )
                        sim_dim_a = gr.Number(label="Dim A", interactive=False)
                        sim_dim_b = gr.Number(label="Dim B", interactive=False)
                        sim_error = gr.Textbox(
                            label="Error", interactive=False, visible=False
                        )

                    def compare_ui(
                        text_a, text_b, task, dimensions, progress=gr.Progress()
                    ):
                        result = compare_embeddings(
                            text_a,
                            text_b,
                            task,
                            int(dimensions) if dimensions else None,
                            progress,
                        )
                        if result["error"]:
                            return (
                                None,
                                0,
                                0,
                                gr.update(value=result["error"], visible=True),
                            )
                        return (
                            result["similarity"],
                            result["dim_a"],
                            result["dim_b"],
                            gr.update(visible=False),
                        )

                    compare_btn.click(
                        compare_ui,
                        inputs=[
                            sim_text_a,
                            sim_text_b,
                            sim_task_dropdown,
                            sim_dim_input,
                        ],
                        outputs=[sim_output, sim_dim_a, sim_dim_b, sim_error],
                    )

        return embedding_interface
