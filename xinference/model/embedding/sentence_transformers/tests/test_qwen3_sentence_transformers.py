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

import os
import shutil
import tempfile
import inspect

import pytest
import torch
from PIL import Image

from ...cache_manager import EmbeddingCacheManager as CacheManager
from ...core import (
    EmbeddingModelFamilyV2,
    TransformersEmbeddingSpecV1,
    create_embedding_model_instance,
)
from ...custom import (
    CustomEmbeddingModelFamilyV2,
    register_embedding,
    unregister_embedding,
)
from ...embed_family import BUILTIN_EMBEDDING_MODELS

QWEN3_EMBEDDING_SPEC = EmbeddingModelFamilyV2(
    version=2,
    model_name="Qwen3-Embedding-0.6B",
    dimensions=1024,
    max_tokens=32768,
    language=["zh", "en"],
    model_specs=[
        TransformersEmbeddingSpecV1(
            model_format="pytorch",
            model_id="Qwen/Qwen3-Embedding-0.6B",
            quantization="none",
        )
    ],
)

QWEN3_VL_EMBEDDING_SPEC = EmbeddingModelFamilyV2(
    version=2,
    model_name="Qwen3-VL-Embedding-2B",
    dimensions=4096,
    max_tokens=8192,
    language=["zh", "en"],
    model_specs=[
        TransformersEmbeddingSpecV1(
            model_format="pytorch",
            model_id="Qwen/Qwen3-VL-Embedding-2B",
            quantization="none",
        )
    ],
)


def _should_skip_gpu_ci_only() -> bool:
    return (
        os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("MODULE") != "gpu"
        or not torch.cuda.is_available()
    )


@pytest.mark.skipif(_should_skip_gpu_ci_only(), reason="Run only on GitHub GPU CI")
async def test_qwen3_embedding_sentence_transformers_smoke():
    model_path = None

    try:
        model_path = CacheManager(QWEN3_EMBEDDING_SPEC).cache()

        model = create_embedding_model_instance(
            "mock",
            "Qwen3-Embedding-0.6B",
            "sentence_transformers",
            model_path=model_path,
        )
        model.load()

        input_texts = ["Hello world.", "The quick brown fox jumps over the lazy dog."]
        result = await model.create_embedding(input_texts)
        assert len(result["data"]) == 2
        for item in result["data"]:
            assert len(item["embedding"]) == 1024
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


@pytest.mark.skipif(_should_skip_gpu_ci_only(), reason="Run only on GitHub GPU CI")
async def test_qwen3_vl_embedding_sentence_transformers_inputs():
    model_path = None
    registered = False
    model_name = "Qwen3-VL-Embedding-2B"

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")
    )
    image_path = os.path.join(repo_root, "examples", "draft.png")
    image_path_2 = os.path.join(repo_root, "assets", "screenshot.png")

    temp_dir = None
    if not (os.path.exists(image_path) and os.path.exists(image_path_2)):
        temp_dir = tempfile.mkdtemp(prefix="qwen3_vl_images_")
        image_path = os.path.join(temp_dir, "image_1.png")
        image_path_2 = os.path.join(temp_dir, "image_2.png")
        Image.new("RGB", (64, 64), color=(255, 0, 0)).save(image_path)
        Image.new("RGB", (64, 64), color=(0, 255, 0)).save(image_path_2)

    try:
        if model_name not in BUILTIN_EMBEDDING_MODELS:
            custom_spec = CustomEmbeddingModelFamilyV2.parse_obj(
                QWEN3_VL_EMBEDDING_SPEC.dict()
            )
            register_embedding(custom_spec, persist=False)
            registered = True

        model_path = CacheManager(QWEN3_VL_EMBEDDING_SPEC).cache()

        model = create_embedding_model_instance(
            "mock",
            model_name,
            "sentence_transformers",
            model_path=model_path,
        )
        model.load()

        text_only = "This is a text-only embedding input."
        result = await model.create_embedding(text_only)
        assert len(result["data"]) == 1
        assert len(result["data"][0]["embedding"]) == 4096

        try:
            import qwen_vl_utils

            process_vision_info = getattr(qwen_vl_utils, "process_vision_info", None)
        except Exception:
            process_vision_info = None

        if process_vision_info is None:
            pytest.skip("qwen_vl_utils.process_vision_info is unavailable")

        signature = inspect.signature(process_vision_info)
        if "image_patch_size" not in signature.parameters:
            pytest.skip(
                "qwen_vl_utils is too old for Qwen3-VL embedding; "
                "please upgrade qwen-vl-utils"
            )

        single_image = {"text": "This is a single image input.", "image": image_path}
        result = await model.create_embedding(single_image)
        assert len(result["data"]) == 1
        assert len(result["data"][0]["embedding"]) == 4096

        multi_image = {
            "text": "This input includes two images.",
            "image": [image_path, image_path_2],
        }
        result = await model.create_embedding(multi_image)
        assert len(result["data"]) == 1
        assert len(result["data"][0]["embedding"]) == 4096
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)
        if registered:
            unregister_embedding(model_name, raise_error=False)
