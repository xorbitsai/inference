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

import json
import os
import shutil
import tempfile

import pytest

from ..cache_manager import EmbeddingCacheManager as CacheManager
from ..core import EmbeddingModelFamilyV2, TransformersEmbeddingSpecV1

TEST_MODEL_SPEC = EmbeddingModelFamilyV2(
    version=2,
    model_name="gte-small",
    dimensions=384,
    max_tokens=512,
    language=["en"],
    model_specs=[
        TransformersEmbeddingSpecV1(
            model_format="pytorch",
            model_id="thenlper/gte-small",
            model_revision="d8e2604cadbeeda029847d19759d219e0ce2e6d8",
            quantization="none",
        )
    ],
)

TEST_MODEL_SPEC2 = EmbeddingModelFamilyV2(
    version=2,
    model_name="gte-small",
    dimensions=384,
    max_tokens=512,
    language=["en"],
    model_specs=[
        TransformersEmbeddingSpecV1(
            model_format="pytorch",
            model_id="thenlper/gte-small",
            model_revision="c20abe89ac0cdf484944ebdc26ecaaa1bfc9cf89",
            quantization="none",
        )
    ],
)

TEST_MODEL_SPEC_FROM_MODELSCOPE = EmbeddingModelFamilyV2(
    version=2,
    model_name="bge-small-zh-v1.5",
    dimensions=512,
    max_tokens=512,
    language=["zh"],
    model_specs=[
        TransformersEmbeddingSpecV1(
            model_format="pytorch",
            model_id="Xorbits/bge-small-zh-v1.5",
            model_revision="v0.0.2",
            quantization="none",
            model_hub="modelscope",
        )
    ],
)
from ..embed_family import EMBEDDING_ENGINES


def test_engine_supported():
    model_name = "bge-small-en-v1.5"
    assert model_name in EMBEDDING_ENGINES
    assert "flag" in EMBEDDING_ENGINES[model_name]
    assert "sentence_transformers" in EMBEDDING_ENGINES[model_name]


async def test_model_from_modelscope():
    from ..core import create_embedding_model_instance

    model_path = CacheManager(TEST_MODEL_SPEC_FROM_MODELSCOPE).cache()
    model = create_embedding_model_instance(
        "mock",
        "bge-small-zh-v1.5",
        "sentence_transformers",
        model_path=model_path,
    )
    # input is a string
    input_text = "乱条犹未变初黄，倚得东风势便狂。解把飞花蒙日月，不知天地有清霜。"
    model.load()
    r = await model.create_embedding(input_text)
    assert len(r["data"]) == 1
    for d in r["data"]:
        assert len(d["embedding"]) == 512
    shutil.rmtree(model_path, ignore_errors=True)


def test_get_cache_status():
    model_path = None
    try:
        cache_manager = CacheManager(TEST_MODEL_SPEC)
        assert cache_manager.get_cache_status() is False
        model_path = cache_manager.cache()
        assert cache_manager.get_cache_status() is True
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


def test_from_local_uri():
    from ..custom import CustomEmbeddingModelFamilyV2

    tmp_dir = tempfile.mkdtemp()

    model_family = CustomEmbeddingModelFamilyV2(
        model_name="custom_test_a",
        dimensions=1024,
        max_tokens=2048,
        language=["zh"],
        model_specs=[
            TransformersEmbeddingSpecV1(
                model_format="pytorch",
                model_id="test/custom_test_a",
                model_uri=os.path.abspath(tmp_dir),
                quantization="none",
            )
        ],
    )

    cache_dir = CacheManager(model_family).cache()
    assert os.path.exists(cache_dir)
    assert os.path.islink(cache_dir)
    assert os.path.samefile(os.path.realpath(cache_dir), tmp_dir)
    os.remove(cache_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_register_custom_embedding():
    from ....constants import XINFERENCE_CACHE_DIR
    from ..custom import (
        CustomEmbeddingModelFamilyV2,
        register_embedding,
        unregister_embedding,
    )

    tmp_dir = tempfile.mkdtemp()

    # correct
    model_family = CustomEmbeddingModelFamilyV2(
        model_name="custom_test_b",
        dimensions=1024,
        max_tokens=2048,
        language=["zh"],
        model_specs=[
            TransformersEmbeddingSpecV1(
                model_format="pytorch",
                model_id="test/custom_test_b",
                model_uri=os.path.abspath(tmp_dir),
                quantization="none",
            )
        ],
    )

    register_embedding(model_family, False)
    CacheManager(model_family).cache()
    model_cache_path = os.path.join(
        XINFERENCE_CACHE_DIR, "v2", f"{model_family.model_name}-pytorch-none"
    )
    assert os.path.exists(model_cache_path)
    assert os.path.islink(model_cache_path)
    os.remove(model_cache_path)

    # Invalid path
    model_family = CustomEmbeddingModelFamilyV2(
        model_name="custom_test_b-v15",
        dimensions=1024,
        max_tokens=2048,
        language=["zh"],
        model_specs=[
            TransformersEmbeddingSpecV1(
                model_format="pytorch",
                model_id="test/custom_test_b",
                model_uri="file:///c/d",
                quantization="none",
            )
        ],
    )
    with pytest.raises(ValueError):
        register_embedding(model_family, False)

    # name conflict
    model_family = CustomEmbeddingModelFamilyV2(
        model_name="custom_test_c",
        dimensions=1024,
        max_tokens=2048,
        language=["zh"],
        model_specs=[
            TransformersEmbeddingSpecV1(
                model_format="pytorch",
                model_id="test/custom_test_c",
                model_uri=os.path.abspath(tmp_dir),
                quantization="none",
            )
        ],
    )
    register_embedding(model_family, False)
    with pytest.raises(ValueError):
        register_embedding(model_family, False)

    # unregister
    unregister_embedding("custom_test_b")
    unregister_embedding("custom_test_c")
    with pytest.raises(ValueError):
        unregister_embedding("custom_test_d")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_register_fault_embedding():
    import warnings

    from ....constants import XINFERENCE_MODEL_DIR
    from .. import _install

    embedding_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "embedding")

    os.makedirs(embedding_dir, exist_ok=True)
    file_path = os.path.join(embedding_dir, "GTE.json")

    data = {
        "model_name": "GTE",
        "model_hub": "huggingface",
        "dimensions": 768,
        "max_tokens": 512,
        "language": ["en", "zh"],
        "model_specs": [
            {
                "model_format": "pytorch",
                "model_id": None,
                "model_revision": None,
                "model_uri": "/new_data/cache/gte-Qwen2",
                "quantization": "none",
            }
        ],
    }

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    all_warnings = []

    def custom_warning_handler(
        message, category, filename, lineno, file=None, line=None
    ):
        warning_info = {
            "message": str(message),
            "category": category.__name__,
            "filename": filename,
            "lineno": lineno,
        }
        all_warnings.append(warning_info)

    old_showwarning = warnings.showwarning
    warnings.showwarning = custom_warning_handler

    try:
        _install()

        warnings.showwarning = old_showwarning

        with pytest.warns(UserWarning) as record:
            _install()

        found_warning = False
        for warning in record:
            message = str(warning.message)
            if (
                "has error" in message
                and (
                    "Invalid model URI" in message
                    or "Model URI cannot be a relative path" in message
                )
                and "/new_data/cache/gte-Qwen2" in message
            ):
                found_warning = True
                break

        assert (
            found_warning
        ), f"Expected warning about invalid model URI not found. Warnings: {[str(w.message) for w in record]}"

    finally:
        warnings.showwarning = old_showwarning

    if os.path.exists(file_path):
        os.remove(file_path)


def test_convert_ids_to_tokens():
    from ..core import create_embedding_model_instance

    model_path = CacheManager(TEST_MODEL_SPEC_FROM_MODELSCOPE).cache()
    model = create_embedding_model_instance(
        "mock",
        "bge-small-zh-v1.5",
        "sentence_transformers",
        model_path=model_path,
    )
    model.load()

    ids = [[8074, 8059, 8064, 8056], [144, 147, 160, 160, 158]]
    tokens = model.convert_ids_to_tokens(ids)

    assert isinstance(tokens, list)
    assert tokens == [["ｘ", "ｉ", "ｎ", "ｆ"], ["b", "e", "r", "r", "p"]]

    shutil.rmtree(model_path, ignore_errors=True)
