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

import os
import shutil
import tempfile

import pytest

from ...utils import valid_model_revision
from ..core import EmbeddingModel, EmbeddingModelSpec, cache

TEST_MODEL_SPEC = EmbeddingModelSpec(
    model_name="gte-small",
    dimensions=384,
    max_tokens=512,
    language=["en"],
    model_id="thenlper/gte-small",
    model_revision="d8e2604cadbeeda029847d19759d219e0ce2e6d8",
)

TEST_MODEL_SPEC2 = EmbeddingModelSpec(
    model_name="gte-small",
    dimensions=384,
    max_tokens=512,
    language=["en"],
    model_id="thenlper/gte-small",
    model_revision="c20abe89ac0cdf484944ebdc26ecaaa1bfc9cf89",
)

TEST_MODEL_SPEC_FROM_MODELSCOPE = EmbeddingModelSpec(
    model_name="bge-small-zh-v1.5",
    dimensions=512,
    max_tokens=512,
    language=["zh"],
    model_id="Xorbits/bge-small-zh-v1.5",
    model_revision="v0.0.2",
)


def test_model():
    model_path = None
    try:
        model_path = cache(TEST_MODEL_SPEC)
        model = EmbeddingModel("mock", model_path)
        # input is a string
        input_text = "what is the capital of China?"
        model.load()
        r = model.create_embedding(input_text)
        assert len(r["data"]) == 1
        for d in r["data"]:
            assert len(d["embedding"]) == 384

        # input is a lit
        input_texts = [
            "what is the capital of China?",
            "how to implement quick sort in python?",
            "Beijing",
            "sorting algorithms",
        ]
        model.load()
        r = model.create_embedding(input_texts)
        assert len(r["data"]) == 4
        for d in r["data"]:
            assert len(d["embedding"]) == 384
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


def test_model_from_modelscope():
    model_path = cache(TEST_MODEL_SPEC_FROM_MODELSCOPE)
    model = EmbeddingModel("mock", model_path)
    # input is a string
    input_text = "乱条犹未变初黄，倚得东风势便狂。解把飞花蒙日月，不知天地有清霜。"
    model.load()
    r = model.create_embedding(input_text)
    assert len(r["data"]) == 1
    for d in r["data"]:
        assert len(d["embedding"]) == 512


def test_meta_file():
    cache_dir = None
    try:
        cache_dir = cache(TEST_MODEL_SPEC)
        meta_path = os.path.join(cache_dir, "__valid_download")
        assert valid_model_revision(meta_path, TEST_MODEL_SPEC.model_revision)

        # test another version of the same model
        assert not valid_model_revision(meta_path, TEST_MODEL_SPEC2.model_revision)
        cache_dir = cache(TEST_MODEL_SPEC2)
        meta_path = os.path.join(cache_dir, "__valid_download")
        assert valid_model_revision(meta_path, TEST_MODEL_SPEC2.model_revision)

        # test functionality of the new version model
        model = EmbeddingModel("mock", cache_dir)
        input_text = "I can do this all day."
        model.load()
        r = model.create_embedding(input_text)
        assert len(r["data"]) == 1
        for d in r["data"]:
            assert len(d["embedding"]) == 384
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_get_cache_status():
    from ..core import get_cache_status

    model_path = None
    try:
        assert get_cache_status(TEST_MODEL_SPEC) is False
        model_path = cache(TEST_MODEL_SPEC)
        assert get_cache_status(TEST_MODEL_SPEC) is True
    finally:
        if model_path is not None:
            shutil.rmtree(model_path, ignore_errors=True)


def test_from_local_uri():
    from ..core import cache_from_uri
    from ..custom import CustomEmbeddingModelSpec

    tmp_dir = tempfile.mkdtemp()

    model_spec = CustomEmbeddingModelSpec(
        model_name="custom_test_a",
        dimensions=1024,
        max_tokens=2048,
        language=["zh"],
        model_id="test/custom_test_a",
        model_uri=os.path.abspath(tmp_dir),
    )

    cache_dir = cache_from_uri(model_spec=model_spec)
    assert os.path.exists(cache_dir)
    assert os.path.islink(cache_dir)
    os.remove(cache_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_register_custom_embedding():
    from ....constants import XINFERENCE_CACHE_DIR
    from ..core import cache_from_uri
    from ..custom import (
        CustomEmbeddingModelSpec,
        register_embedding,
        unregister_embedding,
    )

    tmp_dir = tempfile.mkdtemp()

    # correct
    model_spec = CustomEmbeddingModelSpec(
        model_name="custom_test_b",
        dimensions=1024,
        max_tokens=2048,
        language=["zh"],
        model_id="test/custom_test_b",
        model_uri=os.path.abspath(tmp_dir),
    )

    register_embedding(model_spec, False)
    cache_from_uri(model_spec)
    model_cache_path = os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
    assert os.path.exists(model_cache_path)
    assert os.path.islink(model_cache_path)
    os.remove(model_cache_path)

    # Invalid name
    model_spec = CustomEmbeddingModelSpec(
        model_name="custom_test_b-v1.5",
        dimensions=1024,
        max_tokens=2048,
        language=["zh"],
        model_id="test/custom_test_b",
        model_uri=os.path.abspath(tmp_dir),
    )
    with pytest.raises(ValueError):
        register_embedding(model_spec, False)

    # Invalid path
    model_spec = CustomEmbeddingModelSpec(
        model_name="custom_test_b-v15",
        dimensions=1024,
        max_tokens=2048,
        language=["zh"],
        model_id="test/custom_test_b",
        model_uri="file:///c/d",
    )
    register_embedding(model_spec, False)

    # name conflict
    model_spec = CustomEmbeddingModelSpec(
        model_name="custom_test_c",
        dimensions=1024,
        max_tokens=2048,
        language=["zh"],
        model_id="test/custom_test_c",
        model_uri=os.path.abspath(tmp_dir),
    )
    register_embedding(model_spec, False)
    with pytest.raises(ValueError):
        register_embedding(model_spec, False)

    # unregister
    unregister_embedding("custom_test_b")
    unregister_embedding("custom_test_c")
    with pytest.raises(ValueError):
        unregister_embedding("custom_test_d")

    shutil.rmtree(tmp_dir, ignore_errors=True)
