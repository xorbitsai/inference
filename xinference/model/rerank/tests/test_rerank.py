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

from ....client import Client


@pytest.mark.parametrize("model_name", ["bge-reranker-v2-m3", "bge-reranker-base"])
@pytest.mark.parametrize("model_engine", ["sentence_transformers", "vllm"])
def test_restful_api(model_name, model_engine, setup):
    if model_name == "bge-reranker-base" and model_engine == "vllm":
        pytest.skip("bge-reranker-base exceeds the max_model_len( 560 > 512 ) of vllm")
    if model_engine == "vllm":
        pytest.importorskip("vllm", reason="vllm is not installed")

    # Skip network-intensive tests on CI to avoid timeout issues
    import os

    if os.environ.get("CI") and model_engine == "sentence_transformers":
        pytest.skip("Skip network-intensive rerank test on CI to avoid timeout")
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name=model_name,
        model_type="rerank",
        model_engine=model_engine,
    )
    assert len(client.list_models()) == 1
    model = client.get_model(model_uid)
    # We want to compute the similarity between the query sentence
    query = "A man is eating pasta."

    # With all sentences in the corpus
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]

    scores = model.rerank(corpus, query, return_documents=True)
    assert scores["results"][0]["index"] == 0
    assert scores["results"][0]["document"]["text"] == corpus[0]

    scores = model.rerank(corpus, query, top_n=3, return_documents=True)
    assert len(scores["results"]) == 3
    assert scores["results"][0]["index"] == 0
    assert scores["results"][0]["document"]["text"] == corpus[0]

    scores = model.rerank(corpus, query, return_len=True)
    assert (
        scores["meta"]["tokens"]["input_tokens"]
        == scores["meta"]["tokens"]["output_tokens"]
    )

    scores = model.rerank(corpus, query)
    assert scores["meta"]["tokens"] is None

    # testing long input
    corpus2 = corpus.copy()
    corpus2[-1] = corpus2[-1] * 50
    scores = model.rerank(corpus2, query, top_n=3, return_documents=True)
    assert len(scores["results"]) == 3
    assert scores["results"][0]["index"] == 0
    assert scores["results"][0]["document"]["text"] == corpus2[0]

    kwargs = {
        "invalid": "invalid",
    }
    with pytest.raises(RuntimeError):
        model.rerank(corpus, query, **kwargs)


def test_from_local_uri():
    from ..cache_manager import RerankCacheManager
    from ..core import TransformersRerankSpecV1
    from ..custom import CustomRerankModelFamilyV2

    tmp_dir = tempfile.mkdtemp()

    model_family = CustomRerankModelFamilyV2(
        model_name="custom_test_rerank_a",
        max_tokens=2048,
        language=["zh"],
        model_specs=[
            TransformersRerankSpecV1(
                model_format="pytorch",
                model_id="test/custom_test_a",
                model_uri=os.path.abspath(tmp_dir),
                quantization="none",
            )
        ],
    )

    cache_dir = RerankCacheManager(model_family=model_family).cache()
    assert os.path.exists(cache_dir)
    assert os.path.islink(cache_dir)
    os.remove(cache_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_register_custom_rerank():
    from ....constants import XINFERENCE_CACHE_DIR
    from ..cache_manager import RerankCacheManager
    from ..core import TransformersRerankSpecV1
    from ..custom import CustomRerankModelFamilyV2, register_rerank, unregister_rerank

    tmp_dir = tempfile.mkdtemp()

    # correct
    model_family = CustomRerankModelFamilyV2(
        model_name="custom_test_rerank_b",
        max_tokens=2048,
        language=["zh"],
        model_specs=[
            TransformersRerankSpecV1(
                model_format="pytorch",
                model_id="test/custom_test_b",
                model_uri=os.path.abspath(tmp_dir),
                quantization="none",
            )
        ],
    )

    register_rerank(model_family, False)
    RerankCacheManager(model_family=model_family).cache()
    model_cache_path = os.path.join(
        XINFERENCE_CACHE_DIR, "v2/" + model_family.model_name + "-pytorch-none"
    )
    assert os.path.exists(model_cache_path)
    assert os.path.islink(model_cache_path)
    os.remove(model_cache_path)

    # Invalid path
    model_family = CustomRerankModelFamilyV2(
        model_name="custom_test_rerank_c",
        max_tokens=2048,
        language=["zh"],
        model_specs=[
            TransformersRerankSpecV1(
                model_format="pytorch",
                model_id="test/custom_test_b",
                model_uri="file:///sssad/faf",
                quantization="none",
            )
        ],
    )
    with pytest.raises(ValueError):
        register_rerank(model_family, False)

    # name conflict
    model_family = CustomRerankModelFamilyV2(
        model_name="custom_test_rerank_c",
        max_tokens=2048,
        language=["zh"],
        model_specs=[
            TransformersRerankSpecV1(
                model_format="pytorch",
                model_id="test/custom_test_b",
                model_uri=os.path.abspath(tmp_dir),
                quantization="none",
            )
        ],
    )
    register_rerank(model_family, False)
    with pytest.raises(ValueError):
        register_rerank(model_family, False)

    # unregister
    unregister_rerank("custom_test_rerank_b")
    unregister_rerank("custom_test_rerank_c")
    with pytest.raises(ValueError):
        unregister_rerank("custom_test_rerank_d")

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_auto_detect_type():
    from ..core import RerankModel

    rerank_model_json = os.path.join(os.path.dirname(__file__), "../model_spec.json")
    with open(rerank_model_json, "r") as f:
        rerank_models = json.load(f)
    for m in rerank_models:
        if m["model_name"] == "minicpm-reranker":
            # TODO: we need to fix the auto detect type
            continue
        try:
            assert m["type"] == RerankModel._auto_detect_type(
                m["model_specs"][0]["model_src"]["huggingface"]["model_id"]
            )
        except EnvironmentError:
            # gated repo, ignore
            continue
