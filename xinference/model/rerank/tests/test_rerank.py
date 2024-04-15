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

from ....client import Client


@pytest.mark.parametrize("model_name", ["bge-reranker-base", "bge-reranker-v2-gemma"])
def test_restful_api(model_name, setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(model_name=model_name, model_type="rerank")
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

    scores = model.rerank(corpus, query)
    assert scores["results"][0]["index"] == 0
    assert scores["results"][0]["document"] == corpus[0]

    scores = model.rerank(corpus, query, top_n=3)
    assert len(scores["results"]) == 3
    assert scores["results"][0]["index"] == 0
    assert scores["results"][0]["document"] == corpus[0]

    kwargs = {
        "invalid": "invalid",
    }
    with pytest.raises(RuntimeError) as err:
        scores = model.rerank(corpus, query, **kwargs)
    assert "hasn't support" in str(err.value)


def test_from_local_uri():
    from ...utils import cache_from_uri
    from ..custom import CustomRerankModelSpec

    tmp_dir = tempfile.mkdtemp()

    model_spec = CustomRerankModelSpec(
        model_name="custom_test_rerank_a",
        language=["zh"],
        model_uri=os.path.abspath(tmp_dir),
    )

    cache_dir = cache_from_uri(model_spec=model_spec)
    assert os.path.exists(cache_dir)
    assert os.path.islink(cache_dir)
    os.remove(cache_dir)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_register_custom_rerank():
    from ....constants import XINFERENCE_CACHE_DIR
    from ...utils import cache_from_uri
    from ..custom import CustomRerankModelSpec, register_rerank, unregister_rerank

    tmp_dir = tempfile.mkdtemp()

    # correct
    model_spec = CustomRerankModelSpec(
        model_name="custom_test_b",
        language=["zh"],
        model_uri=os.path.abspath(tmp_dir),
    )

    register_rerank(model_spec, False)
    cache_from_uri(model_spec)
    model_cache_path = os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name)
    assert os.path.exists(model_cache_path)
    assert os.path.islink(model_cache_path)
    os.remove(model_cache_path)

    # Invalid path
    model_spec = CustomRerankModelSpec(
        model_name="custom_test_b-v15",
        language=["zh"],
        model_uri="file:///c/d",
    )
    register_rerank(model_spec, False)

    # name conflict
    model_spec = CustomRerankModelSpec(
        model_name="custom_test_c",
        language=["zh"],
        model_uri=os.path.abspath(tmp_dir),
    )
    register_rerank(model_spec, False)
    with pytest.raises(ValueError):
        register_rerank(model_spec, False)

    # unregister
    unregister_rerank("custom_test_b")
    unregister_rerank("custom_test_c")
    with pytest.raises(ValueError):
        unregister_rerank("custom_test_d")

    shutil.rmtree(tmp_dir, ignore_errors=True)
