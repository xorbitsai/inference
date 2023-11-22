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

from ....client import Client


def test_oscar_api(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(model_name="bge-reranker-base", model_type="rerank")
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
