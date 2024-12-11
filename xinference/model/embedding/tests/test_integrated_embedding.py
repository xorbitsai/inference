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


def test_sparse_embedding(setup):
    endpoint, _ = setup
    client = Client(endpoint)

    model_uid = client.launch_model(
        model_name="bge-m3", model_type="embedding", hybrid_mode=True
    )
    assert len(client.list_models()) == 1

    model = client.get_model(model_uid)

    result = model.create_embedding("What is BGE M3?", return_sparse=True)
    emb = result["data"][0]["embedding"]
    token_ids = []
    values = []
    for token_id, v in emb.items():
        token_ids.append(token_id)
        values.append(v)
    words = model.convert_ids_to_tokens(token_ids)
    assert len(words) == len(token_ids)
    assert isinstance(words[0], str)
