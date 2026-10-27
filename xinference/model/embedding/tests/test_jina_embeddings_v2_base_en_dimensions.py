# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

# CI: linux
from ..embed_family import BUILTIN_EMBEDDING_MODELS


def test_jina_embeddings_v2_base_en_dimensions_is_768():
    # jinaai/jina-embeddings-v2-base-en config.json at the pinned revision
    # 7302ac470bed880590f9344bfeee32ff8722d0e5 sets hidden_size 768 and the
    # sentence-transformers head is plain mean pooling, so each vector has
    # 768 values. That revision's 1_Pooling/config.json says 512; upstream
    # corrected it to 768 in 7aef14b0840b7dded6c7e4ce28ff87f16071284d
    # ("fix: emb dim sentence transformers config").
    from .. import _install

    _install()
    family = BUILTIN_EMBEDDING_MODELS["jina-embeddings-v2-base-en"][0]
    assert family.dimensions == 768
