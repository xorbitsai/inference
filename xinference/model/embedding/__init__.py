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

import codecs
import json
import os

from .core import MODEL_NAME_TO_REVISION, EmbeddingModelSpec, get_cache_status
from .custom import (
    EMBEDDING_LAUNCH_VERSIONS,
    CustomEmbeddingModelSpec,
    get_embedding_launch_versions,
    get_user_defined_embeddings,
    register_embedding,
    unregister_embedding,
)
from .utils import get_launch_version

_model_spec_json = os.path.join(os.path.dirname(__file__), "model_spec.json")
_model_spec_modelscope_json = os.path.join(
    os.path.dirname(__file__), "model_spec_modelscope.json"
)
BUILTIN_EMBEDDING_MODELS = dict(
    (spec["model_name"], EmbeddingModelSpec(**spec))
    for spec in json.load(codecs.open(_model_spec_json, "r", encoding="utf-8"))
)
for model_name, model_spec in BUILTIN_EMBEDDING_MODELS.items():
    MODEL_NAME_TO_REVISION[model_name].append(model_spec.model_revision)
    # register launch version
    EMBEDDING_LAUNCH_VERSIONS.update(get_launch_version(model_spec))
MODELSCOPE_EMBEDDING_MODELS = dict(
    (spec["model_name"], EmbeddingModelSpec(**spec))
    for spec in json.load(
        codecs.open(_model_spec_modelscope_json, "r", encoding="utf-8")
    )
)
for model_name, model_spec in MODELSCOPE_EMBEDDING_MODELS.items():
    MODEL_NAME_TO_REVISION[model_name].append(model_spec.model_revision)
    # register launch version
    if model_spec.model_name not in EMBEDDING_LAUNCH_VERSIONS:
        EMBEDDING_LAUNCH_VERSIONS.update(get_launch_version(model_spec))

from ...constants import XINFERENCE_MODEL_DIR

user_defined_llm_dir = os.path.join(XINFERENCE_MODEL_DIR, "embedding")
if os.path.isdir(user_defined_llm_dir):
    for f in os.listdir(user_defined_llm_dir):
        with codecs.open(os.path.join(user_defined_llm_dir, f), encoding="utf-8") as fd:
            user_defined_llm_family = CustomEmbeddingModelSpec.parse_obj(json.load(fd))
            register_embedding(user_defined_llm_family, persist=False)

for ud_embedding in get_user_defined_embeddings():
    # register launch version
    EMBEDDING_LAUNCH_VERSIONS.update(get_launch_version(ud_embedding))

del _model_spec_json
del _model_spec_modelscope_json
