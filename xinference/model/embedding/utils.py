# Copyright 2022-2024 XProbe Inc.
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
from logging import getLogger

from .core import EmbeddingModelSpec


def get_model_version(embedding_model: EmbeddingModelSpec) -> str:
    return f"{embedding_model.model_name}--{embedding_model.max_tokens}--{embedding_model.dimensions}"


def get_language_from_model_id(model_id: str) -> str:
    split = model_id.split("/")
    if len(split) != 2:
        logger = getLogger(__name__)
        logger.error(f"Invalid model_id: {model_id}, return the default en language")
        return "en"
    model_id = split[-1]
    segments = model_id.split("-")
    for seg in segments:
        if seg.lower() in ["zh", "cn", "chinese", "multilingual"]:
            return "zh"
    return "en"
