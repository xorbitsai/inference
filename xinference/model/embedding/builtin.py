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

import json
import logging
import os
from typing import List

logger = logging.getLogger(__name__)


class BuiltinEmbeddingModelRegistry:
    """
    Registry for built-in embedding models downloaded from official model hub.

    These models are treated as built-in models.
    They are stored in ~/.xinference/model/v2/builtin/embedding/ directory.
    """

    def __init__(self):
        from ...constants import XINFERENCE_MODEL_DIR

        self.builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "embedding")
        os.makedirs(self.builtin_dir, exist_ok=True)

    def get_builtin_models(self) -> List:
        """Load all built-in embedding models from the builtin directory."""
        from .custom import EmbeddingModelFamilyV2

        models = []

        if not os.path.exists(self.builtin_dir):
            return models

        for filename in os.listdir(self.builtin_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.builtin_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        model_data = json.load(f)

                    # Parse using EmbeddingFamilyV2
                    model = EmbeddingModelFamilyV2.parse_obj(model_data)
                    models.append(model)
                    logger.info(f"Loaded built-in embedding model: {model.model_name}")

                except Exception as e:
                    logger.warning(f"Failed to load built-in model from {filename}: {e}")

        return models

    def register_builtin_model(self, model) -> None:
        """Register a built-in embedding model by saving it to the builtin directory."""
        persist_path = os.path.join(self.builtin_dir, f"{model.model_name}.json")

        try:
            with open(persist_path, "w", encoding="utf-8") as f:
                f.write(model.json(exclude_none=True))
            logger.info(f"Registered built-in embedding model: {model.model_name}")
        except Exception as e:
            logger.error(f"Failed to register built-in model {model.model_name}: {e}")
            raise

    def unregister_builtin_model(self, model_name: str) -> None:
        """Unregister a built-in embedding model by removing its JSON file."""
        persist_path = os.path.join(self.builtin_dir, f"{model_name}.json")

        if os.path.exists(persist_path):
            os.remove(persist_path)
            logger.info(f"Unregistered built-in embedding model: {model_name}")
        else:
            logger.warning(f"Built-in model file not found: {persist_path}")


# Global registry instance
_builtin_registry = None


def get_builtin_embedding_registry() -> BuiltinEmbeddingModelRegistry:
    """Get the global built-in embedding model registry instance."""
    global _builtin_registry
    if _builtin_registry is None:
        _builtin_registry = BuiltinEmbeddingModelRegistry()
    return _builtin_registry


def get_builtin_embedding_families() -> List:
    """Get all built-in embedding model families."""
    return get_builtin_embedding_registry().get_builtin_models()


def register_builtin_embedding(embedding_family) -> None:
    """Register a built-in embedding model family."""
    return get_builtin_embedding_registry().register_builtin_model(embedding_family)


def unregister_builtin_embedding(model_name: str) -> None:
    """Unregister a built-in embedding model family."""
    return get_builtin_embedding_registry().unregister_builtin_model(model_name)