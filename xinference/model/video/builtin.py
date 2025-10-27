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
from typing import TYPE_CHECKING, List

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .custom import CustomVideoModelFamilyV2


class BuiltinVideoModelRegistry:
    """
    Registry for built-in video models downloaded from official model hub.

    These models are treated as built-in models and don't require model_family validation.
    They are stored in ~/.xinference/model/v2/builtin/video/ directory.
    """

    def __init__(self):
        from ...constants import XINFERENCE_MODEL_DIR

        self.builtin_dir = os.path.join(XINFERENCE_MODEL_DIR, "v2", "builtin", "video")
        os.makedirs(self.builtin_dir, exist_ok=True)

    def get_builtin_models(self) -> List["CustomVideoModelFamilyV2"]:
        """Load all built-in video models from the builtin directory."""
        from .custom import CustomVideoModelFamilyV2

        models: List["CustomVideoModelFamilyV2"] = []

        if not os.path.exists(self.builtin_dir):
            return models

        for filename in os.listdir(self.builtin_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.builtin_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        model_data = json.load(f)

                    # Apply conversion logic to handle null model_id and other issues
                    if model_data.get("model_id") is None and "model_src" in model_data:
                        model_src = model_data["model_src"]
                        # Extract model_id from available sources
                        if (
                            "huggingface" in model_src
                            and "model_id" in model_src["huggingface"]
                        ):
                            model_data["model_id"] = model_src["huggingface"][
                                "model_id"
                            ]
                        elif (
                            "modelscope" in model_src
                            and "model_id" in model_src["modelscope"]
                        ):
                            model_data["model_id"] = model_src["modelscope"]["model_id"]

                        # Extract model_revision if available
                        if model_data.get("model_revision") is None:
                            if (
                                "huggingface" in model_src
                                and "model_revision" in model_src["huggingface"]
                            ):
                                model_data["model_revision"] = model_src["huggingface"][
                                    "model_revision"
                                ]
                            elif (
                                "modelscope" in model_src
                                and "model_revision" in model_src["modelscope"]
                            ):
                                model_data["model_revision"] = model_src["modelscope"][
                                    "model_revision"
                                ]

                    # Parse using CustomVideoModelFamilyV2
                    model = CustomVideoModelFamilyV2.parse_obj(model_data)
                    models.append(model)
                    logger.info(f"Loaded built-in video model: {model.model_name}")

                except Exception as e:
                    logger.warning(
                        f"Failed to load built-in model from {filename}: {e}"
                    )

        return models

    def register_builtin_model(self, model) -> None:
        """Register a built-in video model by saving it to the builtin directory."""
        persist_path = os.path.join(self.builtin_dir, f"{model.model_name}.json")

        try:
            with open(persist_path, "w", encoding="utf-8") as f:
                f.write(model.json(exclude_none=True))
            logger.info(f"Registered built-in video model: {model.model_name}")
        except Exception as e:
            logger.error(f"Failed to register built-in model {model.model_name}: {e}")
            raise

    def unregister_builtin_model(self, model_name: str) -> None:
        """Unregister a built-in video model by removing its JSON file."""
        persist_path = os.path.join(self.builtin_dir, f"{model_name}.json")

        if os.path.exists(persist_path):
            try:
                os.remove(persist_path)
                logger.info(f"Unregistered built-in video model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to unregister built-in model {model_name}: {e}")
                raise
        else:
            logger.warning(
                f"Built-in video model {model_name} not found for unregistration"
            )
