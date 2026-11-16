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

"""MiniMax-H3-Image through the Diffusers modular pipeline."""

from .modular import ModularDiffusionModel


class MiniMaxH3ImageModel(ModularDiffusionModel):
    _display_name = _component_label = "MiniMax-H3-Image"
    _pipeline_class_name = "MiniMaxH3ImageModularPipeline"
    _diffusers_requirement = "diffusers>=0.43.0"
    _max_reference_images = 16
    # Both hubs publish the same index, whose component repository IDs use the
    # Hugging Face namespace. Resolve them to the selected hub's local snapshot.
    _component_repositories = (
        "MiniMaxAI/MiniMax-H3-Image",
        "MiniMax/MiniMax-H3-Image",
    )
