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

"""FLUX 3 Image Klein through the Diffusers modular pipeline."""

from .modular import ModularDiffusionModel


class Flux3Model(ModularDiffusionModel):
    _display_name = "FLUX 3 Image Klein"
    _component_label = "FLUX 3"
    _pipeline_class_name = "Flux3DistilledModularPipeline"
    _diffusers_requirement = "diffusers>=0.42.0"
    _max_reference_images = 10
