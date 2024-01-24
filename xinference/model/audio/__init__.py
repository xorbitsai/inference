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

from .core import AudioModelFamilyV1, generate_audio_description, get_cache_status

_model_spec_json = os.path.join(os.path.dirname(__file__), "model_spec.json")
BUILTIN_AUDIO_MODELS = dict(
    (spec["model_name"], AudioModelFamilyV1(**spec))
    for spec in json.load(codecs.open(_model_spec_json, "r", encoding="utf-8"))
)

del _model_spec_json
