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
import locale
import os
from typing import Optional


class Locale:
    def __init__(self, language: Optional[str] = None):
        self._language = (
            language if language is not None else locale.getdefaultlocale()[0]
        )
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"{self._language}.json"
        )
        if os.path.exists(json_path):
            self._mapping = json.load(codecs.open(json_path, "r", encoding="utf-8"))
        else:
            self._mapping = None

    def __call__(self, content: str):
        if self._mapping is None:
            return content
        else:
            return self._mapping.get(content, content)
