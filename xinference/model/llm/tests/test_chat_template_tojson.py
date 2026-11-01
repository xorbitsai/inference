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
from ..utils import ChatModelMixin

TOOLS = [{"name": "get_weather", "description": "查询城市天气"}]


def test_tojson_keeps_non_ascii_by_default():
    # Transformers' tojson defaults to ensure_ascii=False, so tool schemas in
    # templates such as `{{ tool | tojson }}` keep their original text.
    rendered = ChatModelMixin()._build_from_raw_template(
        [], "{{ tools | tojson }}", tools=TOOLS
    )
    assert rendered == '[{"name": "get_weather", "description": "查询城市天气"}]'


def test_tojson_still_accepts_ensure_ascii():
    rendered = ChatModelMixin()._build_from_raw_template(
        [], "{{ tools | tojson(ensure_ascii=True) }}", tools=TOOLS
    )
    assert "\\u67e5\\u8be2\\u57ce\\u5e02\\u5929\\u6c14" in rendered
