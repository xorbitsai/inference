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
import pytest
import requests

from .. import MODEL_FAMILIES


@pytest.mark.parametrize(
    "model_spec",
    [model_spec for model_family in MODEL_FAMILIES for model_spec in model_family],
)
def test_model_availability(model_spec):
    attempt = 0
    max_attempt = 3
    if model_spec.model_format != "pytorch":
        while attempt < max_attempt:
            attempt += 1
            try:
                assert requests.head(model_spec.url).status_code != 404
                break
            except Exception:
                continue

        if attempt == max_attempt:
            pytest.fail(f"{str(model_spec)} is not available")
