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


@pytest.mark.asyncio
async def test_restful_api(setup):
    endpoint, _ = setup
    print(endpoint)
    url = f"{endpoint}/v1/models"

    response = requests.get(url)
    response_data = response.json()
    assert len(response_data) == 0
