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

import pytest
from PIL import Image

from .. import utils
from ..utils import MAX_IMAGE_SEED, handle_image_result, resolve_image_seed_list


def test_resolve_image_seed_list_preserves_and_pads_seeds():
    seeds = resolve_image_seed_list([11, -1], 4)

    assert seeds is not None
    assert seeds[0] == 11
    assert len(seeds) == 4
    assert all(0 <= seed <= MAX_IMAGE_SEED for seed in seeds)


def test_resolve_image_seed_list_keeps_scalar_seed_backward_compatible():
    assert resolve_image_seed_list(11, 4) is None


@pytest.mark.parametrize(
    "seeds, n",
    [([1, 2], 1), ([-2], 1), ([MAX_IMAGE_SEED + 1], 1), ([True], 1), ([1.5], 1)],
)
def test_resolve_image_seed_list_rejects_invalid_values(seeds, n):
    with pytest.raises(ValueError):
        resolve_image_seed_list(seeds, n)


def test_handle_image_result_url_raises_when_saving_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(utils, "XINFERENCE_IMAGE_DIR", str(tmp_path))
    saved = Image.new("RGB", (4, 4))
    unsaveable = Image.new("I;16", (4, 4))

    with pytest.raises(OSError):
        handle_image_result("url", [saved, unsaveable])
