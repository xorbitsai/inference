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

import pytest

from ..indextts2 import _resolve_duration_factor


@pytest.mark.parametrize(
    ("speed", "expected"),
    [
        (None, 1.0),
        (0.5, 2.0),
        (1.0, 1.0),
        (2.0, 0.5),
    ],
)
def test_resolve_duration_factor_from_speed(speed, expected):
    assert _resolve_duration_factor(speed, None) == pytest.approx(expected)


@pytest.mark.parametrize(
    "speed",
    [0.0, 0.49, 2.01, float("-inf"), float("inf"), float("nan")],
)
def test_resolve_duration_factor_rejects_invalid_speed(speed):
    with pytest.raises(ValueError, match="speed must be between 0.5 and 2.0"):
        _resolve_duration_factor(speed, None)


@pytest.mark.parametrize("duration_factor", [0.5, 1.0, 2.0])
def test_resolve_duration_factor_accepts_explicit_value(duration_factor):
    assert _resolve_duration_factor(2.0, duration_factor) == pytest.approx(
        duration_factor
    )


@pytest.mark.parametrize(
    "duration_factor",
    [0.0, 0.49, 2.01, float("-inf"), float("inf"), float("nan")],
)
def test_resolve_duration_factor_rejects_invalid_explicit_value(duration_factor):
    with pytest.raises(ValueError, match="duration_factor must be between 0.5 and 2.0"):
        _resolve_duration_factor(1.0, duration_factor)
