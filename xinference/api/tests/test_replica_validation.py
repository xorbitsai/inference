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

"""Tests for REST launch payload validation helpers."""

import math

import pytest
from fastapi import HTTPException

from ..restful_api import _parse_replica_config, _validate_replica


class TestValidateReplica:
    """Unit tests for ``_validate_replica``."""

    # -- valid inputs -----------------------------------------------------------

    @pytest.mark.parametrize(
        "value, expected",
        [
            (1, 1),
            (999, 999),
            (42, 42),
            pytest.param(  # The frontend sends string numbers
                "2", 2, id="string_digit"
            ),
            pytest.param("10", 10, id="string_number"),
        ],
    )
    def test_valid_input(self, value, expected):
        assert _validate_replica(value) == expected

    # -- default value ----------------------------------------------------------

    def test_default_one(self):
        """When the key is absent, ``payload.get("replica", 1)`` passes ``1``."""
        assert _validate_replica(1) == 1

    # -- rejected: bool ---------------------------------------------------------

    @pytest.mark.parametrize("value", [True, False])
    def test_rejects_boolean(self, value):
        with pytest.raises(HTTPException) as exc:
            _validate_replica(value)
        assert exc.value.status_code == 400
        assert "boolean" in exc.value.detail

    # -- rejected: float --------------------------------------------------------

    @pytest.mark.parametrize(
        "value",
        [
            1.9,
            0.5,
            -3.0,
            pytest.param(math.inf, id="infinity"),
            pytest.param(-math.inf, id="negative_infinity"),
            pytest.param(math.nan, id="nan"),
        ],
    )
    def test_rejects_float(self, value):
        with pytest.raises(HTTPException) as exc:
            _validate_replica(value)
        assert exc.value.status_code == 400
        assert "float" in exc.value.detail

    # -- rejected: None ---------------------------------------------------------

    def test_rejects_none(self):
        with pytest.raises(HTTPException) as exc:
            _validate_replica(None)
        assert exc.value.status_code == 400

    # -- rejected: non-integer strings ------------------------------------------

    @pytest.mark.parametrize("value", ["abc", "", "1.5", "0x10"])
    def test_rejects_invalid_string(self, value):
        with pytest.raises(HTTPException) as exc:
            _validate_replica(value)
        assert exc.value.status_code == 400

    # -- rejected: value below 1 ------------------------------------------------

    @pytest.mark.parametrize("value", [0, -1, -100])
    def test_rejects_less_than_one(self, value):
        with pytest.raises(HTTPException) as exc:
            _validate_replica(value)
        assert exc.value.status_code == 400
        assert "at least 1" in exc.value.detail

    # -- rejected: unsupported types --------------------------------------------

    @pytest.mark.parametrize("value", [[], {}, b"bytes"])
    def test_rejects_unsupported_types(self, value):
        with pytest.raises(HTTPException) as exc:
            _validate_replica(value)
        assert exc.value.status_code == 400

    # -- edge: very large float (OverflowError path) ----------------------------

    def test_overflow_error_path(self):
        """``int(float("inf"))`` would raise ``OverflowError`` if it reached
        ``int()``, but our float guard rejects it before that point."""
        with pytest.raises(HTTPException) as exc:
            _validate_replica(float("inf"))
        assert exc.value.status_code == 400
        assert "float" in exc.value.detail


class TestParseReplicaConfig:
    """Unit tests for ``_parse_replica_config``."""

    def test_absent_config_returns_none(self):
        assert _parse_replica_config({}) is None

    def test_valid_config_is_parsed(self):
        configs = _parse_replica_config(
            {
                "replica_config": [
                    {
                        "replica_uid": "replica-a",
                        "devices": [
                            {
                                "worker_ip": "10.0.0.1:9978",
                                "n_gpu": 1,
                                "gpu_idx": [0],
                            }
                        ],
                    }
                ]
            }
        )

        assert configs is not None
        assert len(configs) == 1
        assert configs[0].replica_uid == "replica-a"
        assert configs[0].devices[0].worker_ip == "10.0.0.1:9978"

    def test_empty_config_is_preserved(self):
        assert _parse_replica_config({"replica_config": []}) == []

    @pytest.mark.parametrize(
        "legacy_placement",
        [
            {"worker_ip": "10.0.0.1:9978"},
            {"gpu_idx": []},
            {"n_gpu": 1},
        ],
    )
    def test_rejects_legacy_placement_options(self, legacy_placement):
        payload = {"replica_config": []}
        payload.update(legacy_placement)

        with pytest.raises(HTTPException) as exc:
            _parse_replica_config(payload)

        assert exc.value.status_code == 400
        assert "together with replica_config" in exc.value.detail

    def test_rejects_invalid_config(self):
        with pytest.raises(HTTPException) as exc:
            _parse_replica_config({"replica_config": [None]})

        assert exc.value.status_code == 400
        assert "Invalid replica_config" in exc.value.detail
