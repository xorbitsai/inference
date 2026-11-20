# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# CI: linux

from datetime import datetime

import pytest

from xinference.api.oauth2.advanced.cache import ApiKeyCache


@pytest.mark.parametrize(
    "expires_at, expected, valid",
    [
        ("2099-01-01T00:00:00+00:00", datetime(2099, 1, 1), True),
        ("2000-01-01T00:00:00+08:00", datetime(1999, 12, 31, 16), False),
        ("2099-01-01T00:00:00+08:00", datetime(2098, 12, 31, 16), True),
        ("2099-01-01T00:00:00Z", datetime(2099, 1, 1), True),
        ("2099-01-01T00:00:00", datetime(2099, 1, 1), True),
        ("2000-01-01T00:00:00", datetime(2000, 1, 1), False),
    ],
)
def test_build_entry_expiry(expires_at, expected, valid):
    cache = object.__new__(ApiKeyCache)
    entry = cache._build_entry(
        {
            "id": 1,
            "user_id": 1,
            "key_hash": "test-hash",
            "key_prefix": "test-prefix",
            "expires_at": expires_at,
        }
    )

    assert entry.expires_at == expected
    assert entry.expires_at.tzinfo is None
    assert entry.is_valid() is valid
