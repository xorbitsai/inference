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
"""The bulk-unban endpoints report how many bans they lifted.

The record maps also hold IPs and keys that failed without reaching the ban
threshold, so counting records overstates the answer the admin API returns.
"""

from ..oauth2.advanced.rate_limiter import RateLimitConfig, RateLimiter

KEY_CONFIG = RateLimitConfig(max_failures=2, window_seconds=300, ban_seconds=600)


def _ban_ip(limiter: RateLimiter, ip: str) -> None:
    for _ in range(limiter._ip_config.max_failures):
        limiter.record_invalid_key(ip)


def _ban_key(limiter: RateLimiter, ip: str, key_id: int) -> None:
    for _ in range(KEY_CONFIG.max_failures):
        limiter.record_key_failure(ip, key_id, KEY_CONFIG)


def test_unban_all_ips_counts_bans_not_records():
    limiter = RateLimiter()
    _ban_ip(limiter, "10.0.0.1")
    # One failure short of the threshold: recorded, never banned.
    for _ in range(limiter._ip_config.max_failures - 1):
        limiter.record_invalid_key("10.0.0.2")

    assert len(limiter.get_banned_ips()) == 1
    assert limiter.unban_all_ips() == 1
    assert limiter.get_banned_ips() == []


def test_unban_all_ips_ignores_an_expired_ban():
    limiter = RateLimiter()
    _ban_ip(limiter, "10.0.0.1")
    # An expired ban nothing has read back still leaves its record in place.
    limiter._ip_records["10.0.0.1"].banned_until = 0.0

    assert limiter.get_banned_ips() == []
    assert limiter.unban_all_ips() == 0


def test_unban_all_keys_counts_bans_not_records():
    limiter = RateLimiter()
    _ban_key(limiter, "10.0.0.1", 1)
    limiter.record_key_failure("10.0.0.2", 2, KEY_CONFIG)

    assert len(limiter.get_banned_keys()) == 1
    assert limiter.unban_all_keys() == 1


def test_unban_key_all_counts_bans_not_records():
    limiter = RateLimiter()
    _ban_key(limiter, "10.0.0.1", 7)
    _ban_key(limiter, "10.0.0.2", 7)
    limiter.record_key_failure("10.0.0.3", 7, KEY_CONFIG)
    # A different key, banned: neither counted nor removed.
    _ban_key(limiter, "10.0.0.4", 8)

    assert limiter.unban_key_all(7) == 2
    assert [ban["key_id"] for ban in limiter.get_banned_keys()] == [8]
