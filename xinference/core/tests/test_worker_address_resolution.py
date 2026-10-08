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

from ..supervisor import SupervisorActor


def test_get_worker_ref_by_ip_accepts_host_and_registered_address():
    ipv4_worker = object()
    ipv6_worker = object()
    supervisor = SupervisorActor.__new__(SupervisorActor)
    supervisor._worker_address_to_worker = {
        "10.0.0.8:9978": ipv4_worker,
        "[2001:db8::8]:9978": ipv6_worker,
    }

    assert supervisor._get_worker_ref_by_ip("10.0.0.8") is ipv4_worker
    assert supervisor._get_worker_ref_by_ip("10.0.0.8:9978") is ipv4_worker
    assert supervisor._get_worker_ref_by_ip("2001:db8::8") is ipv6_worker
    assert supervisor._get_worker_ref_by_ip("[2001:db8::8]:9978") is ipv6_worker
    assert supervisor._get_worker_ref_by_ip("10.0.0.9") is None
