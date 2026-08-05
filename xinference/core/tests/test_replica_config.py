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
import asyncio

import pytest

from ..replica_config import DeviceConfig, ReplicaConfig, normalize_replica_configs
from ..supervisor import SupervisorActor


class _FakeWorker:
    """Minimal stand-in for a WorkerActor ref used by _resolve_replica_config."""

    def __init__(
        self, address, total, allow_share=False, models=None, user_specified=None
    ):
        self.address = address
        self._total = list(total)
        self._allow_share = allow_share
        self._models = models or {}
        self._user_specified = user_specified or {}

    async def get_gpu_allocation_status(self):
        return {
            "total": list(self._total),
            "allow_multi_replica_per_gpu": self._allow_share,
            "models": self._models,
            "user_specified": self._user_specified,
        }


def _make_supervisor(workers, address="supervisor:9999"):
    sup = SupervisorActor.__new__(SupervisorActor)
    sup._worker_address_to_worker = {w.address: w for w in workers}
    sup.address = address
    return sup


def _cfg(replica_uid=None, worker_ip="10.0.0.1:9978", n_gpu="auto", gpu_idx=None):
    return ReplicaConfig(
        replica_uid=replica_uid,
        devices=[DeviceConfig(worker_ip=worker_ip, n_gpu=n_gpu, gpu_idx=gpu_idx)],
    )


# --------------------------- pure structural tests ---------------------------


def test_from_dict_roundtrip():
    cfg = ReplicaConfig.from_dict(
        {
            "replica_uid": "m-0",
            "devices": [{"worker_ip": "1.2.3.4:9978", "n_gpu": 1, "gpu_idx": [0]}],
        }
    )
    assert cfg.replica_uid == "m-0"
    assert cfg.devices[0].worker_ip == "1.2.3.4:9978"
    assert cfg.devices[0].n_gpu == 1
    assert cfg.devices[0].gpu_idx == [0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_ip", 9978),
        ("n_gpu", 1.5),
        ("n_gpu", True),
        ("n_gpu", "1"),
        ("gpu_idx", [0.9]),
        ("gpu_idx", [True]),
        ("gpu_idx", [0, 0]),
    ],
)
def test_device_config_rejects_non_strict_placement_values(field, value):
    data = {"worker_ip": "10.0.0.1:9978", field: value}
    with pytest.raises(ValueError):
        DeviceConfig(**data)


def test_placement_config_rejects_unknown_fields():
    with pytest.raises(ValueError):
        DeviceConfig(worker_ip="10.0.0.1:9978", gpu_id=0)
    with pytest.raises(ValueError):
        ReplicaConfig(devices=[], replica_name="replica")


def test_normalize_length_mismatch():
    with pytest.raises(ValueError, match="must equal replica"):
        normalize_replica_configs("m", replica=2, configs=[_cfg()])


def test_normalize_devices_length_not_one():
    cfg = ReplicaConfig(replica_uid=None, devices=[])
    with pytest.raises(ValueError, match="exactly one device"):
        normalize_replica_configs("m", replica=1, configs=[cfg])


def test_normalize_n_gpu_consistency():
    # n_gpu numeric must equal len(gpu_idx).
    with pytest.raises(ValueError, match="must equal"):
        normalize_replica_configs("m", 1, [_cfg(n_gpu=3, gpu_idx=[0, 1])])
    # Consistent is fine.
    normalize_replica_configs("m", 1, [_cfg(n_gpu=2, gpu_idx=[0, 1])])
    # auto is fine with explicit gpu_idx.
    normalize_replica_configs("m", 1, [_cfg(n_gpu="auto", gpu_idx=[0])])


def test_normalize_rejects_duplicate_gpu_indexes_within_replica():
    with pytest.raises(ValueError, match="duplicate indexes"):
        normalize_replica_configs("m", 1, [_cfg(n_gpu=2, gpu_idx=[0, 0])])


def test_normalize_missing_replica_uid_gets_stable_default():
    configs = normalize_replica_configs(
        "my-model", 2, [_cfg(), _cfg(worker_ip="10.0.0.2:9978")]
    )
    assert configs[0].replica_uid == "my-model-0"
    assert configs[1].replica_uid == "my-model-1"


def test_normalize_replica_uid_strips_whitespace():
    configs = normalize_replica_configs("m", 1, [_cfg(replica_uid="  primary  ")])
    assert configs[0].replica_uid == "primary"


@pytest.mark.parametrize("uid", ["", "   "])
def test_normalize_rejects_blank_replica_uid(uid):
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_replica_configs("m", 1, [_cfg(replica_uid=uid)])


def test_normalize_replica_uid_uniqueness():
    with pytest.raises(ValueError, match="Duplicate replica_uid"):
        normalize_replica_configs(
            "m", 2, [_cfg(replica_uid="same"), _cfg(replica_uid="same")]
        )


@pytest.mark.parametrize("uid", ["m-rep0", "my-model-rank0"])
def test_normalize_reserved_replica_uid(uid):
    with pytest.raises(ValueError, match="reserved"):
        normalize_replica_configs("m", 1, [_cfg(replica_uid=uid)])


def test_normalize_valid():
    configs = normalize_replica_configs(
        "m",
        2,
        [
            _cfg(replica_uid="r0", worker_ip="10.0.0.1:9978", gpu_idx=[0]),
            _cfg(replica_uid="r1", worker_ip="10.0.0.2:9978", gpu_idx=[1]),
        ],
    )
    assert [c.replica_uid for c in configs] == ["r0", "r1"]


def test_resolve_preserves_per_replica_n_gpu():
    w1 = _FakeWorker("10.0.0.1:9978", total=[0, 1, 2, 3])
    sup = _make_supervisor([w1])
    targets, _ = asyncio.run(
        sup._resolve_replica_config("m", 1, [_cfg(worker_ip=w1.address, n_gpu=2)])
    )
    assert targets[0][1] is None
    assert targets[0][2] == 2


def test_resolve_normalizes_auto_n_gpu_for_explicit_gpu_indexes():
    w1 = _FakeWorker("10.0.0.1:9978", total=[0, 1, 2, 3])
    sup = _make_supervisor([w1])
    targets, _ = asyncio.run(
        sup._resolve_replica_config(
            "m", 1, [_cfg(worker_ip=w1.address, gpu_idx=[0, 1])]
        )
    )
    assert targets[0][1] == [0, 1]
    assert targets[0][2] == 2


def test_resolve_rejects_existing_gpu_occupancy():
    w1 = _FakeWorker(
        "10.0.0.1:9978", total=[0, 1], allow_share=False, models={0: ["existing"]}
    )
    sup = _make_supervisor([w1])
    with pytest.raises(ValueError, match="already occupied"):
        asyncio.run(
            sup._resolve_replica_config(
                "m", 1, [_cfg(worker_ip=w1.address, gpu_idx=[0])]
            )
        )


# ------------------------- stateful resolve tests ----------------------------


def test_resolve_valid():
    w1 = _FakeWorker("10.0.0.1:9978", total=[0, 1])
    w2 = _FakeWorker("10.0.0.2:9978", total=[0, 1])
    sup = _make_supervisor([w1, w2])
    configs = [
        _cfg(worker_ip="10.0.0.1:9978", gpu_idx=[0]),
        _cfg(worker_ip="10.0.0.2:9978", gpu_idx=[1]),
    ]
    targets, uid_map = asyncio.run(sup._resolve_replica_config("m", 2, configs))
    assert targets[0][0].address == "10.0.0.1:9978"
    assert targets[0][1] == [0]
    assert targets[0][2] == 1
    assert targets[1][0].address == "10.0.0.2:9978"
    assert targets[1][1] == [1]
    assert uid_map == {0: "m-0", 1: "m-1"}


def test_resolve_preserves_explicit_replica_uid():
    w1 = _FakeWorker("10.0.0.1:9978", total=[0, 1])
    w2 = _FakeWorker("10.0.0.2:9978", total=[0, 1])
    sup = _make_supervisor([w1, w2])
    configs = [
        _cfg(replica_uid="primary", worker_ip=w1.address, gpu_idx=[0]),
        _cfg(replica_uid="backup", worker_ip=w2.address, gpu_idx=[1]),
    ]

    _, uid_map = asyncio.run(sup._resolve_replica_config("m", 2, configs))

    assert uid_map == {0: "primary", 1: "backup"}


def test_resolve_unknown_worker():
    w1 = _FakeWorker("10.0.0.1:9978", total=[0])
    sup = _make_supervisor([w1])
    with pytest.raises(ValueError, match="not in the cluster"):
        asyncio.run(
            sup._resolve_replica_config(
                "m", 1, [_cfg(worker_ip="10.0.0.9:9978", gpu_idx=[0])]
            )
        )


def test_resolve_gpu_not_visible():
    w1 = _FakeWorker("10.0.0.1:9978", total=[0, 1])
    sup = _make_supervisor([w1])
    with pytest.raises(ValueError, match="not visible"):
        asyncio.run(
            sup._resolve_replica_config(
                "m", 1, [_cfg(worker_ip="10.0.0.1:9978", gpu_idx=[9])]
            )
        )


def test_resolve_static_conflict_disallowed():
    # Same worker, overlapping gpu_idx, sharing disallowed -> rejected.
    w1 = _FakeWorker("10.0.0.1:9978", total=[0, 1], allow_share=False)
    sup = _make_supervisor([w1])
    configs = [
        _cfg(replica_uid="a", worker_ip="10.0.0.1:9978", gpu_idx=[0]),
        _cfg(replica_uid="b", worker_ip="10.0.0.1:9978", gpu_idx=[0]),
    ]
    with pytest.raises(ValueError, match="conflict"):
        asyncio.run(sup._resolve_replica_config("m", 2, configs))


def test_resolve_allow_share_no_conflict():
    # Same GPU on the same worker is allowed when the worker allows sharing.
    w1 = _FakeWorker("10.0.0.1:9978", total=[0, 1], allow_share=True)
    sup = _make_supervisor([w1])
    configs = [
        _cfg(replica_uid="a", worker_ip="10.0.0.1:9978", gpu_idx=[0]),
        _cfg(replica_uid="b", worker_ip="10.0.0.1:9978", gpu_idx=[0]),
    ]
    targets, _ = asyncio.run(sup._resolve_replica_config("m", 2, configs))
    assert targets[0][1] == [0] and targets[1][1] == [0]


def test_resolve_local_mode_ignores_worker_ip():
    # Local deployment: a single worker whose address == supervisor address.
    local_addr = "127.0.0.1:9978"
    w = _FakeWorker(local_addr, total=[0, 1])
    sup = _make_supervisor([w], address=local_addr)
    assert sup.is_local_deployment() is True
    # worker_ip does not match, but local mode pins to the local worker anyway.
    targets, _ = asyncio.run(
        sup._resolve_replica_config(
            "m", 1, [_cfg(worker_ip="9.9.9.9:9978", gpu_idx=[0])]
        )
    )
    assert targets[0][0].address == local_addr
    assert targets[0][1] == [0]


def test_resolve_auto_gpu_skips_existence_check():
    # gpu_idx omitted (auto) -> no existence check, worker will auto-allocate.
    w1 = _FakeWorker("10.0.0.1:9978", total=[])
    sup = _make_supervisor([w1])
    targets, _ = asyncio.run(
        sup._resolve_replica_config("m", 1, [_cfg(worker_ip="10.0.0.1:9978")])
    )
    assert targets[0][1] is None
