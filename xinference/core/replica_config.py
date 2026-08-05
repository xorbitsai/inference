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
"""Per-replica placement configuration for first-time model launch.

``replica_config`` lets a caller pin each replica to a specific worker
(full ``ip:port``) and GPU on launch, instead of relying on the supervisor's
automatic load-balanced scheduling.

This module only owns the *structural* data model and pure validation
(lengths, uniqueness, ``n_gpu``/``gpu_idx`` consistency, ``replica_uid``
naming). Stateful checks (worker registered in the cluster, GPU index legal,
cross-replica GPU conflicts) live in the supervisor, which has access to the
cluster topology.
"""

from typing import List, Optional, Union

from .._compat import BaseModel
from .utils import build_replica_model_uid, parse_replica_model_uid


class DeviceConfig(BaseModel):
    """Placement of one replica on a single worker.

    Each replica is restricted to exactly one device for now (no cross-worker
    sharding within a single replica); ``devices`` length is validated to be 1.
    """

    worker_ip: str  # full worker address "ip:port"
    n_gpu: Union[int, str] = "auto"  # int count, or "auto" for worker auto-allocation
    gpu_idx: Optional[List[int]] = None  # explicit GPU indexes; None/empty = auto


class ReplicaConfig(BaseModel):
    """Placement spec for a single replica."""

    replica_uid: Optional[str] = None
    devices: List[DeviceConfig] = []

    @classmethod
    def from_dict(cls, data: dict) -> "ReplicaConfig":
        return cls(**data)


def _is_reserved_replica_uid(uid: str) -> bool:
    """True if ``uid`` collides with an internal replica/rank identifier."""
    _, rep = parse_replica_model_uid(uid)
    return rep != -1 or uid.endswith("-rank0")


def normalize_replica_configs(
    model_uid: str,
    replica: int,
    configs: List[ReplicaConfig],
) -> List[ReplicaConfig]:
    """Validate structure and fill in default ``replica_uid``.

    Returns a list aligned by replica index (length ``replica``). Raises
    ``ValueError`` on any structural violation. Stateful validation (worker
    existence, GPU legality, cross-replica GPU conflicts) is intentionally not
    done here — it belongs to the supervisor which holds cluster topology.
    """
    if len(configs) != replica:
        raise ValueError(
            f"replica_config length ({len(configs)}) must equal replica ({replica})."
        )

    seen_uids: set = set()
    resolved: List[ReplicaConfig] = []
    for idx, cfg in enumerate(configs):
        if len(cfg.devices) != 1:
            raise ValueError(
                "Each replica_config entry must have exactly one device "
                f"(replica {idx} has {len(cfg.devices)}). "
                "Cross-worker sharding per replica is not supported yet."
            )
        device = cfg.devices[0]

        _validate_device_consistency(idx, device)

        replica_uid = cfg.replica_uid
        if replica_uid is None:
            replica_uid = build_replica_model_uid(model_uid, idx)
        elif _is_reserved_replica_uid(replica_uid):
            raise ValueError(
                f"replica_uid '{replica_uid}' collides with a reserved internal "
                "replica/rank identifier; please choose another name."
            )
        if replica_uid in seen_uids:
            raise ValueError(
                f"Duplicate replica_uid '{replica_uid}' in replica_config."
            )
        seen_uids.add(replica_uid)

        cfg.replica_uid = replica_uid
        resolved.append(cfg)
    return resolved


def _validate_device_consistency(idx: int, device: DeviceConfig) -> None:
    """Check n_gpu / gpu_idx consistency for one device."""
    if not isinstance(device.n_gpu, int) and device.n_gpu != "auto":
        raise ValueError(
            f"replica_config[{idx}].devices[0].n_gpu must be an int or 'auto', "
            f"got {device.n_gpu!r}."
        )
    gpu_idx = device.gpu_idx or []
    if gpu_idx:
        if device.n_gpu != "auto" and device.n_gpu != len(gpu_idx):
            raise ValueError(
                f"replica_config[{idx}].devices[0].n_gpu ({device.n_gpu}) must equal "
                f"len(gpu_idx) ({len(gpu_idx)}), or be 'auto'."
            )
        if any(not isinstance(g, int) or g < 0 for g in gpu_idx):
            raise ValueError(
                f"replica_config[{idx}].devices[0].gpu_idx must be non-negative integers."
            )
