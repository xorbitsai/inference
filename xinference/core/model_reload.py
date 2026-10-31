# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

"""Model-wide reload policy over independently retained replica generations."""

import asyncio
import copy
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .exceptions import ModelReloadInProgress
from .utils import parse_replica_model_uid

logger = logging.getLogger(__name__)


def _common_values(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in configs[0].items()
        if all(key in config and config[key] == value for config in configs[1:])
    }


def _update_progress(status: Dict[str, Any], active: bool = False) -> None:
    replicas = status["replicas"]
    completed = sum(replica["status"] == "ready" for replica in replicas)
    status["completed_replicas"] = completed
    status["replica_count"] = len(replicas)
    status["weights_reused"] = completed == len(replicas) and all(
        replica.get("weights_reused") is True for replica in replicas
    )
    failed = next((r for r in replicas if r["status"] == "error"), None)
    if failed is not None and status["status"] != "cancelled":
        status.update(
            stage="error",
            error=f"Replica {failed['replica_uid']}: {failed.get('error', 'Reload failed')}",
        )
        if not active:
            status["status"] = "error"
        if "restored" in failed:
            # A restored failed replica does not undo already committed replicas.
            status["restored"] = completed == 0 and failed["restored"]
    if status["status"] == "error":
        for replica in replicas:
            if replica["status"] == "queued":
                replica.update(status="skipped", stage="skipped")
    status["partial"] = status["status"] in (
        "error",
        "cancelled",
    ) and 0 < completed < len(replicas)


class ModelReloadMixin:
    _model_uid_to_replica_info: Dict[str, Any]
    _replica_model_uid_to_worker: Dict[str, Any]
    _model_reload_status: Dict[str, Dict[str, Any]]
    _model_reload_tasks: Dict[str, asyncio.Task]
    _autostart_store_lock: asyncio.Lock
    _launch_history_store: Any

    if TYPE_CHECKING:

        def _iter_active_replica_model_uids(self, model_uid: str) -> List[str]: ...

        def _get_model_replica_lock(self, model_uid: str) -> asyncio.Lock: ...

        def _invalidate_list_models_debounce_cache(self) -> None: ...

    def _get_reload_targets(self, model_uid: str) -> List[Tuple[str, Any]]:
        if model_uid not in self._model_uid_to_replica_info:
            raise ValueError(f"Model not found: {model_uid}")
        if model_uid in getattr(self, "_pd_model_mapping", {}):
            raise ValueError("Parameter reload does not support P/D models")
        targets = []
        for replica_uid in self._iter_active_replica_model_uids(model_uid):
            workers = self._replica_model_uid_to_worker.get(replica_uid)
            if not isinstance(workers, (list, tuple)):
                workers = [workers]
            if len(workers) != 1 or workers[0] is None:
                raise ValueError("Parameter reload requires one worker per replica")
            targets.append((replica_uid, workers[0]))
        if not targets:
            raise ValueError(f"No active replicas for model: {model_uid}")
        return targets

    def _check_model_reload_idle(self, model_uid: str) -> None:
        if model_uid in self._model_reload_tasks:
            raise ModelReloadInProgress("This model already has a reload in progress")

    async def get_model_reload_config(self, model_uid: str) -> Dict[str, Any]:
        targets = self._get_reload_targets(model_uid)
        configs = await asyncio.gather(
            *(worker.get_model_reload_config(uid) for uid, worker in targets)
        )
        if len(configs) == 1:
            return configs[0]
        result = _common_values(configs)
        parameters = _common_values([config["parameters"] for config in configs])
        shared = _common_values([config["model_config"] for config in configs])
        # The UI edits image count only. Other modality limits remain per replica.
        for key, kind in parameters.items():
            if kind == "image_limits":
                values = [
                    config["model_config"].get(key, {}).get("image")
                    for config in configs
                ]
                if values[0] is not None and all(
                    value == values[0] for value in values
                ):
                    shared[key] = {"image": values[0]}
        result.update(
            engine=configs[0]["engine"],
            enabled=all(config["enabled"] for config in configs),
            parameters=parameters,
            model_config={
                key: value for key, value in shared.items() if key in parameters
            },
            mixed_parameters=[
                key
                for key in parameters
                if key not in shared
                and any(key in config["model_config"] for config in configs)
            ],
            hot_parameters=[
                key
                for key in parameters
                if all(key in config.get("hot_parameters", []) for config in configs)
            ],
            replicas=[
                {
                    "replica_uid": uid,
                    **{
                        key: copy.deepcopy(config[key])
                        for key in (
                            "enabled",
                            "engine",
                            "engine_version",
                            "model_config",
                        )
                        if key in config
                    },
                }
                for (uid, _), config in zip(targets, configs)
            ],
        )
        return result

    async def get_model_reload_status(self, model_uid: str) -> Dict[str, Any]:
        targets = self._get_reload_targets(model_uid)
        observed = await asyncio.gather(
            *(worker.get_model_reload_status(uid) for uid, worker in targets),
            return_exceptions=True,
        )
        status = self._model_reload_status.get(model_uid)
        if status is None:
            if len(targets) == 1:
                value = observed[0]
                return {"status": "idle", **(value if isinstance(value, dict) else {})}
            # Reconstruct what retained ModelActors actually report after a
            # Supervisor loses its operation cache. Never restart a rollout here.
            latest = max(
                (
                    value
                    for value in observed
                    if isinstance(value, dict)
                    and (
                        value.get("operation_id")
                        or value.get("status") in ("reloading", "ready", "error")
                    )
                ),
                key=lambda value: value.get("started_at", 0),
                default={},
            )
            status = {"status": "idle", **latest, "replicas": []}
            for (uid, _), value in zip(targets, observed):
                row = {"replica_uid": uid, "status": "queued", "stage": "queued"}
                if isinstance(value, dict) and value.get("operation_id") == latest.get(
                    "operation_id"
                ):
                    row.update(value)
                status["replicas"].append(row)
            if latest:
                status["status"] = "reloading"
                if not latest.get("operation_id"):
                    status.update(
                        status="error",
                        stage="error",
                        error="Cannot reconstruct a reload operation from legacy replica statuses",
                    )
                self._model_reload_status[model_uid] = status
        if "replicas" not in status:
            # Retained statuses from a previous Supervisor used the flat schema.
            status["replicas"] = [{"replica_uid": targets[0][0], **status}]
        rows = {row["replica_uid"]: row for row in status["replicas"]}
        for (uid, _), value in zip(targets, observed):
            member = rows.get(uid)
            if member is None or not isinstance(value, dict):
                continue
            if value.get("operation_id") not in (None, status.get("operation_id")):
                continue
            # A queued member may still be reporting its previous operation.
            if member["status"] == "queued" and model_uid in self._model_reload_tasks:
                continue
            member.update(value)
        if status["status"] == "reloading":
            current = next(
                (r for r in status["replicas"] if r["status"] == "reloading"), None
            )
            if current is not None:
                status["stage"] = current.get("stage", status.get("stage", "loading"))
            elif model_uid not in self._model_reload_tasks:
                if all(r["status"] == "ready" for r in status["replicas"]):
                    status.update(status="ready", stage="ready")
                else:
                    status.update(
                        status="error",
                        stage="error",
                        error="Reload interrupted before all replicas completed",
                    )
        _update_progress(status, active=model_uid in self._model_reload_tasks)
        return copy.deepcopy(status)

    async def restore_model_reload_config(
        self, replica_uid: str, config: Dict[str, Any], observed: Dict[str, Any]
    ) -> None:
        model_uid, _ = parse_replica_model_uid(replica_uid)
        status = self._model_reload_status.get(model_uid, {})
        if status.get("operation_id") not in (None, observed.get("operation_id")):
            return
        async with self._autostart_store_lock:
            if self._model_reload_status.get(model_uid, {}).get("operation_id") not in (
                None,
                observed.get("operation_id"),
            ):
                return
            replicas = status.get("replicas", [])
            if len(replicas) > 1:
                row = next(
                    (r for r in replicas if r["replica_uid"] == replica_uid), None
                )
                if row is not None:
                    row.update(observed)
                    _update_progress(
                        status, active=model_uid in self._model_reload_tasks
                    )
                # One rollback must never overwrite the model-wide desired
                # configuration or other replicas' committed execution limits.
                return
            await asyncio.to_thread(
                self._launch_history_store.update_autostart_launch_config,
                model_uid,
                config,
            )
            status.update(observed)
            if replicas:
                replicas[0].update(observed)
                _update_progress(status, active=model_uid in self._model_reload_tasks)

    async def reload_model(
        self,
        model_uid: str,
        config: Dict[str, Any],
        drain_timeout: float = 300,
        python_executable: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._check_model_reload_idle(model_uid)
        async with self._get_model_replica_lock(model_uid):
            self._check_model_reload_idle(model_uid)
            targets = self._get_reload_targets(model_uid)
            patches = []
            for uid, worker in targets:
                if python_executable is not None:
                    metadata = await worker.get_model_reload_config(uid)
                    if not metadata["enabled"]:
                        raise ValueError(f"Replica {uid} has no retained GPU weights")
                # All members are validated before any engine is changed.
                normalized = (
                    await worker.validate_model_reload(uid, config) if config else {}
                )
                patches.append(config if normalized is None else normalized)
            status: Dict[str, Any] = {
                "operation_id": str(uuid.uuid4()),
                "model_uid": model_uid,
                "status": "reloading",
                "stage": "queued",
                "started_at": time.time(),
                "weights_reused": False,
                "completed_replicas": 0,
                "replica_count": len(targets),
                "replicas": [
                    {"replica_uid": uid, "status": "queued", "stage": "queued"}
                    for uid, _ in targets
                ],
            }
            self._model_reload_status[model_uid] = status

            async def run() -> None:
                from ..model.llm.weight_cache import HOT_RELOAD_FIELDS

                row = status["replicas"][0]
                persisting = False
                try:
                    for (uid, worker), patch, row in zip(
                        targets, patches, status["replicas"]
                    ):
                        if self._get_reload_targets(model_uid) != targets:
                            raise ValueError("Model placement changed during reload")
                        if any(
                            member["status"] == "error" for member in status["replicas"]
                        ):
                            break
                        row.update(
                            status="reloading",
                            started_at=time.time(),
                            stage=(
                                "draining"
                                if python_executable is None
                                and set(patch) <= HOT_RELOAD_FIELDS.keys()
                                else "loading"
                            ),
                        )
                        status.update(stage=row["stage"], current_replica=uid)
                        row["weights_reused"] = (
                            await worker.reload_model(
                                uid,
                                patch,
                                drain_timeout,
                                python_executable=python_executable,
                                operation_id=status["operation_id"],
                            )
                        ) is True
                        if self._get_reload_targets(model_uid) != targets:
                            raise ValueError("Model placement changed during reload")
                        if row["status"] == "error":
                            break
                        row.update(
                            status="ready", stage="ready", finished_at=time.time()
                        )
                        _update_progress(status, active=True)
                    if all(
                        member["status"] == "ready" for member in status["replicas"]
                    ):
                        async with self._autostart_store_lock:
                            if all(
                                member["status"] == "ready"
                                for member in status["replicas"]
                            ):
                                persisting = True
                                committed = patches[0] if len(targets) == 1 else config
                                await asyncio.to_thread(
                                    self._launch_history_store.update_autostart_launch_config,
                                    model_uid,
                                    {
                                        **committed,
                                        **(
                                            {
                                                "model_python_executable": python_executable
                                            }
                                            if python_executable is not None
                                            else {}
                                        ),
                                    },
                                )
                                if all(
                                    member["status"] == "ready"
                                    for member in status["replicas"]
                                ):
                                    status.update(status="ready", stage="ready")
                except asyncio.CancelledError:
                    status.update(status="cancelled", stage="cancelled")
                    for member in status["replicas"]:
                        if member["status"] in ("queued", "reloading"):
                            member.update(status="cancelled", stage="cancelled")
                    raise
                except Exception as exc:
                    if persisting:
                        status.update(status="error", stage="error", error=str(exc))
                    else:
                        row.update(status="error", stage="error", error=str(exc))
                    if not persisting:
                        if hasattr(exc, "restored"):
                            row["restored"] = exc.restored
                        try:
                            observed = await targets[status["replicas"].index(row)][
                                1
                            ].get_model_reload_status(row["replica_uid"])
                            if observed.get("operation_id") in (
                                None,
                                status["operation_id"],
                            ):
                                if "restored" in observed:
                                    row["restored"] = observed["restored"]
                        except Exception:
                            pass
                    _update_progress(status, active=True)
                    logger.exception(
                        "Reload of %s failed at %s", model_uid, row["replica_uid"]
                    )
                finally:
                    status["finished_at"] = time.time()
                    if (
                        self._model_reload_tasks.get(model_uid)
                        is asyncio.current_task()
                    ):
                        self._model_reload_tasks.pop(model_uid, None)
                    _update_progress(status)
                    self._invalidate_list_models_debounce_cache()

            self._model_reload_tasks[model_uid] = asyncio.create_task(run())
            return copy.deepcopy(status)
