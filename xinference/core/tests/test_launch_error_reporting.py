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
"""Tests that a failed launch's real cause reaches the status records.

The Web UI polls ``/v1/models/{uid}/replicas``; these cover the path that
populates it, plus the parts of the supervisor that used to drop the error.
"""
import asyncio

import pytest

from ..status_guard import InstanceInfo, LaunchStatus, StatusGuardActor
from ..supervisor import SupervisorActor, callback_for_async_launch


def _make_guard(model_uid: str = "m1") -> StatusGuardActor:
    guard = StatusGuardActor()
    guard.set_instance_info(
        model_uid,
        InstanceInfo(
            model_name="m",
            model_uid=model_uid,
            model_version=None,
            model_ability=[],
            replica=1,
            status=LaunchStatus.CREATING.name,
            instance_created_ts=0,
        ),
    )
    return guard


def _only_replica(guard: StatusGuardActor, model_uid: str = "m1"):
    statuses = guard.get_replica_statuses(model_uid)
    assert len(statuses) == 1
    return statuses[0]


class TestStickyErrorStatus:
    """A recorded failure must survive the cleanup that immediately follows it."""

    def test_error_survives_terminating_update(self):
        guard = _make_guard()
        guard.update_replica_status(
            "m1", 0, {"status": LaunchStatus.ERROR.name, "error_message": "boom"}
        )
        # This is what terminate_model -> worker._update_model_state does next.
        guard.update_replica_status(
            "m1",
            0,
            {"status": LaunchStatus.TERMINATING.name, "model_state": "stopping"},
        )

        replica = _only_replica(guard)
        assert replica.status == LaunchStatus.ERROR.name
        assert replica.error_message == "boom"
        # Non-status keys must still apply, so cleanup stays observable.
        assert replica.model_state == "stopping"

    def test_error_survives_terminated_update(self):
        guard = _make_guard()
        guard.update_replica_status(
            "m1", 0, {"status": LaunchStatus.ERROR.name, "error_message": "boom"}
        )
        guard.update_replica_status(
            "m1", 0, {"status": LaunchStatus.TERMINATED.name, "model_state": "stopped"}
        )

        replica = _only_replica(guard)
        assert replica.status == LaunchStatus.ERROR.name
        assert replica.model_state == "stopped"

    def test_ready_clears_error(self):
        """A genuine relaunch must be able to clear the failure."""
        guard = _make_guard()
        guard.update_replica_status(
            "m1", 0, {"status": LaunchStatus.ERROR.name, "error_message": "boom"}
        )
        guard.update_replica_status(
            "m1", 0, {"status": LaunchStatus.READY.name, "model_state": "ready"}
        )

        assert _only_replica(guard).status == LaunchStatus.READY.name

    def test_error_message_not_overwritten_by_empty(self):
        guard = _make_guard()
        guard.update_replica_status(
            "m1", 0, {"status": LaunchStatus.ERROR.name, "error_message": "boom"}
        )
        guard.update_replica_status("m1", 0, {"error_message": None})

        assert _only_replica(guard).error_message == "boom"

    def test_error_message_can_be_refined(self):
        """A better message may still replace an earlier one."""
        guard = _make_guard()
        guard.update_replica_status(
            "m1", 0, {"status": LaunchStatus.ERROR.name, "error_message": "boom"}
        )
        guard.update_replica_status("m1", 0, {"error_message": "CUDA out of memory"})

        assert _only_replica(guard).error_message == "CUDA out of memory"

    def test_non_error_replica_is_unaffected(self):
        guard = _make_guard()
        guard.update_replica_status("m1", 0, {"status": LaunchStatus.CREATING.name})
        guard.update_replica_status(
            "m1",
            0,
            {"status": LaunchStatus.TERMINATING.name, "model_state": "stopping"},
        )

        assert _only_replica(guard).status == LaunchStatus.TERMINATING.name


class TestInstanceErrorMessage:
    def test_instance_info_carries_error_message(self):
        guard = _make_guard()
        guard.update_instance_info(
            "m1", {"status": LaunchStatus.ERROR.name, "error_message": "bad path"}
        )

        info = guard.get_instance_info(model_uid="m1")[0]
        assert info.status == LaunchStatus.ERROR.name
        assert info.error_message == "bad path"

    def test_error_message_defaults_to_none(self):
        info = InstanceInfo(
            model_name="m",
            model_uid="m1",
            model_version=None,
            model_ability=[],
            replica=1,
            status=LaunchStatus.CREATING.name,
            instance_created_ts=0,
        )
        assert info.error_message is None
        # Additive field must show up in the serialized payload.
        assert "error_message" in info.dict()


class TestFailedInstanceLookup:
    """``_get_failed_instance_error`` is what lets /progress report a failure."""

    @staticmethod
    def _supervisor(guard: StatusGuardActor) -> SupervisorActor:
        supervisor = SupervisorActor.__new__(SupervisorActor)

        # xoscar actor refs return awaitables; the raw actor's methods are
        # plain sync calls, so wrap the one method under test.
        class _Ref:
            async def get_instance_info(self, model_name=None, model_uid=None):
                return guard.get_instance_info(
                    model_name=model_name, model_uid=model_uid
                )

        supervisor._status_guard_ref = _Ref()  # type: ignore[attr-defined]
        return supervisor

    def test_returns_instance_level_message(self):
        guard = _make_guard()
        guard.update_instance_info(
            "m1", {"status": LaunchStatus.ERROR.name, "error_message": "bad path"}
        )
        supervisor = self._supervisor(guard)

        result = asyncio.run(supervisor._get_failed_instance_error("m1"))
        assert result == "bad path"

    def test_falls_back_to_replica_messages(self):
        guard = _make_guard()
        guard.update_replica_status(
            "m1", 0, {"status": LaunchStatus.ERROR.name, "error_message": "cuda oom"}
        )
        guard.update_instance_info("m1", {"status": LaunchStatus.ERROR.name})
        supervisor = self._supervisor(guard)

        result = asyncio.run(supervisor._get_failed_instance_error("m1"))
        assert result == "cuda oom"

    def test_generic_message_when_nothing_recorded(self):
        guard = _make_guard()
        guard.update_instance_info("m1", {"status": LaunchStatus.ERROR.name})
        supervisor = self._supervisor(guard)

        assert asyncio.run(supervisor._get_failed_instance_error("m1")) is not None

    def test_returns_none_when_not_failed(self):
        guard = _make_guard()
        supervisor = self._supervisor(guard)

        assert asyncio.run(supervisor._get_failed_instance_error("m1")) is None

    def test_returns_none_for_unknown_model(self):
        supervisor = self._supervisor(_make_guard())

        assert asyncio.run(supervisor._get_failed_instance_error("nope")) is None


class TestAsyncLaunchCallback:
    """wait_ready=False used to drop the exception entirely."""

    def test_consumes_and_logs_exception(self, caplog):
        async def _run():
            async def failing():
                raise RuntimeError("engine init failed")

            task = asyncio.ensure_future(failing())
            with pytest.raises(RuntimeError):
                await task
            callback_for_async_launch("m1", task)
            # Retrieved, so no "exception was never retrieved" warning can fire.
            assert task.exception() is not None

        with caplog.at_level("ERROR"):
            asyncio.run(_run())
        assert "async launch failed" in caplog.text

    def test_handles_success(self):
        async def _run():
            async def ok():
                return "done"

            task = asyncio.ensure_future(ok())
            await task
            callback_for_async_launch("m1", task)

        asyncio.run(_run())

    def test_handles_cancellation(self):
        async def _run():
            async def forever():
                await asyncio.sleep(3600)

            task = asyncio.ensure_future(forever())
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # Must not raise CancelledError out of the callback.
            callback_for_async_launch("m1", task)

        asyncio.run(_run())
