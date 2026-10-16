# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "analyze_pd_profile", Path(__file__).parents[1] / "analyze_pd_profile.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_nested_profile_stages_are_not_added_and_failures_are_separate():
    def event(stage, seconds, success=True):
        return (
            "logger Xavier profile: "
            + json.dumps(
                dict(stage=stage, elapsed_s=seconds, succeeded=success, nbytes=64)
            )
            + " request_id=r"
        )

    result = module.summarize(
        [
            "unrelated log",
            event("load_rpc", 0.5),
            event("actor_control", 0.3),
            event("load_rpc", 0.1, False),
        ]
    )
    assert result["stages"]["load_rpc"]["total_s"] == 0.5
    assert result["stages"]["load_rpc"]["failed_calls"] == 1
    assert result["stages"]["load_rpc"]["bytes"] == 64
    assert result["stages"]["actor_control"]["mean_ms"] == 300


@pytest.mark.parametrize(
    "payload",
    [
        '{"stage":',
        "null",
        "[]",
        '"text"',
        "42",
        "{}",
        '{"stage": []}',
        '{"stage": "load_rpc", "succeeded": true, "elapsed_s": "bad"}',
        '{"stage": "load_rpc", "succeeded": true, "elapsed_s": NaN}',
        '{"stage": "load_rpc", "succeeded": true, "elapsed_s": -1}',
        '{"stage": "load_rpc", "succeeded": true, "elapsed_s": 1, "nbytes": null}',
    ],
)
def test_invalid_events_do_not_discard_valid_measurements(payload):
    valid = dict(stage="load_rpc", elapsed_s=0.5, succeeded=True, nbytes=64)
    result = module.summarize(
        [
            "unrelated log",
            module.PREFIX + json.dumps(valid),
            module.PREFIX + payload,
            module.PREFIX + "  " + json.dumps(valid) + " trailing log context",
        ]
    )
    assert result["skipped_events"] == 1
    assert result["stages"]["load_rpc"]["calls"] == 2
    assert result["stages"]["load_rpc"]["total_s"] == 1
    assert result["stages"]["load_rpc"]["bytes"] == 128


def test_cli_tolerates_non_utf8_log_bytes(monkeypatch, tmp_path):
    import sys

    log = tmp_path / "server.log"
    output = tmp_path / "summary.json"
    event = dict(stage="load_rpc", elapsed_s=0.5, succeeded=True)
    log.write_bytes(
        b"invalid byte: \xff\n" + (module.PREFIX + json.dumps(event)).encode()
    )
    monkeypatch.setattr(sys, "argv", ["analyze", str(log), "--output", str(output)])
    module.main()
    assert json.loads(output.read_text())["stages"]["load_rpc"]["calls"] == 1
