# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
import importlib.util
import json
from pathlib import Path

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
