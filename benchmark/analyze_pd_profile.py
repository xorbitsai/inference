# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Summarize opt-in Xavier stage logs; nested stage times must not be added.

Set XINFERENCE_XAVIER_PROFILE=1 in the model process environment, then run a
separate diagnostic workload. GPU copies synchronize and logging adds overhead:
never use this diagnostic run for headline throughput comparisons.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

PREFIX = "Xavier profile: "


def summarize(lines):
    stages = defaultdict(list)
    for line in lines:
        if PREFIX not in line:
            continue
        event, _ = json.JSONDecoder().raw_decode(line.split(PREFIX, 1)[1])
        stages[event["stage"]].append(event)
    summary = {}
    for stage, events in sorted(stages.items()):
        completed = [event for event in events if event["succeeded"]]
        seconds = sum(event["elapsed_s"] for event in completed)
        summary[stage] = {
            "calls": len(completed),
            "failed_calls": len(events) - len(completed),
            "total_s": seconds,
            "mean_ms": seconds * 1000 / len(completed) if completed else None,
            "bytes": sum(event.get("nbytes", 0) for event in completed),
            "blocks": sum(event.get("blocks", 0) for event in completed),
        }
    return {
        "stages": summary,
        "timing_semantics": {
            "load_rpc": "Engine wait including actor_control, actor_receive, serialization and concatenation.",
            "actor_receive": "Includes gloo_receive, CPU buffer allocation and thread scheduling.",
            "gloo_receive": "Blocking receive wait, including any wait for the sender; not isolated wire time.",
            "store_d2h": "GPU gather and synchronized device-to-host copy, excluding preceding GPU work.",
            "load_h2d": "CPU dtype conversion, synchronized host-to-device copy and cache write.",
            "store_rpc": "Producer engine wait for staging RPC, including serialization and CPU snapshot storage.",
            "actor_control": "Lookup, availability check and send-start RPCs per block batch.",
        },
        "warning": "Nested times overlap. Sum across concurrent requests is not wall-clock latency. Profiling synchronizes CUDA and adds logging overhead.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.log.open() as log:
        result = summarize(log)
    if not result["stages"]:
        parser.error("no Xavier profile events found")
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
