#!/usr/bin/env python3
# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline correlation of probe reports and bounded runtime summary log records.

No network, credentials or transcript extraction. Native cancellation is verified
only with both ends of an ASR call surrounding server-side cancellation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


def number(value):
    return type(value) in (float, int) and math.isfinite(value)


def load_summaries(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 2097152 or "diagnostics=" not in line:
                continue
            try:
                value = json.loads(line.split("diagnostics=", 1)[1])
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                yield value


def stages(summary):
    events = summary.get("events", [])
    return sorted(
        (
            event
            for event in events
            if isinstance(event, dict) and number(event.get("at_ms"))
        ),
        key=lambda event: event["at_ms"],
    )


def analyze(probes, summaries):
    indexed = defaultdict(list)
    for summary in summaries:
        trace_id = summary.get("trace_id")
        if isinstance(trace_id, str) and re.fullmatch(r"[0-9a-f]{16}", trace_id):
            indexed[trace_id].append(summary)
    reports, groups = [], defaultdict(list)
    for probe in probes:
        trace_id = probe.get("trace_id")
        if not isinstance(trace_id, str) or not re.fullmatch(r"[0-9a-f]{16}", trace_id):
            trace_id = None
        matches = indexed.get(trace_id, [])
        summary = matches[0] if len(matches) == 1 else {}
        events = stages(summary)
        timeline = {event.get("stage"): event["at_ms"] for event in events}
        replica = summary.get("replica_model_uid", "unverified")
        if not isinstance(replica, str) or not re.fullmatch(r"[\w.:-]{1,128}", replica):
            replica = "unverified"
        item = {
            "trace_id": trace_id,
            "server_association": "matched" if summary else "missing_or_ambiguous",
            "replica": replica,
            "native_cancel_verified": False,
            "stages_ms": {},
        }
        active = None
        for event in events:
            if event.get("stage") == "native_start":
                active = event["at_ms"]
            elif event.get("stage") == "native_end" and active is not None:
                cancelled = timeline.get("cancel_processed")
                if (
                    probe.get("scenario") == "cancel"
                    and probe.get("passed") is True
                    and summary.get("reason") == "cancelled"
                    and cancelled is not None
                    and active < cancelled < event["at_ms"]
                ):
                    item["native_cancel_verified"] = True
                active = None
        for name, start, end in (
            ("input_end_to_drained", "input_end", "input_drained"),
            ("finish", "finish_start", "finish_end"),
            ("final_enqueue_wait", "final_enqueue_start", "final_enqueued"),
            ("input_end_to_outputs_sent", "input_end", "outputs_sent"),
            ("cancel_to_closed", "cancel_processed", "closed"),
        ):
            if start in timeline and end in timeline:
                item["stages_ms"][name] = timeline[end] - timeline[start]
        reports.append(item)
        duration = probe.get("audio_duration_ms")
        scenario = probe.get("scenario")
        if scenario not in {
            "normal",
            "silence",
            "cancel",
            "disconnect",
            "odd_sample",
            "invalid_config",
        }:
            scenario = "unknown"
        key = (replica, duration if number(duration) else None, scenario)
        groups[key].append(probe)
    statistics = []
    for (replica, duration, scenario), runs in groups.items():
        values = sorted(
            r["end_to_final_ms"]
            for r in runs
            if r.get("passed") is True and number(r.get("end_to_final_ms"))
        )
        statistics.append(
            {
                "replica": replica,
                "audio_duration_ms": duration,
                "scenario": scenario,
                "runs": len(runs),
                "failed": sum(r.get("passed") is not True for r in runs),
                "samples": len(values),
                "p50_end_to_final_ms": (
                    values[math.ceil(len(values) * 0.5) - 1] if values else None
                ),
                "p95_end_to_final_ms": (
                    values[math.ceil(len(values) * 0.95) - 1]
                    if len(values) >= 20 and replica != "unverified"
                    else None
                ),
            }
        )
    return {
        "reports": reports,
        "groups": statistics,
        "limitations": [
            "p95 requires >=20 valid samples per replica/duration/scenario; failures are retained",
            "warm/cold and load classes require separate input reports",
            "missing native interval endpoints cannot prove inference-stage cancellation",
            "stage intervals overlap; do not add them or subtract clocks across nodes",
            "client disconnect/cancel success alone does not certify native cleanup or quality",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-report", required=True, type=Path)
    parser.add_argument("--runtime-log", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--require-native-cancel", action="store_true")
    args = parser.parse_args()
    probes = json.loads(args.probe_report.read_text(encoding="utf-8"))
    result = analyze(probes["reports"], load_summaries(args.runtime_log))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.require_native_cancel:
        cancellations = [
            item
            for item, probe in zip(result["reports"], probes["reports"])
            if probe.get("scenario") == "cancel"
        ]
        return (
            0
            if cancellations
            and all(item["native_cancel_verified"] for item in cancellations)
            else 1
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
