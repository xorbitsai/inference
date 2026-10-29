"""Persisted benchmark jobs. One child process runs at a time per API server."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .adapter import RUNNERS

logger = logging.getLogger(__name__)
ACTIVE_STATES = {"queued", "running"}


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_uid: str = Field(min_length=1, max_length=256)
    name: str = Field(default="", max_length=128)
    runner: str = "xinference"
    input_tokens: int = Field(default=256, ge=16, le=32768, strict=True)
    output_tokens: int = Field(default=128, ge=1, le=8192, strict=True)
    concurrency: list[int] = Field(default=[1, 4, 16], min_length=1, max_length=10)
    requests: int = Field(default=100, ge=1, le=5000, strict=True)
    warmup: int = Field(default=3, ge=0, le=10, strict=True)
    seed: int = Field(default=42, ge=0, le=2**31 - 1, strict=True)

    @field_validator("runner")
    @classmethod
    def supported_runner(cls, value):
        if value not in RUNNERS:
            raise ValueError("Unsupported benchmark runner")
        return value

    @field_validator("concurrency", mode="before")
    @classmethod
    def valid_concurrency(cls, value):
        if not isinstance(value, list) or any(
            type(v) is not int or not 1 <= v <= 256 for v in value
        ):
            raise ValueError("Concurrency must contain integers between 1 and 256")
        if len(set(value)) != len(value):
            raise ValueError("Concurrency points must be unique")
        return value

    @model_validator(mode="after")
    def enough_requests(self):
        if max(self.concurrency) > self.requests:
            raise ValueError(
                "Requests per point must be at least the maximum concurrency"
            )
        return self


def now():
    return datetime.now(timezone.utc).isoformat()


class BenchmarkService:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.tasks = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._stopping = set()
        for path in self.directory.glob("*.json"):
            try:
                job = json.loads(path.read_text())
                self.jobs[job["id"]] = job
                if job["state"] in ACTIVE_STATES:
                    job.update(
                        state="interrupted",
                        finished_at=now(),
                        error="API server restarted during the benchmark",
                    )
                    self._save(job)
            except (ValueError, KeyError, OSError):
                logger.warning("Unable to read benchmark %s", path, exc_info=True)

    def _save(self, job):
        target = self.directory / (job["id"] + ".json")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, ensure_ascii=False, allow_nan=False))
        temporary.chmod(0o600)
        os.replace(temporary, target)

    def get(self, job_id, *, raw=False):
        job = self.jobs[job_id]
        if not raw:
            job = {
                **job,
                "results": [
                    {k: v for k, v in stage.items() if k != "raw_requests"}
                    for stage in job["results"]
                ],
            }
        return copy.deepcopy(job)

    def list(self):
        return [
            self.get(j["id"])
            for j in sorted(
                self.jobs.values(), key=lambda j: j["created_at"], reverse=True
            )
        ]

    def create(self, config, snapshot, endpoint, token):
        if self._closed:
            raise RuntimeError("Benchmark service is shutting down")
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "name": config.name or config.model_uid,
            "created_at": now(),
            "started_at": None,
            "finished_at": None,
            "state": "queued",
            "config": config.model_dump(),
            "snapshot": snapshot,
            "runner": RUNNERS[config.runner]().capabilities(),
            "results": [],
            "progress": 0,
            "logs": [],
            "error": None,
        }
        self.jobs[job_id] = job
        self._save(job)
        task = asyncio.create_task(self._execute(job, endpoint, token))
        self.tasks[job_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(job_id, None))
        return self.get(job_id)

    async def _execute(self, job, endpoint, token):
        process = None
        acquired = False
        try:
            await self._lock.acquire()
            acquired = True
            job.update(state="running", started_at=now())
            self._save(job)
            # Credentials travel only over stdin, never argv or persisted JSON.
            with (self.directory / (job["id"] + ".log")).open("wb") as errors:
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "xinference.benchmark",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=errors,
                    limit=16 * 1024 * 1024,
                )
                process.stdin.write(
                    json.dumps(
                        {
                            "config": job["config"],
                            "endpoint": endpoint,
                            "token": token,
                        }
                    ).encode()
                    + b"\n"
                )
                await process.stdin.drain()

                async def collect():
                    last_progress = -1
                    async for line in process.stdout:
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        kind = event.get("type")
                        if kind == "stage":
                            job["results"].append(event["result"])
                        elif kind == "log":
                            job["logs"] = [*job["logs"][-99:], event["message"]]
                        elif kind == "progress":
                            config = job["config"]
                            progress = int(
                                100
                                * (
                                    event["stage"] * config["requests"]
                                    + event["completed"]
                                )
                                / (len(config["concurrency"]) * config["requests"])
                            )
                            if progress == last_progress:
                                continue
                            last_progress = progress
                            job["progress"] = progress
                        else:
                            continue
                        self._save(job)
                    return await process.wait()

                code = await asyncio.wait_for(collect(), timeout=3600)
                if code:
                    raise RuntimeError(
                        "Benchmark runner failed; inspect the task log and request errors"
                    )
            job.update(state="completed", progress=100)
        except asyncio.CancelledError:
            job["state"] = "interrupted" if self._closed else "stopped"
        except Exception as exc:
            logger.warning("Benchmark %s failed: %s", job["id"], exc)
            job.update(state="failed", error=str(exc))
        finally:
            if process is not None and process.returncode is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            job["finished_at"] = now()
            try:
                self._save(job)
            finally:
                if acquired:
                    self._lock.release()

    async def stop(self, job_id):
        job = self.jobs[job_id]
        task = self.tasks.get(job_id)
        if task is not None:
            if job_id not in self._stopping:
                self._stopping.add(job_id)
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._stopping.discard(job_id)
        # A queued task can be cancelled before its coroutine ever starts.
        if job["state"] in ACTIVE_STATES:
            job.update(state="stopped", finished_at=now())
            self._save(job)
        return self.get(job_id)

    def delete(self, job_id):
        if self.jobs[job_id]["state"] in ACTIVE_STATES:
            raise ValueError("Stop the benchmark before deleting it")
        self.jobs.pop(job_id)
        for suffix in (".json", ".log"):
            (self.directory / (job_id + suffix)).unlink(missing_ok=True)

    def logs(self, job_id):
        self.jobs[job_id]
        path = self.directory / (job_id + ".log")
        if not path.exists():
            return ""
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 64000))
            return stream.read().decode(errors="replace")

    async def close(self):
        self._closed = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
