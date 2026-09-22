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

"""Explicit maintenance-time config collection; never called during recommendation.

Writes a fresh staging catalog and a coverage report for review. No weights or
remote Python code are downloaded. GGUF fallback reads only bounded metadata
headers, not tensor data. Config snapshots can be reused across runs.
"""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

from .memory_metadata import ModelMemoryMetadata


def config_url(hub: str, repo: str, revision: str) -> str:
    repo = quote(repo, safe="/")
    revision = quote(revision, safe="")
    if hub == "huggingface":
        return f"https://huggingface.co/{repo}/resolve/{revision}/config.json"
    if hub == "modelscope":
        return f"https://modelscope.cn/models/{repo}/resolve/{revision}/config.json"
    if hub == "openmind_hub":
        return f"https://modelers.cn/api/v1/file/{repo}/{revision}/media/config.json"
    if hub == "csghub":
        return f"https://hub.opencsg.com/api/v1/models/{repo}/resolve/config.json?ref={revision}"
    raise ValueError(f"Unsupported hub: {hub}")


def collect_config(job: tuple, cache_dir: Path, timeout: float) -> dict:
    import requests

    hub, repo, revision = job
    cache = cache_dir / (hashlib.sha256(json.dumps(job).encode()).hexdigest() + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    url = config_url(hub, repo, revision)
    try:
        with requests.get(url, timeout=(5, timeout), stream=True) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > 2 * 1024**2:
                    raise ValueError("config.json exceeds 2 MiB")
            config = json.loads(content)
            if not isinstance(config, dict):
                raise ValueError("config.json must contain an object")
            commit = response.headers.get("x-repo-commit")
            source = config_url(hub, repo, commit) if commit else url
            result = {
                "config": config,
                "config_source": source,
                "config_sha256": hashlib.sha256(content).hexdigest(),
            }
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    except (requests.RequestException, ValueError) as exc:
        # Failures are not cached: retrying collection can recover transient errors.
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect_gguf_config(
    job: tuple, filename: str, cache_dir: Path, timeout: float
) -> dict:
    import requests
    from urllib3.exceptions import HTTPError

    from .gguf_memory_metadata import read_gguf_config

    cache = cache_dir / (
        hashlib.sha256(json.dumps((job, filename)).encode()).hexdigest() + ".json"
    )
    if cache.exists():
        return json.loads(cache.read_text())
    hub, repo, revision = job
    url = config_url(hub, repo, revision).replace(
        "config.json", quote(filename, safe="/")
    )
    try:
        with requests.get(
            url,
            timeout=(5, timeout),
            stream=True,
            headers={"Range": "bytes=0-16777215", "Accept-Encoding": "identity"},
        ) as response:
            response.raise_for_status()
            # read_gguf_config stops at the end of key/value metadata, before
            # tensor descriptors or payloads even when the server ignores Range.
            config, digest, consumed = read_gguf_config(response.raw)
            commit = response.headers.get("x-repo-commit")
            source = (
                config_url(hub, repo, commit).replace(
                    "config.json", quote(filename, safe="/")
                )
                if commit
                else url
            )
            result = {
                "config": config,
                "config_source": source,
                "config_sha256": digest,
                "header_bytes": consumed,
            }
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    except (requests.RequestException, HTTPError, ValueError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def collect_catalog(
    catalog: Path, output: Path, cache: Path, workers: int, timeout: float
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    cache.mkdir(parents=True, exist_ok=True)
    documents = {
        p.name: json.loads(p.read_text()) for p in sorted(catalog.glob("*.json"))
    }
    jobs: set[tuple[str, str, str]] = set()
    entries = []
    for filename, families in documents.items():
        for family in families:
            for spec in family["model_specs"]:
                for hub, source in spec["model_src"].items():
                    revision = source.get("model_revision") or (
                        "master" if hub == "modelscope" else "main"
                    )
                    quant_jobs = {
                        quant: (
                            hub,
                            source["model_id"].replace("{quantization}", quant),
                            revision,
                        )
                        for quant in source["quantizations"]
                    }
                    jobs.update(quant_jobs.values())
                    entries.append((filename, family, spec, hub, source, quant_jobs))
    print(
        f"Collecting {len(jobs)} distinct configs for {len(entries)} sources",
        flush=True,
    )
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(collect_config, job, cache, timeout): job
            for job in sorted(jobs)
        }
        for future in as_completed(pending):
            results[pending[future]] = future.result()
            if len(results) % 50 == 0 or len(results) == len(jobs):
                print(
                    f"Collected {len(results)}/{len(jobs)}; readable configs: {sum('config' in r for r in results.values())}",
                    flush=True,
                )
    fallback = {}
    for _, _, spec, _, source, quant_jobs in entries:
        if spec["model_format"] != "ggufv2":
            continue
        for quant, job in quant_jobs.items():
            try:
                ModelMemoryMetadata.from_config(results[job].get("config", {}))
                continue
            except ValueError:
                pass
            if job in fallback:
                continue
            combined = dict(spec, **source)
            parts = combined.get("quantization_parts", {}).get(quant)
            template = (
                combined.get("model_file_name_split_template")
                if parts
                else combined.get("model_file_name_template")
            )
            if not template:
                continue
            fallback[job] = template.format(
                quantization=quant, part=parts[0] if parts else ""
            )
    print(f"Checking {len(fallback)} GGUF metadata headers", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(collect_gguf_config, job, filename, cache, timeout): job
            for job, filename in fallback.items()
        }
        done = 0
        for future in as_completed(pending):
            result = future.result()
            results[pending[future]] = result
            done += 1
            if done % 25 == 0 or done == len(pending):
                print(f"GGUF headers checked {done}/{len(pending)}", flush=True)
    report = []
    for filename, family, spec, hub, source, quant_jobs in entries:
        by_quant = {}
        for quant, job in quant_jobs.items():
            result = results[job]
            row = {
                "file": filename,
                "model": family["model_name"],
                "format": spec["model_format"],
                "size": spec["model_size_in_billions"],
                "hub": hub,
                "quantization": quant,
                "model_id": job[1],
                "revision": job[2],
            }
            try:
                if "error" in result:
                    raise ValueError(result["error"])
                metadata = ModelMemoryMetadata.from_config(result["config"])
                metadata.config_source = result["config_source"]
                metadata.config_sha256 = result["config_sha256"]
                if (
                    spec.get("activated_size_in_billions")
                    and not metadata.unsupported_reason
                ):
                    metadata.unsupported_reason = "moe"
                # Catalog abilities cover older multimodal configs without nested text_config.
                if set(family["model_ability"]) & {"vision", "audio", "omni"}:
                    metadata.unsupported_reason = "multimodal"
                by_quant[quant] = metadata.dict(exclude_none=True)
                row["status"] = (
                    "metadata_only" if metadata.unsupported_reason else "collected"
                )
                if metadata.unsupported_reason:
                    row["reason"] = metadata.unsupported_reason
            except ValueError as exc:
                row.update(status="unavailable", reason=str(exc))
            report.append(row)
        if not by_quant:
            continue
        # Template repos can have different dimensions at each quantization.
        # Store exact matches rather than copying one quantization's config to all.
        if len(set(quant_jobs.values())) == 1:
            source["memory_estimation"] = next(iter(by_quant.values()))
        else:
            source["memory_estimation_by_quantization"] = by_quant
    staging = output / "models"
    staging.mkdir()
    for filename, families in documents.items():
        (staging / filename).write_text(
            json.dumps(families, ensure_ascii=False, indent=2) + "\n"
        )
    (output / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    from collections import Counter

    print(dict(Counter(r["status"] for r in report)), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument(
        "output", type=Path, help="New staging directory; must not exist"
    )
    parser.add_argument(
        "--cache", type=Path, required=True, help="Config-only snapshot cache"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=15)
    args = parser.parse_args()
    collect_catalog(args.catalog, args.output, args.cache, args.workers, args.timeout)


if __name__ == "__main__":
    main()
