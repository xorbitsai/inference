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
from typing import Optional
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .memory import unsupported_memory_reason
from .model_metadata import ModelMetadata


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
                "resolved_revision": commit or revision,
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

    from .gguf_model_metadata import read_gguf_config

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
                "resolved_revision": commit or revision,
            }
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result
    except (requests.RequestException, HTTPError, ValueError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def gguf_filenames(spec: dict, source: dict, quant: str) -> tuple:
    combined = dict(spec, **source)
    parts = combined.get("quantization_parts", {}).get(quant)
    template = (
        combined.get("model_file_name_split_template")
        if parts
        else combined.get("model_file_name_template")
    )
    if not template:
        return (), ()
    return (
        tuple(
            template.format(quantization=quant, part=part) for part in (parts or [""])
        ),
        tuple(
            name.format(quantization=quant)
            for name in combined.get("multimodal_projectors", [])
        ),
    )


def checkpoint_weight_job(
    spec: dict, source: dict, quant: str, job: tuple, result: dict
) -> Optional[tuple]:
    if spec["model_format"] == "bnb":
        return None
    revision = result.get("resolved_revision")
    if not revision:
        # Older successful config snapshots already contain the resolved URL,
        # but predate resolved_revision. Do not mix them with today's main.
        url = result.get("config_source", "")
        if job[0] == "csghub":
            revision = parse_qs(urlsplit(url).query).get("ref", [None])[0]
        else:
            prefix = config_url(job[0], job[1], "__revision__").split("__revision__")[0]
            if url.startswith(prefix):
                revision = unquote(url[len(prefix) :].split("/", 1)[0])
    pinned = (job[0], job[1], revision or job[2])
    if spec["model_format"] != "ggufv2":
        return pinned
    files, projectors = gguf_filenames(spec, source, quant)
    if not files:
        return None
    return pinned + (files, projectors)


def collect_catalog(
    catalog: Path,
    output: Path,
    cache: Path,
    workers: int,
    timeout: float,
    *,
    weights: bool = False,
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
                ModelMetadata.from_config(results[job].get("config", {}))
                continue
            except ValueError:
                pass
            if job in fallback:
                continue
            files, _ = gguf_filenames(spec, source, quant)
            if not files:
                continue
            fallback[job] = files[0]
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
    weight_results = {}
    weight_jobs_by_spec = {
        (index, quant): checkpoint_weight_job(spec, source, quant, job, results[job])
        for index, (_, _, spec, _, source, quant_jobs) in enumerate(entries)
        for quant, job in quant_jobs.items()
        if "config" in results[job]
    }
    if weights:
        from .weights_metadata import collect_weights_size

        weight_jobs = {job for job in weight_jobs_by_spec.values() if job is not None}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            weight_pending = {
                pool.submit(collect_weights_size, job, cache, timeout): job
                for job in weight_jobs
            }
            for future in as_completed(weight_pending):
                weight_results[weight_pending[future]] = future.result()
    for index, (filename, family, spec, hub, source, quant_jobs) in enumerate(entries):
        old_entries = source.get("model_metadata_by_quantization")
        by_quant = {
            q: old_entries[q] for q in quant_jobs if old_entries and q in old_entries
        }
        if not old_entries and source.get("model_metadata"):
            by_quant = dict.fromkeys(quant_jobs, source["model_metadata"])
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
                metadata = ModelMetadata.from_config(result["config"])
                metadata.config_source = result["config_source"]
                metadata.config_sha256 = result["config_sha256"]
                if (
                    spec.get("activated_size_in_billions")
                    and not metadata.architecture_type
                ):
                    metadata.architecture_type = "moe"
                # Catalog abilities cover older multimodal configs without nested text_config.
                if (
                    set(family["model_ability"]) & {"vision", "audio", "omni"}
                    and not metadata.architecture_type
                ):
                    metadata.architecture_type = "multimodal"
                    metadata.layer_types = [
                        "full_attention"
                    ] * metadata.num_hidden_layers
                previous = by_quant.get(quant, {})
                if (
                    metadata.config_sha256
                    and previous.get("config_sha256") == metadata.config_sha256
                    and previous.get("config_source") == metadata.config_source
                ):
                    for name in ("weights_bytes", "weights_source", "weights_sha256"):
                        if name in previous:
                            setattr(metadata, name, previous[name])
                weight_job = weight_jobs_by_spec.get((index, quant))
                weight_info = (
                    weight_results.get(weight_job) if weight_job is not None else None
                )
                if weight_info is not None:
                    row["weights_status"] = (
                        "unavailable" if "error" in weight_info else "collected"
                    )
                    if "error" in weight_info:
                        row["weights_reason"] = weight_info["error"]
                    else:
                        for name, value in weight_info.items():
                            setattr(metadata, name, value)
                by_quant[quant] = metadata.dict(exclude_none=True)
                reason = unsupported_memory_reason(metadata)
                row["status"] = "metadata_only" if reason else "collected"
                if reason:
                    row["reason"] = reason
            except ValueError as exc:
                row.update(status="unavailable", reason=str(exc))
            report.append(row)
        if not by_quant:
            continue
        # Template repos can have different dimensions at each quantization.
        # Store exact matches rather than copying one quantization's config to all.
        # One GGUF config can describe several quantizations with different
        # resident weight sizes. Share facts only when the entire object agrees.
        distinct = {json.dumps(value, sort_keys=True) for value in by_quant.values()}
        if len(set(quant_jobs.values())) == 1 and len(distinct) == 1:
            source["model_metadata"] = next(iter(by_quant.values()))
            source.pop("model_metadata_by_quantization", None)
        else:
            source["model_metadata_by_quantization"] = by_quant
            source.pop("model_metadata", None)
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
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Collect bounded Safetensors index/header sizes at the same config revision",
    )
    args = parser.parse_args()
    collect_catalog(
        args.catalog,
        args.output,
        args.cache,
        args.workers,
        args.timeout,
        weights=args.weights,
    )


if __name__ == "__main__":
    main()
