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

"""Bounded maintenance-time reads of checkpoint sizes, never tensor payloads."""

import hashlib
import json
import struct
from pathlib import Path
from typing import Optional


def read_safetensors_size(stream, limit: int = 8 * 1024**2) -> tuple:
    length = stream.read(8)
    if len(length) != 8:
        raise ValueError("Truncated Safetensors header")
    size = struct.unpack("<Q", length)[0]
    if size <= 0 or size > limit:
        raise ValueError("Safetensors header exceeds collection limit")
    header = stream.read(size)
    if len(header) != size:
        raise ValueError("Truncated Safetensors header")
    tensors = json.loads(header)
    if not isinstance(tensors, dict):
        raise ValueError("Safetensors header must contain an object")
    ends = []
    for name, tensor in tensors.items():
        if name == "__metadata__":
            continue
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or not 0 <= offsets[0] <= offsets[1]
        ):
            raise ValueError("Invalid Safetensors data offsets")
        ends.append(offsets[1])
    if not ends or max(ends) <= 0:
        raise ValueError("Safetensors header has no tensors")
    return max(ends), hashlib.sha256(length + header).hexdigest()


def collect_weights_size(job: tuple, cache_dir: Path, timeout: float) -> dict:
    weights_bytes: Optional[int]
    import requests
    from urllib3.exceptions import HTTPError

    from .collect_memory_catalog import config_url

    cache = cache_dir / (
        hashlib.sha256(json.dumps((job, "weights")).encode()).hexdigest() + ".json"
    )
    if cache.exists():
        return json.loads(cache.read_text())
    hub, repo, revision = job[:3]
    base = config_url(hub, repo, revision)
    try:
        if len(job) == 5:
            from urllib.parse import quote

            sizes = {}
            sources = {}
            for filename in dict.fromkeys(job[3] + job[4]):
                url = base.replace("config.json", quote(filename, safe="/"))
                with requests.head(
                    url, timeout=(5, timeout), allow_redirects=True
                ) as response:
                    response.raise_for_status()
                    value = response.headers.get("Content-Length")
                    if not value or not value.isdigit() or int(value) <= 0:
                        raise ValueError("GGUF file size is unavailable")
                    sizes[filename] = int(value)
                    sources[filename] = url
            weights_bytes = sum(sizes[name] for name in job[3])
            # Launch selects one projector. Reserve the largest registered
            # choice rather than summing mutually exclusive encoders.
            if job[4]:
                weights_bytes += max(sizes[name] for name in job[4])
            result = {
                "weights_bytes": weights_bytes,
                "weights_source": sources[job[3][0]],
                "weights_sha256": hashlib.sha256(
                    json.dumps(sizes, sort_keys=True).encode()
                ).hexdigest(),
            }
            cache.write_text(json.dumps(result, indent=2) + "\n")
            return result
        url = base.replace("config.json", "model.safetensors.index.json")
        with requests.get(url, timeout=(5, timeout), stream=True) as response:
            if response.status_code == 404:
                weights_bytes = None
            else:
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_content(65536):
                    content.extend(chunk)
                    if len(content) > 8 * 1024**2:
                        raise ValueError("Safetensors index exceeds collection limit")
                index = json.loads(content)
                if not isinstance(index, dict) or not isinstance(
                    index.get("metadata"), dict
                ):
                    raise ValueError("Safetensors index must contain metadata")
                weights_bytes = index.get("metadata", {}).get("total_size")
                if type(weights_bytes) is not int or weights_bytes <= 0:
                    raise ValueError("Safetensors index has no valid total_size")
                digest = hashlib.sha256(content).hexdigest()
        if weights_bytes is None:
            url = base.replace("config.json", "model.safetensors")
            with requests.get(
                url,
                timeout=(5, timeout),
                stream=True,
                headers={"Range": "bytes=0-8388615", "Accept-Encoding": "identity"},
            ) as response:
                response.raise_for_status()
                weights_bytes, digest = read_safetensors_size(response.raw)
        result = {
            "weights_bytes": weights_bytes,
            "weights_source": url,
            "weights_sha256": digest,
        }
        cache.write_text(json.dumps(result, indent=2) + "\n")
        return result
    except (requests.RequestException, HTTPError, ValueError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
