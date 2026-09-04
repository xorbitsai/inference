# Copyright 2022-2023 XProbe Inc.
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
"""Resolve ``oci://`` model URIs to a local path.

A model published as a CNCF ModelPack (https://github.com/modelpack/model-spec)
artifact lives in an ordinary container registry, so it reuses the registry,
credentials, mirroring and air-gap tooling a deployment already has.

Acquisition is delegated to a running ``llmman serve``
(https://github.com/llmmanorg/llmman), which already implements the ModelPack
media types, registry auth, resumable blob download and a content-addressed
store. The cache entry symlinks that store path, exactly as for ``file://``.
"""

import logging

from . import llmman

logger = logging.getLogger(__name__)

SCHEME = "oci"


def resolve_oci_model(reference: str) -> str:
    """Pull an OCI reference through llmman and return the local path.

    ``reference`` is the scheme-less address ``parse_uri`` yields, such as
    ``ghcr.io/org/model:tag``.
    """
    if not reference or not reference.strip():
        raise ValueError("OCI model reference cannot be empty")

    def _progress(status, completed, total):
        if total:
            logger.info("llmman: %s (%s/%s bytes)", status, completed, total)
        else:
            logger.info("llmman: %s", status)

    return llmman.pull_and_resolve(reference.strip(), progress=_progress)
