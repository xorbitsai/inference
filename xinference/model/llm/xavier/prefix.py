# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Portable identities for complete, causal shared-prefix pages.

The byte chain matches SGLang's HiCache storage interface: SHA256 of the prior
32-byte digest and uint32 little-endian token IDs. Model semantics live in the
storage actor's validated KVCacheContract; request namespaces seed the chain.
"""

import hashlib
import struct
from typing import Optional, Sequence

from ....cache_isolation import validate_cache_salt


def prefix_keys(
    tokens: Sequence[int],
    block_size: int = 64,
    cache_salt: Optional[str] = None,
    extra_key: Optional[str] = None,
) -> list[str]:
    validate_cache_salt(cache_salt)
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("Invalid shared-prefix page size")
    previous = b""
    if extra_key is not None or cache_salt is not None:
        seed = hashlib.sha256(b"sglang-cache-namespace-v1")
        for part in (extra_key, cache_salt):
            if part is None:
                seed.update(b"\x00")
            else:
                value = part.encode("utf-8")
                seed.update(b"\x01" + len(value).to_bytes(8, "little") + value)
        previous = seed.digest()
    result = []
    for offset in range(0, len(tokens) - block_size + 1, block_size):
        page = tokens[offset : offset + block_size]
        if any(type(token) is not int or not 0 <= token < 2**32 for token in page):
            raise ValueError("Invalid shared-prefix token ID")
        previous = hashlib.sha256(
            previous + struct.pack(f"<{block_size}I", *page)
        ).digest()
        result.append(previous.hex())
    return result
