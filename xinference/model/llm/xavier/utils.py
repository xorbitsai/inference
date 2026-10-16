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
import struct
from functools import lru_cache
from typing import List, Optional, Tuple

import xxhash


def hash_block_tokens(
    is_first_block: bool,
    prev_block_hash: Optional[int],
    cur_block_token_ids: List[int],
    extra_hash: Optional[int] = None,
    none_hash: int = -1,
) -> int:
    """
    Computes a stable hash value corresponding to the contents of a block
    and the contents of the preceding block(s), using xxhash instead of
    Python hash() for cross-process stability.
    """

    if prev_block_hash is None:
        prev_block_hash = none_hash
    if extra_hash is None:
        extra_hash = none_hash

    return _hash_block_tokens(
        is_first_block,
        int(prev_block_hash) & 0xFFFFFFFFFFFFFFFF,
        tuple(cur_block_token_ids),
        int(extra_hash) & 0xFFFFFFFFFFFFFFFF,
    )


@lru_cache(maxsize=16384)
def _hash_block_tokens(
    is_first_block: bool,
    prev_block_hash: int,
    cur_block_token_ids: Tuple[int, ...],
    extra_hash: int,
) -> int:
    # Cache whole hash inputs, including the predecessor. Identical token
    # chunks at different prefix positions must not share a content address.
    # The bounded cache retains no request or tensor storage.

    buf = bytearray()

    # 0. hash version
    buf += b"v1"

    # 1. is_first_block: bool -> 1 byte
    buf += struct.pack("<?", is_first_block)

    # 2. prev_block_hash: int64
    buf += struct.pack("<Q", int(prev_block_hash) & 0xFFFFFFFFFFFFFFFF)

    # 3. token count: uint32
    buf += struct.pack("<I", len(cur_block_token_ids))

    # 4. token ids: int32[]
    buf += struct.pack(
        f"<{len(cur_block_token_ids)}i", *(int(tid) for tid in cur_block_token_ids)
    )

    # 5. extra_hash: int64
    buf += struct.pack("<Q", int(extra_hash) & 0xFFFFFFFFFFFFFFFF)

    # xxhash64, fixed seed for determinism
    return xxhash.xxh64_intdigest(bytes(buf), seed=0)
