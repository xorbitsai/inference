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
import functools
import ipaddress
import logging
import os
import struct
from typing import List, Optional

import xxhash

logger = logging.getLogger(__name__)


def vllm_check(fn):
    try:
        from vllm.engine.async_llm_engine import AsyncEngineDeadError
    except:
        return fn

    @functools.wraps(fn)
    async def _async_wrapper(self, *args, **kwargs):
        try:
            return await fn(self, *args, **kwargs)
        except AsyncEngineDeadError:
            logger.info("Detecting vLLM is not health, prepare to quit the process")
            try:
                self.stop()
            except:
                # ignore error when stop
                pass
            # Just kill the process and let xinference auto-recover the model
            os._exit(1)

    return _async_wrapper


def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"  # noqa E231
    else:
        return f"tcp://{ip}:{port}"  # noqa E231


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False


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
    for tid in cur_block_token_ids:
        buf += struct.pack("<i", int(tid))

    # 5. extra_hash: int64
    buf += struct.pack("<Q", int(extra_hash) & 0xFFFFFFFFFFFFFFFF)

    # xxhash64, fixed seed for determinism
    return xxhash.xxh64_intdigest(bytes(buf), seed=0)
