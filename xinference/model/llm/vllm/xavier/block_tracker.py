# Copyright 2022-2025 XProbe Inc.
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
import random
from typing import Dict, List, Optional, Set, Tuple

import xoscar as xo


class VLLMBlockTracker(xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return f"vllm-block-tracker-actor"

    def __init__(self):
        super().__init__()
        # engine -> hash_to_address_and_block_id
        self._hash_to_address_and_block_id: Dict[
            int, Dict[int, Set[Tuple[str, int]]]
        ] = {}
        # engine -> address_to_hash_and_block_id
        self._address_to_hash_and_block_id: Dict[
            int, Dict[str, Set[Tuple[int, int]]]
        ] = {}

    def register_blocks(
        self, virtual_engine: int, block_infos: List[Tuple[int, int]], address: str
    ):
        # Update query meta
        if virtual_engine not in self._hash_to_address_and_block_id:
            self._hash_to_address_and_block_id[virtual_engine] = {}
        hash_to_address_and_block_id = self._hash_to_address_and_block_id[
            virtual_engine
        ]
        for hash_content, block_id in block_infos:
            if hash_content not in hash_to_address_and_block_id:
                hash_to_address_and_block_id[hash_content] = {
                    (address, block_id),
                }
            else:
                hash_to_address_and_block_id[hash_content].add((address, block_id))

        # Update remove meta
        if virtual_engine not in self._address_to_hash_and_block_id:
            self._address_to_hash_and_block_id[virtual_engine] = {}
        address_to_hash_and_block_id = self._address_to_hash_and_block_id[
            virtual_engine
        ]
        if address not in address_to_hash_and_block_id:
            address_to_hash_and_block_id[address] = set()
        address_to_hash_and_block_id[address].update(block_infos)

    def query_blocks(
        self, virtual_engine: int, hash_contents: List[Tuple[int, int]]
    ) -> Dict[str, Set[Tuple[int, int, int]]]:
        if virtual_engine not in self._hash_to_address_and_block_id:
            return {}
        hash_to_address_and_block_id = self._hash_to_address_and_block_id[
            virtual_engine
        ]
        remote: Dict[str, Set[Tuple[int, int, int]]] = {}
        for hash_content, _id in hash_contents:
            if (
                hash_content in hash_to_address_and_block_id
            ) and hash_to_address_and_block_id[hash_content]:
                # TODO: Randomly select here, and try to distribute requests as evenly as possible.
                # There may be better methods in the future.
                address, block_id = random.choice(
                    list(hash_to_address_and_block_id[hash_content])
                )
                if address not in remote:
                    remote[address] = {
                        (hash_content, block_id, _id),
                    }
                else:
                    remote[address].add((hash_content, block_id, _id))
        return remote

    def unregister_block(self, virtual_engine: int, address: str, block_id: int):
        if (virtual_engine not in self._address_to_hash_and_block_id) or (
            virtual_engine not in self._hash_to_address_and_block_id
        ):
            return

        # Update remove meta
        address_to_hash_and_block_id = self._address_to_hash_and_block_id[
            virtual_engine
        ]
        if address not in address_to_hash_and_block_id:
            return
        hash_and_block_id = address_to_hash_and_block_id[address]
        detail: Optional[Tuple[int, int]] = None
        for hash_content, _id in hash_and_block_id.copy():
            if _id == block_id:
                detail = (hash_content, block_id)
                hash_and_block_id.discard(detail)
                break

        # Update query meta
        if detail is not None:
            hash_to_address_and_block_id = self._hash_to_address_and_block_id[
                virtual_engine
            ]
            _hash = detail[0]
            if _hash in hash_to_address_and_block_id:
                hash_to_address_and_block_id[_hash].discard((address, detail[1]))
