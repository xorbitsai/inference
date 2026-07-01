# Copyright 2022-2026 XProbe Inc.
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
        # engine -> hash -> (rank, block_id)
        self._hash_to_rank_and_block_id: Dict[int, Dict[int, Set[Tuple[int, int]]]] = {}  # type: ignore
        # engine -> rank -> (hash, block_id)
        self._rank_to_hash_and_block_id: Dict[int, Dict[int, Set[Tuple[int, int]]]] = {}  # type: ignore
        self._unavailable_ranks: Set[int] = set()  # type: ignore

    def register_blocks(
        self, virtual_engine: int, block_infos: List[Tuple[int, int]], rank: int
    ):
        # Update query meta
        if virtual_engine not in self._hash_to_rank_and_block_id:
            self._hash_to_rank_and_block_id[virtual_engine] = {}
        hash_to_rank_and_block_id = self._hash_to_rank_and_block_id[virtual_engine]
        for hash_content, block_id in block_infos:
            if hash_content not in hash_to_rank_and_block_id:
                hash_to_rank_and_block_id[hash_content] = {
                    (rank, block_id),
                }
            else:
                hash_to_rank_and_block_id[hash_content].add((rank, block_id))

        # Update remove meta
        if virtual_engine not in self._rank_to_hash_and_block_id:
            self._rank_to_hash_and_block_id[virtual_engine] = {}
        rank_to_hash_and_block_id = self._rank_to_hash_and_block_id[virtual_engine]
        if rank not in rank_to_hash_and_block_id:
            rank_to_hash_and_block_id[rank] = set()
        rank_to_hash_and_block_id[rank].update(block_infos)

    def query_blocks(
        self, virtual_engine: int, hash_contents: List[Tuple[int, int]]
    ) -> Dict[int, Set[Tuple[int, int, int]]]:
        if virtual_engine not in self._hash_to_rank_and_block_id:
            return {}
        hash_to_rank_and_block_id = self._hash_to_rank_and_block_id[virtual_engine]
        remote: Dict[int, Set[Tuple[int, int, int]]] = {}
        for hash_content, _id in hash_contents:
            if (
                hash_content in hash_to_rank_and_block_id
            ) and hash_to_rank_and_block_id[hash_content]:
                # exclude ranks that are in the recovery process
                rank_and_block_id = [
                    (r, b)
                    for r, b in hash_to_rank_and_block_id[hash_content]
                    if r not in self._unavailable_ranks
                ]
                if rank_and_block_id:
                    # TODO: Randomly select here, and try to distribute requests as evenly as possible.
                    # There may be better methods in the future.
                    rank, block_id = random.choice(rank_and_block_id)
                    if rank not in remote:
                        remote[rank] = {
                            (hash_content, block_id, _id),
                        }
                    else:
                        remote[rank].add((hash_content, block_id, _id))
        return remote

    def unregister_block(self, virtual_engine: int, rank: int, block_id: int):
        if (virtual_engine not in self._rank_to_hash_and_block_id) or (
            virtual_engine not in self._hash_to_rank_and_block_id
        ):
            return

        # Update remove meta
        rank_to_hash_and_block_id = self._rank_to_hash_and_block_id[virtual_engine]
        if rank not in rank_to_hash_and_block_id:
            return
        hash_and_block_id = rank_to_hash_and_block_id[rank]
        detail: Optional[Tuple[int, int]] = None
        for hash_content, _id in hash_and_block_id.copy():
            if _id == block_id:
                detail = (hash_content, block_id)
                hash_and_block_id.discard(detail)
                break

        # Update query meta
        if detail is not None:
            hash_to_rank_and_block_id = self._hash_to_rank_and_block_id[virtual_engine]
            _hash = detail[0]
            if _hash in hash_to_rank_and_block_id:
                hash_to_rank_and_block_id[_hash].discard((rank, detail[1]))

    def unregister_rank(self, rank: int):
        """
        This rank is in the recovery process, and its query results will be excluded.
        """
        self._unavailable_ranks.add(rank)

    def register_rank(self, rank: int):
        """
        After recovery is successful, clear all stale data of the rank and mark the rank as available.
        """
        for _, rank_to_hash_and_block_id in self._rank_to_hash_and_block_id.items():
            rank_to_hash_and_block_id.pop(rank, None)

        for _, hash_to_rank_and_block_id in self._hash_to_rank_and_block_id.items():
            for _, rank_and_block_id in hash_to_rank_and_block_id.items():
                to_delete = [(r, b) for r, b in rank_and_block_id if r == rank]
                if to_delete:
                    rank_and_block_id.difference_update(to_delete)

        self._unavailable_ranks.discard(rank)
