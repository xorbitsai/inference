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
        if virtual_engine not in self._hash_to_rank_and_block_id:
            self._hash_to_rank_and_block_id[virtual_engine] = {}
        hash_to_rank_and_block_id = self._hash_to_rank_and_block_id[virtual_engine]

        if virtual_engine not in self._rank_to_hash_and_block_id:
            self._rank_to_hash_and_block_id[virtual_engine] = {}
        rank_to_hash_and_block_id = self._rank_to_hash_and_block_id[virtual_engine]
        if rank not in rank_to_hash_and_block_id:
            rank_to_hash_and_block_id[rank] = set()
        hash_and_block_id = rank_to_hash_and_block_id[rank]

        # Replace-on-register: a vLLM block id can be reused for new content
        # (its KV cache slot is reallocated), so before registering the new
        # (hash, block_id) mappings, drop any stale (old_hash, block_id) this
        # rank still holds for the same block ids. Without this a stale hash
        # keeps pointing at a block that now holds different KV, and a later
        # query on that hash would load the wrong KV. This also bounds the
        # tracker to one entry per (rank, block_id).
        new_block_ids = {block_id for _, block_id in block_infos}
        new_pairs = set(block_infos)
        stale = [
            (stale_hash, stale_block_id)
            for stale_hash, stale_block_id in hash_and_block_id
            if stale_block_id in new_block_ids
            and (stale_hash, stale_block_id) not in new_pairs
        ]
        for stale_hash, stale_block_id in stale:
            hash_and_block_id.discard((stale_hash, stale_block_id))
            stale_entries = hash_to_rank_and_block_id.get(stale_hash)
            if stale_entries is not None:
                stale_entries.discard((rank, stale_block_id))
                if not stale_entries:
                    hash_to_rank_and_block_id.pop(stale_hash, None)

        for hash_content, block_id in block_infos:
            hash_to_rank_and_block_id.setdefault(hash_content, set()).add(
                (rank, block_id)
            )
            hash_and_block_id.add((hash_content, block_id))

    def query_blocks(
        self,
        virtual_engine: int,
        hash_contents: List[Tuple[int, int]],
        exclude_rank: Optional[int] = None,
    ) -> Dict[int, Set[Tuple[int, int, int]]]:
        if virtual_engine not in self._hash_to_rank_and_block_id:
            return {}
        hash_to_rank_and_block_id = self._hash_to_rank_and_block_id[virtual_engine]
        remote: Dict[int, Set[Tuple[int, int, int]]] = {}
        for hash_content, _id in hash_contents:
            if (
                hash_content in hash_to_rank_and_block_id
            ) and hash_to_rank_and_block_id[hash_content]:
                # exclude ranks that are in the recovery process or the caller's own rank
                rank_and_block_id = [
                    (r, b)
                    for r, b in hash_to_rank_and_block_id[hash_content]
                    if r not in self._unavailable_ranks and r != exclude_rank
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
        self.unregister_blocks(virtual_engine, rank, [block_id])

    def unregister_blocks(self, virtual_engine: int, rank: int, block_ids: List[int]):
        entries = self._rank_to_hash_and_block_id.get(virtual_engine, {}).get(rank)
        hashes = self._hash_to_rank_and_block_id.get(virtual_engine, {})
        if not entries:
            return
        removed_ids = set(block_ids)
        removed = {
            (content_hash, block_id)
            for content_hash, block_id in entries
            if block_id in removed_ids
        }
        entries.difference_update(removed)
        for content_hash, block_id in removed:
            locations = hashes.get(content_hash)
            if locations is not None:
                locations.discard((rank, block_id))
                if not locations:
                    hashes.pop(content_hash, None)

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
