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
import sys
from contextlib import nullcontext
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import xoscar as xo

if TYPE_CHECKING:
    from .local_directory import SharedBlockDirectory


class BlockTracker(xo.StatelessActor):
    @classmethod
    def default_uid(cls):
        return f"vllm-block-tracker-actor"

    def __init__(self):
        super().__init__()
        # engine -> hash -> (rank, block_id)
        self._hash_to_rank_and_block_id: Dict[int, Dict[int, Set[Tuple[int, int]]]] = {}  # type: ignore
        # engine -> rank -> (hash, block_id)
        self._rank_to_hash_and_block_id: Dict[int, Dict[int, Set[Tuple[int, int]]]] = {}  # type: ignore
        # engine -> rank -> block id -> content hashes. Updates must scale with
        # the incoming batch rather than scanning a rank's entire cache.
        self._rank_to_block_hashes: Dict[int, Dict[int, Dict[int, Set[int]]]] = {}
        self._unavailable_ranks: Set[int] = set()  # type: ignore
        # Immutable V1 snapshots use their content hash as their address. Keep
        # only integer keys/values on that path: per-block sets and location
        # tuples otherwise trigger collections of the serving process's heap.
        self._snapshot_ranks: Dict[int, Dict[int, Dict[int, None]]] = {}
        self._rank_snapshots: Dict[int, Dict[int, Dict[int, None]]] = {}
        self._snapshot_directories: Dict[int, "SharedBlockDirectory"] = {}
        self._snapshot_directory_disabled: Set[int] = set()

    def get_snapshot_directory(self, virtual_engine: int):
        if (
            sys.platform != "linux"
            or virtual_engine in self._snapshot_directory_disabled
        ):
            return None
        directory = self._snapshot_directories.get(virtual_engine)
        if directory is None:
            from .local_directory import SharedBlockDirectory

            try:
                directory = SharedBlockDirectory()
                with directory.update():
                    for key in self._snapshot_ranks.get(virtual_engine, {}):
                        directory.add(key)
                self._snapshot_directories[virtual_engine] = directory
            except OSError:
                return None
        try:
            return directory.metadata()
        except OSError:
            return None

    async def __pre_destroy__(self):
        for directory in self._snapshot_directories.values():
            directory.close()
        self._snapshot_directories.clear()

    def register_snapshot_blocks(self, virtual_engine: int, keys: List[int], rank: int):
        if self._rank_to_hash_and_block_id.get(virtual_engine, {}).get(rank):
            self.unregister_blocks(virtual_engine, rank, keys)
        locations = self._snapshot_ranks.setdefault(virtual_engine, {})
        owned = self._rank_snapshots.setdefault(virtual_engine, {}).setdefault(rank, {})
        directory = self._snapshot_directories.get(virtual_engine)
        with directory.update() if directory is not None else nullcontext():
            for key in keys:
                if key not in locations:
                    if directory is not None:
                        directory.add(key)
                    locations[key] = {}
                locations[key][rank] = None
                owned[key] = None

    def _unregister_snapshots(self, virtual_engine: int, rank: int, keys):
        owned = self._rank_snapshots.get(virtual_engine, {}).get(rank)
        if not owned:
            return
        locations = self._snapshot_ranks[virtual_engine]
        directory = self._snapshot_directories.get(virtual_engine)
        with directory.update() if directory is not None else nullcontext():
            for key in keys:
                if key in owned:
                    del owned[key]
                    ranks = locations[key]
                    del ranks[rank]
                    if not ranks:
                        del locations[key]
                        if directory is not None:
                            directory.remove(key)

    def register_blocks(
        self, virtual_engine: int, block_infos: List[Tuple[int, int]], rank: int
    ):
        # Physical-slot registrations are a separate protocol. Disable the
        # negative shortcut before admitting keys outside the snapshot index.
        directory = self._snapshot_directories.pop(virtual_engine, None)
        if directory is not None:
            directory.close()
        self._snapshot_directory_disabled.add(virtual_engine)
        self._unregister_snapshots(
            virtual_engine, rank, (block_id for _, block_id in block_infos)
        )
        if virtual_engine not in self._hash_to_rank_and_block_id:
            self._hash_to_rank_and_block_id[virtual_engine] = {}
        hash_to_rank_and_block_id = self._hash_to_rank_and_block_id[virtual_engine]

        if virtual_engine not in self._rank_to_hash_and_block_id:
            self._rank_to_hash_and_block_id[virtual_engine] = {}
        rank_to_hash_and_block_id = self._rank_to_hash_and_block_id[virtual_engine]
        if rank not in rank_to_hash_and_block_id:
            rank_to_hash_and_block_id[rank] = set()
        hash_and_block_id = rank_to_hash_and_block_id[rank]

        # Replace-on-register: an engine block id can be reused for new content
        # (its KV cache slot is reallocated), so before registering the new
        # (hash, block_id) mappings, drop any stale (old_hash, block_id) this
        # rank still holds for the same block ids. Without this a stale hash
        # keeps pointing at a block that now holds different KV, and a later
        # query on that hash would load the wrong KV. This also bounds the
        # tracker to one entry per (rank, block_id).
        by_block = self._rank_to_block_hashes.setdefault(virtual_engine, {}).setdefault(
            rank, {}
        )
        incoming: Dict[int, Set[int]] = {}
        for content_hash, block_id in block_infos:
            incoming.setdefault(block_id, set()).add(content_hash)
        for block_id, new_hashes in incoming.items():
            for stale_hash in by_block.get(block_id, set()) - new_hashes:
                hash_and_block_id.discard((stale_hash, block_id))
                stale_entries = hash_to_rank_and_block_id.get(stale_hash)
                if stale_entries is not None:
                    stale_entries.discard((rank, block_id))
                    if not stale_entries:
                        hash_to_rank_and_block_id.pop(stale_hash, None)
            by_block[block_id] = new_hashes

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
        hash_to_rank_and_block_id = self._hash_to_rank_and_block_id.get(
            virtual_engine, {}
        )
        snapshots = self._snapshot_ranks.get(virtual_engine, {})
        remote: Dict[int, Set[Tuple[int, int, int]]] = {}
        for hash_content, _id in hash_contents:
            snapshot_ranks = [
                rank
                for rank in snapshots.get(hash_content, ())
                if rank not in self._unavailable_ranks and rank != exclude_rank
            ]
            if snapshot_ranks:
                rank = random.choice(snapshot_ranks)
                remote.setdefault(rank, set()).add((hash_content, hash_content, _id))
                continue
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
        self._unregister_snapshots(virtual_engine, rank, block_ids)
        entries = self._rank_to_hash_and_block_id.get(virtual_engine, {}).get(rank)
        hashes = self._hash_to_rank_and_block_id.get(virtual_engine, {})
        if not entries:
            return
        by_block = self._rank_to_block_hashes.get(virtual_engine, {}).get(rank, {})
        removed = {
            (content_hash, block_id)
            for block_id in set(block_ids)
            for content_hash in by_block.pop(block_id, ())
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
        for engine, owned in self._rank_snapshots.items():
            self._unregister_snapshots(engine, rank, list(owned.get(rank, ())))
            owned.pop(rank, None)
        for _, rank_to_hash_and_block_id in self._rank_to_hash_and_block_id.items():
            rank_to_hash_and_block_id.pop(rank, None)
        for rank_to_block_hashes in self._rank_to_block_hashes.values():
            rank_to_block_hashes.pop(rank, None)

        for _, hash_to_rank_and_block_id in self._hash_to_rank_and_block_id.items():
            for _, rank_and_block_id in hash_to_rank_and_block_id.items():
                to_delete = [(r, b) for r, b in rank_and_block_id if r == rank]
                if to_delete:
                    rank_and_block_id.difference_update(to_delete)

        self._unavailable_ranks.discard(rank)


# Keep the old class import and actor UID for existing deployments.
VLLMBlockTracker = BlockTracker
