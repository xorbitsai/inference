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

from ..block_tracker import VLLMBlockTracker


def _query_rank_block(tracker, hash_content):
    """Return the (rank, block_id) set a hash currently resolves to."""
    remote = tracker.query_blocks(0, [(hash_content, 0)])
    return {(rank, blk) for rank, details in remote.items() for _, blk, _ in details}


class TestBlockTrackerReuse:
    def test_reused_block_id_evicts_stale_hash(self):
        tracker = VLLMBlockTracker()
        # Rank 1 stages content H1 in physical block 5.
        tracker.register_blocks(0, [(1001, 5)], rank=1)
        assert _query_rank_block(tracker, 1001) == {(1, 5)}

        # vLLM reuses block 5 for new content H2 on the same rank.
        tracker.register_blocks(0, [(2002, 5)], rank=1)
        # The stale H1 -> (1, 5) entry must be gone, or a later query on H1
        # would load H2's KV out of the reused block.
        assert _query_rank_block(tracker, 1001) == set()
        assert _query_rank_block(tracker, 2002) == {(1, 5)}

    def test_reuse_does_not_touch_other_ranks(self):
        tracker = VLLMBlockTracker()
        tracker.register_blocks(0, [(1001, 5)], rank=1)
        tracker.register_blocks(0, [(1001, 5)], rank=2)
        # Reuse of block 5 on rank 1 must not evict rank 2's identical mapping.
        tracker.register_blocks(0, [(2002, 5)], rank=1)
        assert _query_rank_block(tracker, 1001) == {(2, 5)}
        assert _query_rank_block(tracker, 2002) == {(1, 5)}

    def test_reregister_same_hash_block_is_idempotent(self):
        tracker = VLLMBlockTracker()
        tracker.register_blocks(0, [(1001, 5)], rank=1)
        tracker.register_blocks(0, [(1001, 5)], rank=1)
        assert _query_rank_block(tracker, 1001) == {(1, 5)}
        # Internal bookkeeping stays at a single entry for (rank, block_id).
        assert tracker._rank_to_hash_and_block_id[0][1] == {(1001, 5)}

    def test_tracker_size_bounded_by_block_id(self):
        tracker = VLLMBlockTracker()
        # Same block id reused for 100 distinct contents keeps exactly one
        # tracker entry, instead of accumulating 100 dangling ones.
        for content in range(100):
            tracker.register_blocks(0, [(9000 + content, 7)], rank=1)
        assert tracker._rank_to_hash_and_block_id[0][1] == {(9099, 7)}
        assert _query_rank_block(tracker, 9099) == {(1, 7)}
        assert _query_rank_block(tracker, 9000) == set()

    def test_batch_register_mixes_new_and_reused(self):
        tracker = VLLMBlockTracker()
        tracker.register_blocks(0, [(1001, 5), (1002, 6)], rank=1)
        # block 5 reused for H3, block 6 keeps H2, block 7 is new.
        tracker.register_blocks(0, [(3003, 5), (1002, 6), (4004, 7)], rank=1)
        assert _query_rank_block(tracker, 1001) == set()
        assert _query_rank_block(tracker, 3003) == {(1, 5)}
        assert _query_rank_block(tracker, 1002) == {(1, 6)}
        assert _query_rank_block(tracker, 4004) == {(1, 7)}

    def test_unregister_block_still_works(self):
        tracker = VLLMBlockTracker()
        tracker.register_blocks(0, [(1001, 5)], rank=1)
        tracker.unregister_block(0, rank=1, block_id=5)
        assert _query_rank_block(tracker, 1001) == set()
        assert tracker._rank_to_hash_and_block_id[0][1] == set()
