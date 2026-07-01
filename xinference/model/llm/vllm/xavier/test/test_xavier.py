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

import pytest
import xoscar as xo

from ..block_tracker import VLLMBlockTracker


class ExtendedBlockTracker(VLLMBlockTracker):
    def get_hash_to_rank_and_block_id(self):
        return self._hash_to_rank_and_block_id

    def get_rank_to_hash_and_block_id(self):
        return self._rank_to_hash_and_block_id


@pytest.fixture
async def actor_pool_context():
    pool = await xo.create_actor_pool("127.0.0.1", n_process=2)
    async with pool:
        yield pool


@pytest.mark.asyncio
async def test_block_tracker(actor_pool_context):
    actor_pool = actor_pool_context
    addr = actor_pool.external_address
    tracker_ref: xo.ActorRefType[ExtendedBlockTracker] = await xo.create_actor(  # type: ignore
        ExtendedBlockTracker,
        address=addr,
        uid=VLLMBlockTracker.default_uid(),
    )

    virtual_engine = 0
    rank = 0
    block_infos = [(123, 0), (456, 1), (789, 2)]

    # register blocks
    await tracker_ref.register_blocks(virtual_engine, block_infos, rank)

    # query blocks
    res = await tracker_ref.query_blocks(virtual_engine, [(123, 4), (789, 5)])
    assert len(res) == 1
    assert rank in res
    assert len(res[rank]) == 2
    assert {x[0] for x in res[rank]} == {123, 789}
    assert {x[1] for x in res[rank]} == {0, 2}
    assert {x[2] for x in res[rank]} == {4, 5}

    # query with extra info
    res = await tracker_ref.query_blocks(virtual_engine, [(123, 4), (789, 5), (110, 6)])
    assert len(res) == 1
    assert rank in res
    assert len(res[rank]) == 2
    assert {x[0] for x in res[rank]} == {123, 789}
    assert {x[1] for x in res[rank]} == {0, 2}
    assert {x[2] for x in res[rank]} == {4, 5}

    # unregister block
    await tracker_ref.unregister_block(virtual_engine, rank, 1)
    res = await tracker_ref.query_blocks(virtual_engine, [(123, 4), (456, 7)])
    assert len(res) == 1
    assert rank in res
    assert len(res[rank]) == 1
    assert {x[0] for x in res[rank]} == {123}
    assert {x[1] for x in res[rank]} == {
        0,
    }
    assert {x[2] for x in res[rank]} == {
        4,
    }
    # nothing happens
    await tracker_ref.unregister_block(virtual_engine, rank, 3)
    res = await tracker_ref.query_blocks(virtual_engine, [(123, 4), (456, 7)])
    assert len(res) == 1
    assert rank in res
    assert len(res[rank]) == 1
    assert {x[0] for x in res[rank]} == {123}
    assert {x[1] for x in res[rank]} == {
        0,
    }
    assert {x[2] for x in res[rank]} == {
        4,
    }
    # query returns empty
    res = await tracker_ref.query_blocks(virtual_engine, [(456, 8)])
    assert res == {}

    # check internal data
    hash_to_rank_and_block_id = await tracker_ref.get_hash_to_rank_and_block_id()
    assert virtual_engine in hash_to_rank_and_block_id
    assert hash_to_rank_and_block_id[virtual_engine] == {
        123: {
            (rank, 0),
        },
        456: set(),
        789: {(rank, 2)},
    }

    rank_to_hash_and_block_id = await tracker_ref.get_rank_to_hash_and_block_id()
    assert virtual_engine in rank_to_hash_and_block_id
    assert rank_to_hash_and_block_id[virtual_engine] == {rank: {(123, 0), (789, 2)}}

    # register blocks
    new_rank = 1
    block_infos = [(111, 7), (222, 8), (333, 9), (123, 10)]
    await tracker_ref.register_blocks(virtual_engine, block_infos, new_rank)

    # test unregister rank
    await tracker_ref.unregister_rank(0)
    res = await tracker_ref.query_blocks(virtual_engine, [(789, 5)])
    assert len(res) == 0
    res = await tracker_ref.query_blocks(virtual_engine, [(123, 6)])
    assert len(res) == 1
    assert new_rank in res

    # check internal data
    rank_to_hash_and_block_id = await tracker_ref.get_rank_to_hash_and_block_id()
    assert rank in rank_to_hash_and_block_id[virtual_engine]
    assert new_rank in rank_to_hash_and_block_id[virtual_engine]

    # test register rank
    await tracker_ref.register_rank(0)
    rank_to_hash_and_block_id = await tracker_ref.get_rank_to_hash_and_block_id()
    assert rank not in rank_to_hash_and_block_id[virtual_engine]
    assert new_rank in rank_to_hash_and_block_id[virtual_engine]
