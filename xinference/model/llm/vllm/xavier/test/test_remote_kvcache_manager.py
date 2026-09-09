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

from unittest.mock import AsyncMock

import pytest

from ..block_tracker import VLLMBlockTracker
from ..transfer import TransferActor
from ..xavier_remote_kvcache_manager import XavierRemoteKVCacheManager


@pytest.fixture
def mock_block_tracker():
    # Mock the VLLMBlockTracker class to simulate its async behavior
    mock_tracker = AsyncMock(spec=VLLMBlockTracker)
    # ActorRef exposes RPC coroutines even for synchronous actor methods.
    mock_tracker.register_blocks = AsyncMock()
    mock_tracker.unregister_block = AsyncMock()
    mock_tracker.query_blocks = AsyncMock()
    return mock_tracker


@pytest.fixture
def mock_transfer_actor():
    # Mock the TransferActor class to simulate its async behavior
    mock_transfer = AsyncMock(spec=TransferActor)
    return mock_transfer


@pytest.fixture
def cache_manager(mock_block_tracker, mock_transfer_actor):
    # Create a XavierRemoteKVCacheManager instance and inject mocked dependencies
    cache_manager = XavierRemoteKVCacheManager()
    cache_manager._block_tracker_ref = mock_block_tracker
    cache_manager._transfer_ref = mock_transfer_actor
    return cache_manager


@pytest.mark.asyncio
async def test_register_blocks(cache_manager, mock_block_tracker):
    # Arrange mock data for the test
    engine_metadata = {"virtual_engine": 0, "rank": 0}
    cache_metadatas = [
        {"content_hash": "hash1", "block_id": 1},
        {"content_hash": "hash2", "block_id": 2},
    ]

    # Call the method
    await cache_manager.register_blocks(engine_metadata, cache_metadatas)


@pytest.mark.asyncio
async def test_unregister_blocks(cache_manager, mock_block_tracker):
    # Arrange mock data for the test
    engine_metadata = {"virtual_engine": 0, "rank": 0}
    cache_metadatas = [{"rank": 0, "block_id": 1}, {"rank": 0, "block_id": 2}]

    # Call the method
    await cache_manager.unregister_blocks(engine_metadata, cache_metadatas)


@pytest.mark.asyncio
async def test_query_blocks(cache_manager, mock_block_tracker):
    # Arrange mock data for the test
    engine_metadata = {"virtual_engine": 0, "rank": 0}
    cache_metadatas = [
        {"content_hash": "hash1", "block_id": 1},
        {"content_hash": "hash2", "block_id": 2},
    ]

    # Simulate the return value of query_blocks on block_tracker
    mock_block_tracker.query_blocks = AsyncMock()
    mock_block_tracker.query_blocks.return_value = {0: [(1, 4), (2, 5)]}

    # Call the method
    result = await cache_manager.query_blocks(engine_metadata, cache_metadatas)

    assert result == {0: [(1, 4), (2, 5)]}
    mock_block_tracker.query_blocks.assert_awaited_once_with(
        0, [("hash1", 1), ("hash2", 2)], exclude_rank=0
    )


@pytest.mark.asyncio
async def test_read_blocks(cache_manager, mock_transfer_actor):
    # Arrange mock data for the test
    engine_metadata = {"virtual_engine": 0, "rank": 0}
    cache_metadatas = [{"from_rank": 0, "remote_block_metadata": [(1, 4, 5)]}]

    # Simulate the return value of read_blocks, make sure cache manager can
    # return same data
    mock_transfer_actor.read_blocks = AsyncMock()
    mock_transfer_actor.read_blocks.return_value = ((), {}, {})

    # Call the method
    result = await cache_manager.read_blocks(engine_metadata, cache_metadatas)

    assert result == ((), {}, {})
