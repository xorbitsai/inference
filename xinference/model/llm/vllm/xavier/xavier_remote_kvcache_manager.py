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
from logging import getLogger
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import xoscar as xo

from .block_tracker import VLLMBlockTracker
from .remote_kvcache_manager import RemoteKVCacheManager
from .transfer import TransferActor

logger = getLogger(__name__)


class XavierRemoteKVCacheManager(RemoteKVCacheManager):
    def __init__(self):
        super().__init__()

        self._transfer_ref: Optional[xo.ActorRefType["TransferActor"]] = None
        self._block_tracker_ref: Optional[xo.ActorRefType["VLLMBlockTracker"]] = None

    async def setup(
        self,
        xavier_config: Dict[str, Any],
        transfer_metadata: Optional[Dict[str, Any]] = None,
        block_tracker_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Lazy setup actor reference with transfer actor and tracker actor.
        """
        from .transfer import TransferActor

        # Get transfer actor reference.
        if self._transfer_ref is None:
            transfer_address = xavier_config.get("rank_address")
            rank = xavier_config.get("rank")
            self._transfer_ref = await xo.actor_ref(
                address=transfer_address, uid=f"{TransferActor.default_uid()}-{rank}"
            )

        # Setup with transfer actor metadata.
        if transfer_metadata is not None and self._transfer_ref is not None:
            cache_engine = transfer_metadata.get("cache_engine")
            scheduler = transfer_metadata.get("scheduler")
            num_buffer = transfer_metadata.get("num_buffer")
            buffer_shape = transfer_metadata.get("buffer_shape")
            buffer_dtype = transfer_metadata.get("buffer_dtype")
            buffer_device = transfer_metadata.get("buffer_device")
            pin_memory = transfer_metadata.get("pin_memory")

            await self._transfer_ref.setup(
                cache_engine,
                scheduler,
                num_buffer=num_buffer,
                buffer_shape=buffer_shape,
                buffer_dtype=buffer_dtype,
                buffer_device=buffer_device,
                pin_memory=pin_memory,
            )

        # Get block tracker actor reference.
        if self._block_tracker_ref is None:
            block_tracker_address = xavier_config.get("block_tracker_address")
            block_tracker_uid = xavier_config.get("block_tracker_uid")
            self._block_tracker_ref = await xo.actor_ref(
                address=block_tracker_address, uid=block_tracker_uid
            )

    async def register_blocks(
        self,
        engine_metadata: Dict[str, Any],
        cache_metadatas: List[Dict[str, Any]],
    ):
        """
        Used to register metadata in the cache manager.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        virtual_engine = engine_metadata.get("virtual_engine")
        rank = engine_metadata.get("rank")
        executed_blocks_details = [
            (cache_metadata["content_hash"], cache_metadata["block_id"])
            for cache_metadata in cache_metadatas
        ]
        logger.debug(f"Register blocks: {executed_blocks_details}")

        assert self._block_tracker_ref is not None
        await self._block_tracker_ref.register_blocks(
            virtual_engine,
            executed_blocks_details,
            rank,
        )

    async def write_blocks(
        self,
        engine_metadata: Dict[str, Any],
        cache_metadata: List[Dict[str, Any]],
        cache_data: List[torch.Tensor],
    ):
        """
        Used to write cache data to the storage.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        cache_data: a list of kvcache data, espically for decoder llm each layer.
        """
        # In P2P xavier, we do not need to write cache to anywhere.
        pass

    async def query_blocks(
        self,
        engine_metadata: Dict[str, Any],
        cache_metadatas: List[Dict[str, Any]],
    ) -> Dict[int, Set[Tuple[int, int, int]]]:
        """
        Used to query cache metadata from remote storage.

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.

        return:
        remote: a dict of remote cache metadata, .
        """
        virtual_engine = engine_metadata.get("virtual_engine")
        current_rank = engine_metadata.get("rank")
        executed_blocks_details = [
            (metadata["content_hash"], metadata["block_id"])
            for metadata in cache_metadatas
        ]

        assert self._block_tracker_ref is not None
        res = await self._block_tracker_ref.query_blocks(
            virtual_engine,
            executed_blocks_details,
            exclude_rank=current_rank,
        )
        return res

    async def read_blocks(
        self,
        engine_metadata: Dict[str, Any],
        cache_metadata: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, List[int], int]:
        """
        Used to read cache metadata from remote storage, these data will be read at the buffer in self._buffer

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.

        return:
        1. A full buffer reference.
        2. a dict of block id to swap in index.
        3. A full buffer reference's metadata.
        """
        # cache_metadata only contains one item in xaiver.
        metadata = cache_metadata[0]
        from_rank = metadata["from_rank"]
        remote_block_metadata = metadata["remote_block_metadata"]
        src_to_dst = (
            dict(remote_block_metadata)
            if isinstance(remote_block_metadata, dict)
            else {x[1]: x[2] for x in remote_block_metadata}
        )

        assert self._transfer_ref is not None
        res = await self._transfer_ref.read_blocks(from_rank, src_to_dst)
        return res

    async def free_blocks(self, buffer_metadata: Dict[str, Any]):
        """
        Used to free buffer metadata from current storage

        buffer_metadata: a dict of buffer metadata, maybe contains cpu_buf_index and so on.
        """
        cpu_buf_index_dict = buffer_metadata.get("cpu_buf_index_dict")
        assert self._transfer_ref is not None
        await self._transfer_ref.free_buffer_index(cpu_buf_index_dict)

    async def unregister_blocks(
        self,
        engine_metadata: Dict[str, Any],
        cache_metadatas: List[Dict[str, Any]],
    ):
        """
        Used to remove metadata from remote storage

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        virtual_engine = engine_metadata.get("virtual_engine")

        for cache_metadata in cache_metadatas:
            rank = cache_metadata.get("rank")
            block_id = cache_metadata.get("block_id")
            assert self._block_tracker_ref is not None
            await self._block_tracker_ref.unregister_block(
                virtual_engine,
                rank,
                block_id,
            )

    async def remove_blocks(
        self,
        engine_metadata: Dict[str, Any],
        cache_metadatas: List[Dict[str, Any]],
    ):
        """
        Used to remove cache metadata from remote storage

        engine_metadata: virtual engine for llm backend, used to choose engine by
        cache_metadata: key value for this kvcache metadata, maybe contains hash_content, prefix promopt and so on.
        """
        # In P2P xavier, we do not need to remove cache from anywhere.
        pass
