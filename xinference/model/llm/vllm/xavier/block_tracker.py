from typing import Dict, List, Set, Tuple

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

    def set_blocks(
        self, virtual_engine: int, block_infos: List[Tuple[int, int]], address: str
    ):
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
            if hash_content in hash_to_address_and_block_id:
                # TODO
                address, block_id = next(
                    iter(hash_to_address_and_block_id[hash_content])
                )
                if address not in remote:
                    remote[address] = {
                        (hash_content, block_id, _id),
                    }
                else:
                    remote[address].add((hash_content, block_id, _id))
        return remote
