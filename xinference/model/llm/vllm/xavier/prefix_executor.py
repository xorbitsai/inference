# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Resolve Xavier's default matching unit from the loaded vLLM model."""

import logging
from typing import Optional

from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec, MambaSpec

logger = logging.getLogger(__name__)


def _default_matching_unit(specs: list[dict[str, KVCacheSpec]]) -> Optional[int]:
    # Ordinary Xavier replicas have one worker and use aligned attention and
    # recurrent pages. Identical page sizes also rule out allocator resizing.
    if len(specs) != 1:
        return None
    layers = list(specs[0].values())
    if (
        not any(isinstance(spec, FullAttentionSpec) for spec in layers)
        or not any(isinstance(spec, MambaSpec) for spec in layers)
        or any(
            not isinstance(spec, (FullAttentionSpec, MambaSpec))
            or not spec.prefix_cacheable
            or getattr(spec, "non_causal", False)
            or (isinstance(spec, MambaSpec) and spec.mamba_cache_mode != "align")
            for spec in layers
        )
        or len({(spec.block_size, spec.page_size_bytes) for spec in layers}) != 1
    ):
        return None
    block_size = layers[0].block_size
    alignments = {
        alignment
        for spec in layers
        if isinstance(alignment := spec.tokens_per_state, int) and alignment > 1
    }
    # Pick a divisor close to 32 without creating a hash for every few tokens.
    for unit in range(min(32, block_size // 2), 15, -1):
        if block_size % unit == 0 and all(unit % size == 0 for size in alignments):
            return unit
    return None


class _XavierHybridExecutor(Executor):
    def get_kv_cache_specs(self) -> list[dict[str, KVCacheSpec]]:
        specs = super().get_kv_cache_specs()
        # The model and attention backend have resolved their physical pages.
        # vLLM will use this value when it initializes workers and the scheduler.
        if self.cache_config.prefix_match_unit is None:
            unit = _default_matching_unit(specs)
            if unit is not None:
                self.cache_config.prefix_match_unit = unit
                logger.info("Xavier selected a %d-token prefix matching unit.", unit)
        return specs


class XavierHybridUniProcExecutor(_XavierHybridExecutor, UniProcExecutor):
    pass


class XavierHybridMultiprocExecutor(_XavierHybridExecutor, MultiprocExecutor):
    pass
