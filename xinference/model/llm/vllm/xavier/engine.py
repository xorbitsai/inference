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
import logging
from typing import Any, Dict, Optional

from packaging import version
from vllm import AsyncEngineArgs
from vllm import __version__ as VLLM_VERSION
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.usage.usage_lib import UsageContext

from .transport import (
    XAVIER_CONNECTOR,
    XAVIER_CONNECTOR_MODULE,
    XAVIER_TRANSPORT_XAVIER,
    set_xavier_transport_backend,
)

logger = logging.getLogger(__name__)

XAVIER_EAGER_VLLM_VERSION = version.parse("0.21.0")


class XavierEngine:
    _xavier_config: Optional[Dict] = None

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        if isinstance(value, dict):
            return {str(k): XavierEngine._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [XavierEngine._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @classmethod
    def _patch_v1_engine_args(
        cls, engine_args: AsyncEngineArgs, xavier_config: Optional[Dict]
    ) -> None:
        from vllm.config import KVTransferConfig

        if (
            getattr(engine_args, "tensor_parallel_size", 1) != 1
            or getattr(engine_args, "pipeline_parallel_size", 1) != 1
        ):
            raise ValueError("Xavier V1 currently requires TP=1 and PP=1")
        if getattr(engine_args, "enable_lora", False):
            raise ValueError(
                "Xavier V1 currently supports text-only models without LoRA"
            )

        if xavier_config is None:
            xavier_config = {}
        xavier_config = dict(xavier_config)
        set_xavier_transport_backend(xavier_config, XAVIER_TRANSPORT_XAVIER)

        additional_config = dict(getattr(engine_args, "additional_config", {}) or {})
        # vLLM V1 includes additional_config in its compilation cache hash and
        # serializes it as JSON. Keep the connector's full config below, but
        # only expose a JSON-safe copy here.
        additional_config["xavier_config"] = cls._json_safe(xavier_config)
        engine_args.additional_config = additional_config

        role = xavier_config.get("role")
        if role == "prefill":
            kv_role = "kv_producer"
            kv_rank = 0
        elif role == "decode":
            kv_role = "kv_consumer"
            kv_rank = 1
        else:
            kv_role = "kv_both"
            kv_rank = xavier_config.get("rank", 0)

        extra_config = dict(xavier_config.get("kv_connector_extra_config") or {})
        extra_config["xavier_config"] = xavier_config

        if (
            version.parse(VLLM_VERSION) >= XAVIER_EAGER_VLLM_VERSION
            and xavier_config.get("enforce_eager", True)
            and not getattr(engine_args, "enforce_eager", False)
        ):
            # vLLM V1 may create XavierConnector twice during CUDA graph
            # setup. In CUDA-heavy processes this can trip glibc static TLS
            # allocation, so keep Xavier on the eager execution path.
            engine_args.enforce_eager = True
            logger.info(
                "Set enforce_eager=True for Xavier V1 on vLLM %s.",
                VLLM_VERSION,
            )

        engine_args.kv_transfer_config = KVTransferConfig(
            kv_connector=XAVIER_CONNECTOR,
            kv_connector_module_path=XAVIER_CONNECTOR_MODULE,
            engine_id=xavier_config.get("engine_id"),
            kv_role=kv_role,
            kv_rank=xavier_config.get("kv_rank", kv_rank),
            kv_parallel_size=xavier_config.get("kv_parallel_size", 2),
            kv_connector_extra_config=extra_config,
            kv_load_failure_policy=xavier_config.get(
                "kv_load_failure_policy", "recompute"
            ),
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        engine_config=None,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: Optional[Any] = None,
        xavier_config: Optional[Dict] = None,
    ) -> "AsyncLLMEngine":
        if version.parse(VLLM_VERSION) < version.parse("0.11.0"):
            from .legacy_engine import XavierEngine as LegacyXavierEngine

            return LegacyXavierEngine.from_engine_args(
                engine_args,
                engine_config=engine_config,
                start_engine_loop=start_engine_loop,
                usage_context=usage_context,
                stat_loggers=stat_loggers,
                xavier_config=xavier_config,
            )
        if version.parse(VLLM_VERSION) < version.parse("0.21.0"):
            raise RuntimeError("Xavier V1 requires vLLM >= 0.21.0")
        cls._xavier_config = xavier_config
        cls._patch_v1_engine_args(engine_args, xavier_config)
        logger.debug("Start Xavier V1 adapter for vLLM with config: %s", xavier_config)
        return AsyncLLMEngine.from_engine_args(
            engine_args,
            start_engine_loop,
            usage_context,
            stat_loggers,  # type: ignore[arg-type]
        )
