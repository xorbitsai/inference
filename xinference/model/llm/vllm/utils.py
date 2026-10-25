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
import functools
import ipaddress
import logging
import os
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

# vLLM has moved the engine-dead exception between releases.  Import the
# concrete exception where possible instead of falling back to RuntimeError
# whenever vLLM is installed: a RuntimeError raised by request processing is
# not, by itself, evidence that the EngineCore has died.
_vllm_engine_dead_errors = []
try:
    from vllm.engine.async_llm_engine import AsyncEngineDeadError
except Exception:
    # vLLM 0.19.0+ removed AsyncEngineDeadError from this module.
    AsyncEngineDeadError = None  # type: ignore[assignment,misc]
else:
    _vllm_engine_dead_errors.append(AsyncEngineDeadError)

try:
    from vllm.v1.engine.exceptions import EngineDeadError
except Exception:
    EngineDeadError = None  # type: ignore[assignment,misc]
else:
    _vllm_engine_dead_errors.append(EngineDeadError)

VLLM_ENGINE_DEAD_ERRORS = tuple(_vllm_engine_dead_errors) or (RuntimeError,)

_F = TypeVar("_F", bound=Callable[..., Awaitable[Any]])


def _stop_after_engine_death(model: Any) -> None:
    logger.exception("vLLM EngineCore is dead; terminating the model process")
    try:
        model.stop()
    except Exception:
        # Ignore errors while stopping a broken engine.
        logger.debug(
            "Failed to stop the vLLM model after EngineCore death", exc_info=True
        )
    # Let Xinference's process supervisor recover the model.  In particular,
    # do not let the vLLM exception cross the xoscar Worker/Supervisor boundary:
    # the Supervisor may not have vLLM installed and would replace the useful
    # EngineDeadError with a misleading ModuleNotFoundError during unpickling.
    os._exit(1)


async def _guard_async_iterator(model: Any, iterable: Any) -> AsyncIterator[Any]:
    completed = False
    iterator = None
    try:
        # AsyncIterable objects do not have to implement __anext__ themselves;
        # close the concrete iterator returned by __aiter__ on cancellation.
        iterator = aiter(iterable)
        async for item in iterator:
            yield item
        completed = True
    except VLLM_ENGINE_DEAD_ERRORS:
        _stop_after_engine_death(model)
    finally:
        # Closing the wrapper (for example after a client disconnects) must
        # also close the wrapped async iterator so that vLLM can release the
        # request and run its cleanup handlers.  Avoid an unnecessary aclose
        # call after normal exhaustion.
        if not completed and iterator is not None:
            aclose = getattr(iterator, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except VLLM_ENGINE_DEAD_ERRORS:
                    _stop_after_engine_death(model)


def vllm_check(fn: _F) -> _F:
    @functools.wraps(fn)
    async def _async_wrapper(self, *args, **kwargs):
        try:
            result = await fn(self, *args, **kwargs)
            # async_generate/async_chat return an async generator.  Exceptions
            # from `async for` happen after the decorated coroutine has already
            # returned, so the outer try/except cannot see them unless the
            # iterator itself is wrapped.
            if hasattr(result, "__aiter__"):
                return _guard_async_iterator(self, result)
            return result
        except VLLM_ENGINE_DEAD_ERRORS:
            _stop_after_engine_death(self)
        return cast(Any, None)

    return cast(_F, _async_wrapper)


def get_distributed_init_method(ip: str, port: int) -> str:
    return get_tcp_uri(ip, port)


def get_tcp_uri(ip: str, port: int) -> str:
    if is_valid_ipv6_address(ip):
        return f"tcp://[{ip}]:{port}"  # noqa E231
    else:
        return f"tcp://{ip}:{port}"  # noqa E231


def is_valid_ipv6_address(address: str) -> bool:
    try:
        ipaddress.IPv6Address(address)
        return True
    except ValueError:
        return False
