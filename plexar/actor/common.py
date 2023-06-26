# Copyright 2022-2023 XProbe Inc.
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
from typing import Any, Generic, Iterator, TypeVar

import xoscar as xo


class IteratorActor(xo.Actor):
    def __init__(self, it: Iterator[Any]):
        super().__init__()
        self._iter = it

    def __iter__(self):
        return IteratorWrapper(self.address, self.uid)

    def next(self):
        try:
            return self._iter.__next__()
        except StopIteration:
            raise Exception("StopIteration")


T = TypeVar("T")


class IteratorWrapper(Generic[T]):
    def __init__(self, iter_actor_addr: str, iter_actor_uid: str):
        self._iter_actor_addr = iter_actor_addr
        self._iter_actor_uid = iter_actor_uid
        self._iter_actor_ref = None

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        if self._iter_actor_ref is None:
            self._iter_actor_ref = await xo.actor_ref(
                address=self._iter_actor_addr, uid=self._iter_actor_uid
            )

        try:
            assert self._iter_actor_ref is not None
            return await self._iter_actor_ref.next()
        except Exception as e:
            if str(e) == "StopIteration":
                await xo.destroy_actor(self._iter_actor_ref)
                raise StopAsyncIteration
            else:
                raise
