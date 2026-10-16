"""Minimal string-keyed registries for swappable components."""

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class Registry:
    """Name -> constructor map with a decorator-based registration API."""

    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, Callable] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(fn: T) -> T:
            if name in self._entries:
                raise KeyError(f"duplicate {self.kind} '{name}'")
            self._entries[name] = fn
            return fn

        return deco

    def get(self, name: str) -> Callable:
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(sorted(self._entries))
            raise KeyError(f"unknown {self.kind} '{name}' (known: {known})") from None

    def build(self, name: str, *args, **kwargs):
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._entries)


BLOCKS = Registry("block")
TEXT_ENCODERS = Registry("text_encoder")
OPTIMIZERS = Registry("optimizer")
TIMESTEP_SAMPLERS = Registry("timestep_sampler")
DATASETS = Registry("dataset")
