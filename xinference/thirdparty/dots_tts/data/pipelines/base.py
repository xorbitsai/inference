from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator


class BaseSamplePipeline(ABC):
    """1:1 sample pipeline that preserves adapter resume metadata."""

    @staticmethod
    def _validate_input_sample(sample: dict) -> None:
        if "_adapter_state" not in sample:
            raise RuntimeError(
                "Source sample is missing required '_adapter_state' for resume."
            )

    @abstractmethod
    def process_sample(self, sample: dict) -> dict:
        """Transform one raw sample into one processed sample."""

    def __call__(self, samples: Iterable[dict]) -> Iterator[dict]:
        for raw_sample in samples:
            self._validate_input_sample(raw_sample)
            processed = self.process_sample(dict(raw_sample))
            if not isinstance(processed, dict):
                raise RuntimeError(
                    f"{self.__class__.__name__}.process_sample() must return a dict."
                )
            item = dict(raw_sample)
            item.update(processed)
            self._validate_input_sample(item)
            yield item
