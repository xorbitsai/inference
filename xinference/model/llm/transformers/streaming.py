# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.

from threading import Event, Thread
from typing import Any, Callable, Dict, Iterator


def generate_stream(generate: Callable, kwargs: Dict[str, Any]) -> Iterator:
    """Own a Transformers generation thread through cancellation and completion."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    kwargs = dict(kwargs)
    streamer = kwargs["streamer"]
    cancelled = Event()
    errors = []

    class Cancelled(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return cancelled.is_set()

    kwargs["stopping_criteria"] = StoppingCriteriaList(
        [*(kwargs.get("stopping_criteria") or []), Cancelled()]
    )

    def run():
        try:
            generate(**kwargs)
        except Exception as exc:
            errors.append(exc)
            streamer.end()

    thread = Thread(target=run, daemon=True)
    thread.start()

    def iterate():
        try:
            yield None  # Establish ownership before returning an unread iterator.
            yield from streamer
            thread.join()  # A streamer may end before generate returns or raises.
            if errors:
                raise errors[0]
        finally:
            cancelled.set()
            thread.join()

    result = iterate()
    next(result)
    return result
