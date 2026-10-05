# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Keep initialized engine libraries out of request-time cycle scans."""

import gc


class InitializationGCFreeze:
    def __init__(self) -> None:
        self._owned = False
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        # Some runtimes start with a small frozen graph. That does not cover
        # the libraries loaded for this model. Extend it once, preserving the
        # external owner's right to unfreeze it when its own lifetime ends.
        self._owned = gc.get_freeze_count() == 0
        gc.collect()
        gc.freeze()
        self._started = True

    def close(self) -> None:
        if self._owned:
            gc.unfreeze()
            self._owned = False
        self._started = False
