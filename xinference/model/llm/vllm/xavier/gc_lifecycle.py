# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Exclude initialized libraries from snapshot allocation cycle scans."""

import gc


class InitializationGCFreeze:
    def __init__(self) -> None:
        self._owned = False
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._owned = gc.get_freeze_count() == 0
        gc.collect()
        gc.freeze()
        self._started = True

    def close(self) -> None:
        if self._owned:
            gc.unfreeze()
            self._owned = False
        self._started = False
