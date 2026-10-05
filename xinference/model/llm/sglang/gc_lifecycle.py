# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Keep initialized engine libraries out of request-time cycle scans."""

import gc


class InitializationGCFreeze:
    def __init__(self) -> None:
        self._owned = False

    def start(self) -> None:
        if self._owned or gc.get_freeze_count():
            return
        gc.collect()
        gc.freeze()
        self._owned = True

    def close(self) -> None:
        if self._owned:
            gc.unfreeze()
            self._owned = False
