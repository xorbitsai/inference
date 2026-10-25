# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Internal entry point for a Worker-launched replacement process."""

import asyncio
import json
import logging.config
import multiprocessing
import sys
from pathlib import Path

from .worker import _start_worker
from .worker_host import register_worker_host


def main():
    path = Path(sys.argv[1])
    config = json.loads(path.read_text())
    path.unlink()
    multiprocessing.set_start_method("spawn", force=True)
    if config.get("logging_conf"):
        logging.config.dictConfig(config["logging_conf"])

    async def run():
        # Register before starting any ownership transfer. Failure leaves the
        # original Worker serving, rather than allowing PID 1 to lose its child.
        await register_worker_host()
        await _start_worker(**config)

    asyncio.run(run())


if __name__ == "__main__":
    main()
