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


def main():
    path = Path(sys.argv[1])
    config = json.loads(path.read_text())
    path.unlink()
    multiprocessing.set_start_method("spawn", force=True)
    if config.get("logging_conf"):
        logging.config.dictConfig(config["logging_conf"])
    asyncio.run(_start_worker(**config))


if __name__ == "__main__":
    main()
