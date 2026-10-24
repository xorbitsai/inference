# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Run one Supervisor generation in its selected Python environment."""

import asyncio
import json
import logging.config
import sys
from pathlib import Path

import xoscar as xo

from ..core.supervisor import SupervisorActor
from ..core.system_settings_store import SystemSettings, apply_system_settings


def main():
    path = Path(sys.argv[1])
    configuration = json.loads(path.read_text())
    path.unlink()
    baseline = SystemSettings.from_dict(configuration["settings"])
    apply_system_settings(baseline)
    logging_conf = configuration.get("logging_conf")
    from .utils import update_all_formatter_addresses, update_logging_config_addresses

    update_all_formatter_addresses("supervisor", configuration["service_address"])
    update_logging_config_addresses(
        logging_conf, "supervisor", configuration["service_address"]
    )
    if logging_conf:
        logging.config.dictConfig(logging_conf)

    async def run():
        pool = await xo.create_actor_pool(
            address=configuration["address"],
            n_process=0,
            logging_conf={"dict": logging_conf} if logging_conf else None,
        )
        try:
            await xo.create_actor(
                SupervisorActor,
                service_address=configuration["service_address"],
                standby=True,
                address=configuration["address"],
                uid=SupervisorActor.default_uid(),
            )
            await pool.join()
        finally:
            await pool.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
