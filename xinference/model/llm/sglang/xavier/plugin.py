# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Register Xavier with SGLang's supported plugin hook mechanism."""

from enum import Enum


def _kv_class(original, backend, class_type):
    if backend.value != "xavier":
        return original(backend, class_type)
    from sglang.srt.disaggregation.base.conn import KVArgs

    from .gpu import XavierKVManager, XavierKVReceiver, XavierKVSender

    return {
        "kvargs": KVArgs,
        "manager": XavierKVManager,
        "sender": XavierKVSender,
        "receiver": XavierKVReceiver,
        "bootstrap_server": XavierBootstrapServer,
    }[class_type.value]


class XavierBootstrapServer:
    """The Xinference directory supplies bootstrap metadata through actor RPC."""

    def __init__(self, host, port):
        pass


def register():
    # Importing the entry point is safe without SGLang; heavy imports stay here.
    from sglang.srt.disaggregation.utils import TransferBackend
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.server_args import add_disagg_transfer_backend_choices

    if "xavier" not in {member.value for member in TransferBackend}:
        backend = Enum(
            "TransferBackend",
            {
                **{member.name: member.value for member in TransferBackend},
                "XAVIER": "xavier",
            },
        )
        HookRegistry.register(
            "sglang.srt.disaggregation.utils.TransferBackend", backend, HookType.REPLACE
        )
    add_disagg_transfer_backend_choices("xavier")
    HookRegistry.register(
        "sglang.srt.disaggregation.utils.get_kv_class", _kv_class, HookType.AROUND
    )
