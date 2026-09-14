# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-efficient entry point for LingBot-World-V2's official runner."""

import runpy
import sys
from pathlib import Path


def _patch_t5_checkpoint_loading() -> None:
    """Avoid a full in-memory checkpoint copy while constructing UMT5-XXL."""
    import torch
    from wan.modules import t5 as t5_module

    def init(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=None,
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        if device is None:
            device = torch.cuda.current_device()
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        model = (
            t5_module.umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=dtype,
                device=device,
            )
            .eval()
            .requires_grad_(False)
        )
        t5_module.logging.info("loading %s", checkpoint_path)
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        except (RuntimeError, TypeError, ValueError):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        del state_dict

        self.model = shard_fn(model, sync_module_states=False) if shard_fn else model
        if shard_fn is None:
            self.model.to(self.device)
        self.tokenizer = t5_module.HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=text_len,
            clean="whitespace",
        )

    t5_module.T5EncoderModel.__init__ = init


def main() -> None:
    _patch_t5_checkpoint_loading()
    sys.argv[0] = "generate.py"
    runpy.run_path(str(Path.cwd() / "generate.py"), run_name="__main__")


if __name__ == "__main__":
    main()
