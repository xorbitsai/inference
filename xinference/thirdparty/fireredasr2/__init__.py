# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)
# Modified by Xinference to expose the vendored ASR module without process-wide setup.

from .asr import FireRedAsr2, FireRedAsr2Config

__all__ = ["FireRedAsr2", "FireRedAsr2Config"]
