# Breeze TTS 2 inference runtime

This directory contains the inference subset vendored from
[breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts) at
commit `ca632ce6c4d05f7985da4eab29b1a5d445b43f7b`. The upstream source code is
licensed under Apache-2.0.

Only `breeze_infer` helpers, model runtime files, and the fast warmup profile
required by Xinference are included. Model weights are not vendored. They
remain governed by the BreezeBlue Research and Non-Commercial License.

Xinference changes the vendored runtime's `models` and `breeze_infer` imports
to package-relative imports so they cannot collide with unrelated top-level
Python packages.
