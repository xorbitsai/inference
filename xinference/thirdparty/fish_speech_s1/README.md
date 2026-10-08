# Fish Audio S1 inference runtime

This directory contains the inference subset vendored from
[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) at commit
`d3df50503b36314a964f66cac1af1e19e95bcfa3`. The upstream source code at that
revision is licensed under Apache-2.0.

Only the tokenizer, text-to-semantic runtime, ModifiedDAC codec, inference
engine, schemas, and supporting utilities needed by Xinference are included.
Model weights are not vendored. Fish Audio S1-mini weights are licensed under
CC-BY-NC-SA-4.0 and remain subject to the model repository's access terms.

Xinference changes upstream imports to package-qualified imports, trims the
training-only utility initializer, and resolves the bundled Hydra codec config
relative to this package. It also backports reference-ID validation and
torchaudio 2.9 backend detection from the newer runtime.
