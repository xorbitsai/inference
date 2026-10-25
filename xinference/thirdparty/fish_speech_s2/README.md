# Fish Audio S2 inference runtime

Built with Fish Audio.

This directory contains the inference subset vendored from
[fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) at commit
`befe4001745417f8c42131739d862b8a6fdbd15a`. The upstream software is licensed
under the Fish Audio Research License. Research and non-commercial use are
permitted under that license; commercial use requires a separate license from
Fish Audio.

Only the tokenizer, conversation format, text-to-semantic runtime, ModifiedDAC
codec, inference engine, schemas, and supporting utilities needed by Xinference
are included. Model weights are not vendored and remain governed by their model
repository license.

Xinference changes upstream imports to package-qualified imports, trims the
training-only utility initializer, and resolves the bundled Hydra codec config
relative to this package. See `NOTICE` for the required attribution and change
notice.
