# dots.tts.edit inference runtime

This directory contains the inference subset of
[studio-dots-ai/dots.tts](https://github.com/studio-dots-ai/dots.tts) at commit
`4cf7d6b13f42066f97784e7d7cf2a9ed5f956997` (package version `0.3.1`).
The upstream Apache-2.0 license is included in `LICENSE`.

The bundled files are the Python import closure of `DotsTtsEditRuntime`,
including the standard TTS runtime, edit instruction parser, model and codec
implementations, text processing, and inference profiling helpers. The shared
loss declarations imported by the model remain in `training/losses.py`.
Training entry points, dataset loaders, CLIs, the double-streaming runtime,
playground applications, and model weights are not included.

Xinference changes the upstream `dots_tts` imports to package-relative imports.
This keeps the bundled runtime independent of an installed `dots.tts` package
and prevents collisions with top-level packages. No inference behavior is
otherwise changed. Runtime dependencies are declared in the built-in model
specification; installing or launching this model does not fetch upstream
Python source from GitHub.
