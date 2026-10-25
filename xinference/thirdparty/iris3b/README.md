# Iris-3B inference code

Source: https://github.com/speridlabs/iris-3b

Commit: `56b7926f4a619e11e101c5dedb86934d71fdbf53`

This copy contains the Python modules required for Iris-3B and Iris-3B Turbo
inference in Xinference. Internal imports were changed to relative imports.
The upstream Apache-2.0 license and copyright notices are retained in `LICENSE`
and `NOTICE`.

Training, distillation, datasets, downstream depth/restoration models, demos,
and checkpoint weights are not included. Training-only dependencies such as
Dion, wandb, and matplotlib are not required. Model weights and the Qwen3-VL
text encoder remain managed by Xinference's model cache and selected download hub.
