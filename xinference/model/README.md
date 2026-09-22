# Built-in model catalogs

Each model type keeps its built-in metadata in `models/<model_name>.json`.
A file contains a JSON array, including every engine/format/source variant of
that model. World families with nested named variants use `model_family` as the
filename. Model record fields are unchanged.

Model files are discovered automatically in case-insensitive filename order.
Adding or deleting a model only requires adding or deleting its file; there is
no shared index or ordering field. Record order within each file is preserved,
including engine, format, and source preferences for that model.

Use `xinference._model_catalog.load_model_catalog` to read catalogs. Runtime
normalization still happens in the corresponding model-family loader. Do not
read only the first record of a model file or reorder its records.

## Models Hub integration

The Hub download API and locally downloaded `<type>_models.json` files remain
aggregate arrays. Custom registration formats and timestamp-based merge rules
are unchanged. Splitting the repository does not require users to redownload
their model catalogs or contact the network at startup.

The Hub's repository PR writer must target the new model files. For a
full-catalog sync, convert the aggregate response into a fresh staging directory:

```sh
python xinference/_model_catalog.py split llm_models.json /tmp/new-llm-models
python xinference/_model_catalog.py validate /tmp/new-llm-models
```

Review the diff and replace the corresponding `xinference/model/llm/models`
directory with that staging directory, including deletions. Do not split a
partial update and use it to replace the full catalog. For a single-model
update, preserve the other model files.

The conversion rejects existing destination directories to prevent accidental
overwrites. These maintenance commands use only the Python standard library.
The Hub's publisher can generate the unchanged aggregate format for older
clients rather than maintain a second authored copy:

```sh
python xinference/_model_catalog.py export xinference/model/llm/models /tmp/llm_models.json
```

Coordinate deployment of the Hub PR writer with this source-layout change;
its implementation lives outside this repository. PRs that still modify the
removed aggregate source paths must be converted before merging. The trusted
documentation workflow accepts only catalog JSON data, validates filenames and
model names, and refuses symlinks; it never runs PR-supplied Python code.

After changing a catalog, run `python gen_docs.py` from `doc/source`. Packaging
includes the model files recursively. Aggregate exports use the same stable
filename order, rather than preserving the old cross-model array order.

## Offline LLM memory metadata

An LLM `model_specs` entry can optionally include `memory_estimation`. It stores
architecture dimensions, not a fixed memory requirement: context length and
quantization still affect the estimate. Store it per specification, since sizes
within the same family can have different architectures.

Maintainers and Models Hub jobs can extract this object from an already obtained
`config.json` using:

```sh
python -m xinference.model.llm.collect_memory_metadata /path/to/config.json
```

The command only reads that local file and prints JSON; it does not fetch weights,
download configuration, or execute model-provided code. Copy the output into the
spec's `memory_estimation` field. Required dimensions are `vocab_size`,
`num_attention_heads`, `hidden_size`, `intermediate_size`, and `num_hidden_layers`;
`num_key_value_heads` and `head_dim` are optional. Preserve an explicit `head_dim`
when present: it need not equal hidden size divided by attention heads.
`config_source` records the configuration URL, pinned to a commit when the hub
provides one. `config_sha256` records the collected bytes (the metadata header
for GGUF). Missing dimensions fail extraction rather than being guessed. Nested
text configurations are retained for maintenance, with `unsupported_reason`
preventing the dense estimator from treating multimodal, MoE, MLA or hybrid
architectures as supported.

Use the configuration belonging to that exact checkpoint/revision. Re-extract
metadata when updating a checkpoint; do not copy it to another size or assume all
conversions share the same architecture. Hub updates must preserve this field
when exporting aggregate catalogs. Existing catalogs without it remain valid.
The external Hub job still needs to opt into extraction; this command does not
change its deployment or automatically populate the built-in catalog.

To refresh all registered LLM sources, use the explicit maintenance command:

```sh
python -m xinference.model.llm.collect_memory_catalog \
  xinference/model/llm/models /tmp/new-memory-catalog \
  --cache /tmp/model-config-snapshots
```

This command contacts the registered hubs, reads bounded configuration files
(or GGUF metadata headers, never tensor payloads), and writes a fresh staging
`models/` directory plus `coverage.json`. Review both before applying the
generated model files. It does not run during startup or recommendation. Missing,
gated, malformed or inaccessible configurations are reported, not replaced by
guesses. Cached successful snapshots allow retrying failures; use a new cache
directory to refresh moving revisions. The external Models Hub publisher must
invoke this step explicitly and preserve its fields when syncing the catalog.

Template repositories with separate quantization checkpoints store
`memory_estimation_by_quantization` in each source. Runtime flattening selects
only the matching quantization's metadata; an absent entry stays unknown and
never borrows another quantization's dimensions. Common MLX 2/3/4/5/6/8-bit and
FP16/BF16 names are accepted. Unsupported mixed quantizations remain unverified.

`estimate_llm_gpu_memory(..., allow_download=False)` only uses the matching spec's
metadata. It returns `None` when metadata is absent, including when no model name
was supplied. The default remains backward compatible: use metadata first, then
the existing config download or unnamed-model heuristic. Catalog quantization
`"none"` is accepted as unquantized, like Python `None`.

For ordinary full-attention MHA/GQA/MQA models, KV cache payload is estimated as
`2 * cached_tokens * layers * KV_heads * head_dim * bytes_per_element`.
The factor of two includes both K and V, also for GGUF. An explicit `head_dim`
takes precedence over `hidden_size / num_attention_heads`. Missing KV head count
means MHA. Weight quantization does not determine KV cache precision; the
`kv_cache_dtype` argument specifies 8, 16 or 32 bits independently.

`estimate_kv_cache_memory(info, num_tokens, kv_cache_dtype)` returns MiB before
rounding, across the whole model rather than per GPU. Pass the sum of cached
tokens across requests for concurrent sequences without prefix sharing. The
full model estimator uses one sequence's `context_length` for this component.
Activation estimates use attention heads, not KV heads, but still use the old
approximation rather than modeling FlashAttention or engine-specific workspaces.

These estimates exclude cache block padding, quantization scales, prefix sharing
and engine preallocation. MLA, sliding-window, hybrid and multimodal caches need
their own accounting; MoE weight estimation also remains limited. This does not
inspect free device memory itself. LLM recommendations separately compare it
with 80% of the worker's current free memory, assuming one sequence, 2048 tokens
and FP16 KV cache. Single-device candidates with a usable estimate are ranked
largest-first; known oversized candidates are excluded. Missing estimates or
multi-GPU requests retain compatibility-only selection. Other model types do not
use memory ranking. A successful estimate is not a guarantee that a model fits.
