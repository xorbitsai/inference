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

## LLM model metadata

An LLM `model_specs` entry can optionally include `model_metadata`. It stores
architecture dimensions and optional checkpoint size, not a fixed memory requirement:
context length and concurrency still affect the estimate. Store it per specification, since sizes
within the same family can have different architectures.

Maintainers and Models Hub jobs can extract this object from an already obtained
`config.json` using:

```sh
python -m xinference.model.llm.collect_model_metadata /path/to/config.json
```

The command only reads that local file and prints JSON; it does not fetch weights,
download configuration, or execute model-provided code. Copy the output into the
spec's `model_metadata` field. Required dimensions are `vocab_size`,
`num_attention_heads`, `hidden_size`, `intermediate_size`, and `num_hidden_layers`;
`num_key_value_heads` and `head_dim` are optional. Preserve an explicit `head_dim`
when present: it need not equal hidden size divided by attention heads.
`config_source` records the configuration URL, pinned to a commit when the hub
provides one. `config_sha256` records the collected bytes (the metadata header
for GGUF). Missing dimensions fail extraction rather than being guessed. Nested
text configurations are retained with an `architecture_type` category such as
`multimodal`, `moe`, `mla` or `hybrid_attention`. This describes the model, not
whether a particular consumer supports it. The memory estimator owns its support
checks; the catalog does not store an estimator-specific rejection reason.

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
  --cache /tmp/model-config-snapshots --weights
```

This command contacts the registered hubs, reads bounded configuration files
(or GGUF metadata headers, never tensor payloads), and writes a fresh staging
`models/` directory plus `coverage.json`. Review both before applying the
generated model files. It does not run during startup or recommendation. Missing,
gated, malformed or inaccessible configurations are reported, not replaced by
guesses. Cached successful snapshots allow retrying failures; use a new cache
directory to refresh moving revisions. The external Models Hub publisher must
invoke this step explicitly and preserve its fields when syncing the catalog.

For GGUF sources without vision dimensions in their text configuration, the
collector also reads the first registered projector's metadata at the same
revision. Its dimensions, source URL and header hash are stored in `vision`.
Unreadable projector metadata leaves image workloads unverified.

`--weights` additionally reads bounded Safetensors indexes or headers and GGUF
file-size responses at the resolved configuration revision. `weights_bytes`
includes all resident experts and vision weights, including the largest registered
GGUF projector and every model shard. It never uses the activated MoE parameter
count as the resident weight size. `weights_source` records the selected revision,
pinned when the hub provides a commit;
`weights_sha256` hashes the collected index/header, or the filename-size manifest
for GGUF, not the tensor payload. Missing sizes retain nominal weight estimates
when supported; quantized hybrid, MoE and multimodal models require a measured
checkpoint size. GGUF collection falls back to a streamed one-byte range response
when HEAD omits size. Config-only refreshes and failed weight reads preserve
previous sizes only when the config hash and source remain unchanged and
`weights_source` matches that quantization's checkpoint file at the resolved
revision. If configuration collection fails, existing metadata is retained with
coverage status `stale_retained`; it has not been refreshed or verified against
the current remote configuration. Sizes from another quantization are never
retained. Removed quantizations are pruned, and shared metadata is emitted only
when every registered quantization has an identical entry.

Template repositories with separate quantization checkpoints store
`model_metadata_by_quantization` in each source. Runtime flattening selects
only the matching quantization's metadata; an absent entry stays unknown and
never borrows another quantization's dimensions. Common MLX 2/3/4/5/6/8-bit and
FP16/BF16 names are accepted. Mixed quantizations can use a collected checkpoint
size; otherwise unsupported quantizations remain unverified. Quantizations sharing
one GGUF config still keep separate metadata when their weight sizes differ.

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
full model estimator accepts `concurrency` and counts `context_length` text tokens
plus `vision_tokens` per sequence. Sliding-attention layers use the conservative
full-cache bound. `vision_tokens_per_image` bounds attention within each image;
zero treats all image tokens as one image. No attention scores cross image boundaries.
`layer_types` or `full_attention_interval` records the layout; Gated DeltaNet's five
`linear_*` dimensions describe its FP32 recurrent matrix and convolution history.
Missing or unfamiliar layouts remain unverified, including MLA and other state
spaces. Optional `vision` dimensions bound encoder workspace. Additional retained
cache is supplied separately as `prefix_cache_bytes`.

The optional `engine` selects conservative workspace bounds. MLX and llama.cpp
use a 2048-token prefill bound; Transformers retains the full requested context
and eager-attention bound. CUDA profiles for vLLM and SGLang use a shared 16,384-token batch profile,
fused-attention buffers and logits only for sampled rows. Their prefill workspace
is not multiplied by sequence count after reaching that batch cap. Other devices
retain an eager-attention bound. These account
for active-expert FFNs and vision encoding, not calibrated kernel peaks. Weight
storage is counted once; cached states grow with concurrency. Without `engine`, ordinary
models retain the legacy activation approximation.

The footprint does not model cache block padding or all runtime allocations.
LLM recommendations compare it with 80% of the worker's current free memory,
defaulting to one sequence, the registered model's maximum context, FP16 KV cache and no image or retained
prefix cache. The CUDA single-GPU vLLM default (90% of total memory) is checked
separately against free memory when normalized total counters are available.
SGLang derives its cache budget from available initialization memory. Automatic
placement still uses the smallest free-memory value among eligible devices,
because the allocator chooses by process load rather than free-memory counters.
Workload integers are bounded to avoid arithmetic overflow. An explicit
`context_length` overrides the default; otherwise encoded image tokens share
the model's maximum context with text tokens. Missing model context metadata
stays unverified, including older worker estimates made at a fixed context.
Recommended vLLM
`max_model_len` and SGLang `context_length` equal text plus vision tokens per
sequence. llama.cpp receives `n_parallel=concurrency` and
`n_ctx=(context_length + vision_tokens) * concurrency`, since it divides context
across slots. The Web UI applies these as advanced launch parameters. Other
workload fields affect estimates rather than request limits. Engine preallocation
is not modeled beyond the separate vLLM default reservation check. Older workers
that cannot accept workload options are retried without them and their estimates
are discarded, with a version warning. The CLI prints recurrent state and retained prefix cache
separately and reports unsupported estimates without a traceback.
Single-device candidates with a usable estimate are ranked
largest-first; known oversized candidates are excluded. Missing estimates or
multi-GPU requests retain compatibility-only selection. Other model types do not
use memory ranking. A successful estimate is not a guarantee that a model fits.
