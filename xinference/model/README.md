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
