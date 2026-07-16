# AI Agent Guidance

Guidance for AI coding agents working in this repository.

## Project Overview

Xinference is a Python model-serving project for language, embedding, rerank,
image, video, audio, and multimodal models. It exposes CLI commands, a Python
client, REST/OpenAI-compatible APIs, an xoscar-based distributed runtime, and a
React Web UI.

Primary package and entry points:

- `xinference/`: Python package.
- `xinference/deploy/cmdline.py`: CLI entry points for `xinference`,
  `xinference-local`, `xinference-supervisor`, and `xinference-worker`.
- `xinference/core/`: supervisor/worker runtime and actor orchestration.
- `xinference/model/`: model families, engines, built-in model specs, and model
  tests.
- `xinference/api/`: API server and OpenAI-compatible routes.
- `xinference/client/`: sync and async Python clients.
- `frontend/`: Next.js Web UI.
- `doc/source/`: Sphinx documentation.
- `.github/workflows/python.yaml`: main lint and test CI.

## Working Rules

- Prefer small, focused changes that match the current module's style.
- Preserve backward compatibility. If a breaking change is unavoidable, document
  the reason and add deprecation behavior where practical.
- Do not edit `xinference/thirdparty/` unless the task explicitly concerns
  vendored code.
- Avoid broad refactors while fixing a localized bug.
- Keep public API behavior, request/response schemas, model registration names,
  and CLI flags stable unless the task requires changing them.
- Add or update tests for behavior changes. For model-runtime changes, prefer
  tests close to the affected model family under `xinference/model/**/tests/`.
- Use type hints for new Python code when practical; this project encourages
  PEP 484 style annotations.
- Treat docs and examples as user-facing API. Keep command examples accurate.

## Environment Setup

Recommended local setup:

```bash
conda create --name xinf python=3.12 nodejs
conda activate xinf
pip install -e ".[dev]"
```

Notes:

- `setup.py develop/install/sdist` runs the Web UI build unless
  `NO_WEB_UI=1` is set.
- For Python 3.12 and newer, CI installs `setuptools<82`; use the same pin if
  packaging or editable installs fail.
- The project supports Python 3.10 through 3.13 in CI.
- Optional model engines have extras in `setup.cfg`, such as `transformers`,
  `vllm`, `mlx`, `embedding`, `rerank`, `image`, `video`, and `audio`.

## Formatting and Linting

Python formatting and checks are managed through pre-commit:

```bash
pip install pre-commit
pre-commit run --files <modified-files>
```

For a branch-wide check against upstream main:

```bash
pre-commit run --from-ref=upstream/main --to-ref=HEAD --all-files
```

Configured hooks include Black, end-of-file/trailing-whitespace checks, Flake8,
isort, mypy with missing imports ignored, and codespell. Configuration lives in
`.pre-commit-config.yaml`, `pyproject.toml`, and `setup.cfg`.

## Python Tests

Run focused tests first:

```bash
pytest -vv path/to/test_file.py
```

The broad CI-style non-GPU test command is approximately:

```bash
pytest --timeout=3000 -W ignore::PendingDeprecationWarning -vv \
  --cov-config=setup.cfg --cov-report=xml --cov=xinference \
  --ignore xinference/core/tests/test_continuous_batching.py \
  --ignore xinference/model/image/tests/test_stable_diffusion.py \
  --ignore xinference/model/image/tests/test_got_ocr2.py \
  --ignore xinference/model/audio/tests \
  --ignore xinference/model/embedding/tests/test_integrated_embedding.py \
  --ignore xinference/model/llm/transformers/tests/test_tensorizer.py \
  --ignore xinference/model/llm/tests/test_llm_model.py \
  --ignore xinference/model/llm/vllm \
  --ignore xinference/model/llm/sglang \
  --ignore xinference/client/tests/test_client.py \
  --ignore xinference/client/tests/test_async_client.py \
  --ignore xinference/model/llm/mlx \
  xinference
```

Use narrower commands for daily development. Many model tests require large
dependencies, GPU, Metal, network access, or model downloads.

## Frontend

The Web UI is under `frontend` and is a Next.js app (React, TypeScript,
Tailwind CSS), built as a static export and served by the Python backend from
`xinference/ui/web/dist`.

Common commands:

```bash
cd frontend
npm ci
npm run dev
npm run build
npx eslint .
```

Use `npm run format` only when you intentionally want Prettier writes across
the frontend tree.

## Documentation

Documentation source is in `doc/source`.

Common docs dependencies are included in the `doc` extra:

```bash
pip install -e ".[doc]"
cd doc
make html
```

When changing CLI behavior, API behavior, deployment behavior, or model support,
update the relevant documentation pages in `doc/source`.

## Model and Runtime Conventions

- Keep model-family logic inside the relevant `xinference/model/<family>/`
  package.
- Keep built-in model metadata changes close to existing specs and tests.
- Be careful with lazy imports and optional dependencies. Import heavyweight
  model libraries only where needed so unrelated installs still work.
- Preserve platform guards for Linux-only, CUDA-only, and macOS Metal/MLX paths.
- For distributed runtime changes, consider both local mode and
  supervisor/worker mode.
- For OpenAI-compatible behavior, verify request/response fields and streaming
  behavior against existing API and client tests.

## CI Expectations

The main CI workflow:

- Runs `pre-commit run --all-files`.
- Runs UI `npm ci`, `npx eslint .`, and Prettier check.
- Tests Python 3.10 through 3.13 across Linux, macOS, and Windows.
- Has special GPU and macOS Metal jobs for model-specific paths.

Before marking a change done, run the smallest meaningful validation command
that covers the behavior you changed, and mention any broader checks that were
not run because of environment cost or missing hardware.

## Git and Review Hygiene

- Keep commits scoped to the requested change.
- Do not rewrite or revert user changes in an existing worktree unless asked.
- If the active checkout is busy or on an unrelated branch, use a separate
  worktree and a semantic branch name such as `fix/...`, `feat/...`, or
  `docs/...`.
- In PR reviews, inspect current GitHub review threads before adding duplicate
  comments.
