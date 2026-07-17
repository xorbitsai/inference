# xinference-pypiserver image

Builds `xprobe/xinference-pypiserver`: a [pypiserver](https://github.com/pypiserver/pypiserver)
preloaded with the index-compatible wheels the xinference runtime may install
into per-model virtual environments, for offline / air-gapped deployments. It
is the default package source of the `offline` compose profile in
`xinference/deploy/docker` (see the "Offline / Air-gapped Deployment" section
of the Docker Compose docs).

The mirror's GPU stack targets CUDA 13.0 and Python 3.12 (matching the
`xprobe/xinference` runtime image). CUDA 12.8/12.9 remain supported by online
runtime installs, but their stacks are not included in this prebuilt mirror.

Git and non-wheel direct references cannot be reproduced faithfully through a
simple index. The generator records them, the selfcheck reports them as
unsupported, and `XINFERENCE_VIRTUAL_ENV_OFFLINE_INSTALL=1` makes the runtime
reject them before attempting egress. Preinstall those exact source revisions
in a custom runtime image when the corresponding model is needed air-gapped.

## How the build works

1. **`generate_package_lists.py`** loads `ENGINE_VIRTUALENV_PACKAGES` and the
   marker-filtering logic straight from the xinference sources (no install),
   so the lists cannot drift from what the runtime installs. It emits one
   requirement set per engine, the per-model concrete pins, direct wheel URLs
   and git sources, filtered for the target platform/CUDA.
2. **`download_packages.py`** locks the torch family to the runtime image's
   version, locks each engine set with `uv pip compile`, and fetches the
   fully-pinned locks with `pip download --no-deps` — one coherent resolution
   per engine, so unpinned specs cannot fan out into many versions. Model pins
   are fetched constrained to the shared locks (with a recorded unconstrained
   fallback on conflicts), git sources and sdist-only downloads are built into
   wheels.
3. **`selfcheck.py`** (a dedicated build stage) re-resolves every
   index-compatible engine set, pin and direct wheel against the baked mirror
   ALONE (`--disable-fallback`). Any gap fails the image build; recorded git
   and non-wheel direct references are reported as unsupported.

The generation manifest and download report are baked into the image under
`/data/manifest/` for debugging (`report.json` lists total size, packages with
multiple versions, and any sdists that could not be built into wheels).

## Building locally

From the repository root:

```bash
docker build -f xinference/deploy/docker/pypiserver/Dockerfile.pypiserver \
  --build-arg PLATFORM=amd64 -t xinference-pypiserver:amd64 .
```

Build args: `PLATFORM` (amd64/arm64), `TORCH_VERSION` (must match the torch of
the `xprobe/xinference` runtime image), `CUDA_SUFFIX` (default cu130),
`PYTHON_VERSION` (default 3.12).

CI (`.github/workflows/build-pypiserver-image.yaml`) builds both architectures
on release tags and merges them into multi-arch `:<tag>` / `:latest` manifests.

## Verifying offline behavior end to end

Use the air-gap compose override, which moves xinference and the mirror onto
an `internal` Docker network with no external routing:

```bash
cd xinference/deploy/docker
cp offline.env.example offline.env
# uncomment the [global] block in pip.conf
docker compose --profile offline \
  -f docker-compose.yml -f docker-compose.airgap.yml up -d
```

Then launch a model with a virtual environment and confirm in the worker log
that the install settings carry the private index, e.g.:

```
Installing packages [...] with settings(index_url=http://xinference-pypiserver:8080/simple, ...)
```

A quick way to confirm traffic actually flows through the mirror is comparing
the mirror container's TX byte counter before/after the launch:

```bash
docker compose exec xinference-pypiserver \
  sh -c 'cat /sys/class/net/eth0/statistics/tx_bytes'
```
