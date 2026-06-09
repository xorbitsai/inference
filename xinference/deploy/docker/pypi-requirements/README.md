# PyPI Server Requirements

This directory contains pip requirements files organized by architecture and
wheel category. They are consumed by [Dockerfile.pypi](../Dockerfile.pypi) to
build a custom PyPI server image that hosts all third-party dependencies
xinference needs.

## Directory Structure

```
pypi-requirements/
├── common/                 # Same for both amd64 and arm64
│   ├── base.txt            # Core framework deps (API, logging, OTEL, etc.)
│   ├── ml.txt              # ML ecosystem deps (transformers, diffusers, etc.)
│   └── models.txt          # Model-specific deps (ChatTTS, kokoro, funasr, etc.)
├── amd64/                  # x86_64 Linux only
│   ├── compiled.txt        # Native code deps — not CUDA-specific (uvloop, onnxruntime)
│   ├── torch.txt           # PyTorch ecosystem (torch, torchvision, torchaudio, torchcodec)
│   ├── torch-cpu.txt       # PyTorch CPU-only variant (no torchcodec)
│   ├── cuda-ext.txt        # CUDA extensions (bitsandbytes, sgl-kernel, triton, etc.)
│   └── vllm.txt            # vLLM ecosystem (vllm, flashinfer, xllamacpp)
└── arm64/                  # aarch64 / Grace Hopper / Jetson
    ├── compiled.txt        # Native code deps — same package names as amd64
    ├── torch.txt           # PyTorch ecosystem (no torchcodec on arm64)
    ├── torch-cpu.txt       # PyTorch CPU-only variant
    └── cuda-ext.txt        # CUDA extensions (includes triton, no vllm/flashinfer)
```

## File Purposes

### common/

| File      | Type            | Description |
|-----------|-----------------|-------------|
| `base.txt`  | Pure Python     | Framework infrastructure: xoscar, gradio, fastapi, pydantic, OpenTelemetry, gguf, etc. (38 packages) |
| `ml.txt`    | Pure Python     | Common ML libraries: transformers, accelerate, peft, diffusers, sentence-transformers, numpy, etc. (47 packages) |
| `models.txt`| Pure Python*    | Model family dependencies: ChatTTS, kokoro, funasr, spacy, librosa, soundfile, etc. (45 packages) |

\* Some packages in models.txt have compiled sub-dependencies (e.g., mecab-python3, fugashi)
but are fetched via the same pip download path.

### Why three separate common files?

- **base.txt** changes when the framework adds/removes infrastructure libraries.
- **ml.txt** changes when ML library versions need pinning (e.g., transformers).
- **models.txt** changes when a new model family is added or a model's deps change.

Keeping them separate makes dependency auditing and PR review easier. They are
downloaded in a single `pip download` invocation, so there is no build-time
penalty for the separation.

### amd64/ and arm64/

| File          | Type                  | Description |
|---------------|-----------------------|-------------|
| `compiled.txt`  | Native (arch-specific) | Platform-compiled packages not tied to a CUDA version: `uvloop`, `onnxruntime==1.16.0`. Same package names for both architectures — `pip download` resolves the correct platform wheel at build time. |
| `torch.txt`     | CUDA-specific          | PyTorch ecosystem packages downloaded once per CUDA version with `--index-url https://download.pytorch.org/whl/${CUDA_VER}`. The `--index-url` (not the file content) determines which CUDA wheel is fetched. |
| `torch-cpu.txt` | CPU-specific           | PyTorch CPU-only variant downloaded from `https://download.pytorch.org/whl/cpu`. Separate file because it excludes `torchcodec` (no CPU build available). |
| `cuda-ext.txt`  | CUDA-specific          | CUDA-dependent extensions: `bitsandbytes`, `sgl-kernel`, `triton`, `torchao`, `xgrammar`, `onnxruntime-gpu`. Downloaded once per CUDA version for error isolation. |
| `vllm.txt`      | CUDA-specific, amd64   | vLLM ecosystem (`vllm`, `flashinfer`, `xllamacpp`). **amd64 only** — vLLM and flashinfer lack prebuilt arm64 wheels. |

### Per-CUDA File Merge

`torch-*.txt`, `cuda-ext-*.txt`, and `vllm-*.txt` files were merged into single
files (`torch.txt`, `cuda-ext.txt`, `vllm.txt`) because the package lists were
identical across CUDA versions (cu124/cu126/cu128). The CUDA version
differentiation comes from the `--index-url` parameter passed to `pip download`
at build time, not from the file content.

## Build Process

### Step 1: Downloader Stage (`python:3.12-slim`)

The Dockerfile downloads all wheels in a temporary stage before copying them to
the final image:

```
1. common/base.txt + common/ml.txt + common/models.txt   → /data/packages/
2. ${TARGET_ARCH}/compiled.txt                           → /data/packages/
3. ${TARGET_ARCH}/torch.txt      × {cu124, cu126, cu128}  → /data/packages/
   (each iteration uses a different --index-url)
4. ${TARGET_ARCH}/torch-cpu.txt                          → /data/packages/
5. ${TARGET_ARCH}/cuda-ext.txt   × {cu124, cu126, cu128}  → /data/packages/
6. ${TARGET_ARCH}/vllm.txt       × {cu124, cu126, cu128}  → /data/packages/
   (skipped entirely for arm64)
7. download-wheels.sh             External sources        → /data/packages/
   (flash-attn from GitHub releases, xllamacpp from custom index)
8. xinference build artifacts (dist/*.whl)               → /data/packages/
```

### Step 2: Final Stage (`pypiserver/pypiserver:latest`)

All wheels are copied from the downloader stage into a stock pypiserver image:

```dockerfile
COPY --from=downloader /data/packages/ /data/packages/
```

At runtime, pypiserver serves `/data/packages/` as a PEP 503 simple repository.

### Error Handling

- Each CUDA version loop collects failures individually — one failing CUDA
  version does not block others (e.g., cu124 torch may not be published yet
  while cu126 is available).
- External downloads (flash-attn, xllamacpp) emit warnings on failure but do
  not abort the build — those packages may not have wheels for every
  architecture/CUDA combination.
- The build **will** fail if no xinference build artifacts are found in `dist/`.

### Build Commands

```bash
# Build amd64 image (all CUDA versions included)
docker build -f xinference/deploy/docker/Dockerfile.pypi \
  --build-arg TARGET_ARCH=amd64 \
  -t pypiserver:amd64 .

# Build arm64 image (all CUDA versions included, no vLLM)
docker build -f xinference/deploy/docker/Dockerfile.pypi \
  --build-arg TARGET_ARCH=arm64 \
  -t pypiserver:arm64 .
```

### Usage

Once the pypiserver is running, users install packages by pointing `pip` at it:

```bash
# Install a specific CUDA version of torch
pip install torch==2.5.0+cu128 --index-url https://<pypiserver>/simple

# The pypiserver serves everything — all CUDA versions, all architectures
pip install vllm --index-url https://<pypiserver>/simple
```

## External Wheel Sources

Some packages are not available from PyPI or PyTorch indexes and are downloaded
separately via [scripts/download-wheels.sh](../scripts/download-wheels.sh):

| Package       | Source |
|---------------|--------|
| flash-attn    | GitHub releases ([mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels)) |
| xllamacpp     | Custom index ([xorbitsai.github.io/xllamacpp/whl/](https://xorbitsai.github.io/xllamacpp/whl/)) |
| triton        | Standard PyPI (downloaded by the script for consistency) |
| flashinfer    | Standard PyPI (amd64 only) |

## Adding New Dependencies

1. **Framework dependency** → add to `common/base.txt`
2. **ML library** → add to `common/ml.txt`
3. **Model-specific dependency** → add to `common/models.txt`
4. **Compiled, not CUDA-specific** → add to both `amd64/compiled.txt` and `arm64/compiled.txt`
5. **PyTorch ecosystem** → add to `torch.txt` and/or `torch-cpu.txt`
6. **CUDA extension** → add to `cuda-ext.txt` (both architectures)
7. **vLLM ecosystem** → add to `amd64/vllm.txt` only
8. **From an external index or GitHub** → add to `download-wheels.sh`

Always verify that prebuilt wheels exist for both `amd64` and `arm64` before
adding a compiled/CUDA package. If a package supports only one architecture,
use an arch-gated section in the Dockerfile (like vllm).

## Updating to a New CUDA Version

When a new CUDA minor version is released (e.g., cu130), follow this process
step by step. Each step has a checklist of what to verify and where to make
changes.

### Overview

Adding a new CUDA version touches these layers:

| Layer | What Changes | Affected Files |
|-------|-------------|----------------|
| PyTorch ecosystem | New index URL, verify wheel availability | None (reuses `torch.txt`) |
| CUDA extensions | Verify package compatibility | `cuda-ext.txt` (both archs) |
| vLLM ecosystem | Verify wheel availability (amd64) | `amd64/vllm.txt` |
| External sources | Hardcoded URLs and version tags | `scripts/download-wheels.sh` |
| Dockerfile | CUDA_VERSIONS default value | `Dockerfile.pypi` |
| CI workflow | Build matrix (if arch-specific tags) | `.github/workflows/build-pypiserver-image.yaml` |

The good news: because `torch.txt`, `cuda-ext.txt`, and `vllm.txt` are reused
for every CUDA version, you do **not** need to create new requirements files.
The only code changes are in the Dockerfile default value and the external
wheel download script.

### Step-by-Step

Assume the new CUDA version is `cu130` (CUDA 13.0).

#### Step 1: Verify PyTorch Wheel Availability

Check that PyTorch publishes wheels for the new CUDA version:

```bash
# Verify the index exists
curl -I https://download.pytorch.org/whl/cu130/

# Check specific packages (same list as in torch.txt)
# torch, torchvision, torchaudio, torchcodec
```

For each package, verify both architectures have wheels:

| Package | amd64 check | arm64 check |
|---------|------------|------------|
| torch | Search for `cp312-cp312-linux_x86_64` | Search for `cp312-cp312-linux_aarch64` |
| torchvision | Same pattern | Same pattern |
| torchaudio | Same pattern | Same pattern |
| torchcodec | Same pattern | May not exist for arm64 |

If `torchcodec` is missing for arm64, update `arm64/torch.txt` to exclude it
(arm64 currently already excludes it for this reason).

**No file changes needed for `torch.txt`** — the package list stays the same.

#### Step 2: Check CUDA Extension Package Compatibility

Packages in `cuda-ext.txt` come from standard PyPI (no special index). Each
must have a wheel compatible with the new CUDA toolkit:

| Package | Check | Notes |
|---------|-------|-------|
| `bitsandbytes` | PyPI release notes | Tied to CUDA runtime, may lag |
| `sgl-kernel` | GitHub releases | Version-bump often needed |
| `cuda-python` | PyPI / NVIDIA index | NVIDIA-synchronized, usually available early |
| `torchao` | PyPI | Requires compatible torch first |
| `xgrammar` | PyPI | Check version compatibility |
| `triton` | PyPI | Generally CUDA-version-agnostic |
| `onnxruntime-gpu` | PyPI | Microsoft-synchronized, often lags |

```bash
# Quick check: try downloading one
pip download --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu130 \
  sgl-kernel -d /tmp/test/
```

If a package requires a minimum version bump for the new CUDA version, update
its constraint in **both** `amd64/cuda-ext.txt` and `arm64/cuda-ext.txt`.

Common scenario: `sgl-kernel` often needs a new version when a new CUDA version
is released. Example:

```diff
- sgl-kernel>=0.0.3.post3
+ sgl-kernel>=0.3.20   # first release with cu130 support
```

#### Step 3: Check vLLM Ecosystem (amd64 Only)

vLLM publishes wheels indexed by CUDA version. Verify:

```bash
# Check if vllm has a cu130-compatible wheel on PyPI
pip download --no-cache-dir vllm -d /tmp/test/
# If vllm itself supports cu130, flashinfer usually follows
```

**No file changes needed for `vllm.txt`** — the package list stays the same.
If vllm/flashinfer don't support the new CUDA version yet, the per-CUDA error
isolation in the Dockerfile will emit a warning and continue.

#### Step 4: Update External Wheel Sources

This is the most manual step. `scripts/download-wheels.sh` contains hardcoded
wheel URLs and version strings.

**4a. flash-attn**

Locate the `download_flash_attn()` function. The `wheels` array lists specific
`.whl` files by version and CUDA tag. You need to find or build new wheels for
the new CUDA version:

```bash
# Current pattern (example):
# "v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-linux_x86_64.whl"
```

Steps:
1. Check [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)
   for a release matching the new CUDA + torch versions.
2. If a release exists, add the wheel entry for each architecture:

```bash
case "${ARCH}" in
    amd64)
        wheels=(
            # ... existing entries ...
            "v0.9.5/flash_attn-2.9.0+cu130torch2.11-cp312-cp312-linux_x86_64.whl"
        )
        ;;
    arm64)
        wheels=(
            # ... existing entries ...
            "v0.9.5/flash_attn-2.9.0+cu130torch2.11-cp312-cp312-linux_aarch64.whl"
        )
        ;;
esac
```

3. If no prebuilt wheel exists, this is acceptable — the download fails softly
   with a warning, and the rest of the build continues.

**4b. xllamacpp**

The download URL is currently hardcoded:

```bash
local index_url="https://xorbitsai.github.io/xllamacpp/whl/cu128"
```

Check if the custom index has a directory for the new CUDA version:

```bash
curl -I https://xorbitsai.github.io/xllamacpp/whl/cu130/
```

If available, update the URL:

```diff
-    local index_url="https://xorbitsai.github.io/xllamacpp/whl/cu128"
+    local index_url="https://xorbitsai.github.io/xllamacpp/whl/cu130"
```

If the index only has one CUDA version at a time, consider making it a
parameter or iterating over a list (similar to how the Dockerfile handles
torch downloads).

#### Step 5: Update Dockerfile Default

In `Dockerfile.pypi`, update the `CUDA_VERSIONS` build argument to include the
new version:

```diff
- ARG CUDA_VERSIONS="cu124 cu126 cu128"
+ ARG CUDA_VERSIONS="cu124 cu126 cu128 cu130"
```

This is the **only required Dockerfile change**. Users can also override it at
build time without changing the Dockerfile:

```bash
docker build -f Dockerfile.pypi \
  --build-arg CUDA_VERSIONS="cu124 cu126 cu128 cu130" \
  --build-arg TARGET_ARCH=amd64 \
  -t pypiserver:amd64 .
```

#### Step 6: Build and Test

Build both architectures and verify every CUDA version loop succeeds:

```bash
# Test amd64 build
docker build -f xinference/deploy/docker/Dockerfile.pypi \
  --build-arg TARGET_ARCH=amd64 \
  -t pypiserver:amd64-cu130-test . 2>&1 | tee build-amd64.log

# Test arm64 build (on an arm64 host, or via QEMU/docker buildx)
docker build -f xinference/deploy/docker/Dockerfile.pypi \
  --build-arg TARGET_ARCH=arm64 \
  -t pypiserver:arm64-cu130-test . 2>&1 | tee build-arm64.log
```

Check the build logs for these warning patterns:

| Warning Pattern | Meaning | Action |
|-----------------|---------|--------|
| `WARNING: failed to download: torch-cu130` | PyTorch cu130 wheels not available | Wait for upstream release |
| `WARNING: failed to download: cuda-ext-cu130` | A cuda-ext package lacks cu130 support | Check individual package, bump version constraint |
| `WARNING: failed to download: vllm-cu130` | vllm not yet available for cu130 | Expected; vllm often lags |
| `WARNING: failed to download https://github.com/...flash_attn...` | No prebuilt flash-attn wheel for cu130 | Find/build one, or accept as known gap |

Run the container and verify it serves all expected wheels:

```bash
docker run -d -p 8080:8080 pypiserver:amd64-cu130-test
curl http://localhost:8080/simple/ | grep "torch-"
```

#### Step 7: Drop Deprecated CUDA Versions (Optional)

Over time, old CUDA versions accumulate build time and image size. When a
CUDA version is no longer widely used, remove it from `CUDA_VERSIONS`:

```diff
- ARG CUDA_VERSIONS="cu118 cu121 cu124 cu126 cu128 cu130"
+ ARG CUDA_VERSIONS="cu124 cu126 cu128 cu130"
```

Before dropping, verify:
- No xinference CI job tests against the deprecated CUDA version.
- The upstream PyTorch index still serves the old CUDA version (it usually
  does for years, but PyTorch eventually drops old versions).
- No critical downstream user is pinned to the old CUDA version.

### Summary Checklist

```
New CUDA version: cu____

□ 1. PyTorch: https://download.pytorch.org/whl/cu____/ accessible
□ 1. PyTorch: torch, torchvision, torchaudio, torchcodec have amd64 wheels
□ 1. PyTorch: torch, torchvision, torchaudio have arm64 wheels
□ 2. cuda-ext: All 7 packages have wheels for both architectures
□ 2. cuda-ext: Version constraints in cuda-ext.txt still valid
□ 3. vllm: vllm has amd64 wheel (or accept delay)
□ 3. vllm: flashinfer has amd64 wheel (or accept delay)
□ 4. External: flash-attn prebuilt wheels found/added to download-wheels.sh
□ 4. External: xllamacpp custom index checked/updated in download-wheels.sh
□ 5. Dockerfile: CUDA_VERSIONS arg updated
□ 6. Build: amd64 image builds without unexpected errors
□ 6. Build: arm64 image builds without unexpected errors
□ 6. Test: Container serves all CUDA version wheels
```
