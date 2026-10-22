.. _speculative_decoding:

Speculative decoding
====================
Speculative decoding proposes several tokens and lets the target model verify
them together. The available methods and launch options depend on the engine.
Use a compatible drafter and measure latency and throughput for your workload.

Paired MTP drafter (Gemma 4)
----------------------------

Some models ship a small paired drafter checkpoint that predicts several tokens
ahead, which the target model then verifies in one pass. Output is unchanged,
decoding gets faster. Gemma 4 calls this multi-token prediction (MTP) and
publishes a ``*-it-assistant`` drafter for every variant.

Pass ``--enable_mtp true`` at launch to download the drafter declared by the
model spec and run it alongside the target model:

.. code-block:: bash

   xinference launch --model-name gemma-4 --model-engine vllm \
     --model-format pytorch --size-in-billions 12 --quantization none \
     --enable_mtp true

In the Web UI the same options live under *Advanced Configuration → Speculative
Decoding*, which only appears for a format/size that actually ships a drafter.

Optional parameters:

* ``--num_speculative_tokens <n>``: how many tokens the drafter proposes per
  round, including the bonus token. Left unset, MLX reads it from the drafter,
  which runs at the depth it was trained for (``4`` for Gemma 4). Gemma 4 on
  llama.cpp, vLLM, and SGLang follows the same model-size recipe: ``2`` for
  E2B, ``4`` for E4B and 26B-A4B, and the lower end (``4``) of the recommended
  ``4-8`` range for 12B and 31B. Other llama.cpp models keep xllamacpp's own
  default.
* ``--draft_quantization <quantization>``: which drafter conversion to use, when
  the spec declares more than one — the MLX build of Gemma 4 12B publishes
  eight. Defaults to the first declared, which is the least quantized one: a
  drafter is small and quantizing it costs acceptance rate, so pairing a
  quantized target with an unquantized drafter is the recommended setup.
* ``--draft_model_path <path>``: use a local drafter instead of the one declared
  by the spec. Implies ``--enable_mtp true``.

For these paired Gemma 4 drafters, the family and size must match the target —
the drafter shares the target's KV cache, so a mismatched checkpoint is rejected
rather than silently degrading.

Gemma 4 engine support:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Engine
     - Requirement
     - Notes
   * - :ref:`vLLM <vllm_backend>`
     - ``vllm>=0.22.0``, ``transformers>=5.8.0``
     - Translated into ``speculative_config`` with ``method: mtp``. An
       explicitly provided ``speculative_config`` is left untouched. Older
       vLLM treats the drafter as a generic draft model, while older
       Transformers does not recognize ``gemma4_assistant``; either case is
       rejected before engine initialization. Virtual-environment launches
       also synchronize ``flashinfer-cubin`` with ``flashinfer-python`` before
       starting vLLM, repairing stale environments that contain mismatched
       FlashInfer packages.
   * - :ref:`SGLang <sglang_backend>`
     - ``sglang==0.5.13.post1``, ``transformers==5.8.1``
     - Translated into ``--speculative-algorithm NEXTN`` with the matching
       ``speculative_num_steps`` / ``speculative_num_draft_tokens``. An
       explicitly provided ``speculative_algorithm`` is left untouched.
   * - :ref:`MLX <mlx_backend>`
     - ``mlx-vlm>=0.5.0`` (``>=0.6.1`` for Gemma 4 12B)
     - Served by the MLX vision engine, the one that runs multimodal models
       such as Gemma 4. The drafter is validated against the target when the
       model loads.
   * - :ref:`llama.cpp <llama_cpp_backend>`
     - ``xllamacpp>=2026.6.9713``
     - Translated into the ``draft-mtp`` speculative implementation. The
       drafter is a single gguf published inside the target's own repository,
       so its quantizations are its own (``BF16``, ``F16``, ``Q8_0`` for Gemma
       4) and independent of the target's. Earlier llama.cpp builds do not
       know the ``gemma4-assistant`` architecture and cannot load it.

Not supported by the Transformers engine: it runs its own continuous-batching
loop rather than ``generate()``, so there is nowhere to attach a drafter.

TRT-LLM
-------

.. versionadded:: v4.1.0

Ordinary TRT-LLM replicas support NGram and Draft/Target speculative decoding
with ``tensorrt-llm==1.2.1`` on Linux/CUDA. Model architectures and checkpoint
formats follow :ref:`trt_llm_backend`, with TP=PP=CP=1. Xavier shared caching, native P/D and Xavier
P/D do not support speculation through this adapter; see
:ref:`user_guide_pd_separation` for their supported configurations.

NGram reuses token patterns from the prompt and generated text without loading
a draft model. For a workload with repeated text, pass the native
``speculative_config`` at launch:

.. code-block:: python

   from xinference.client import Client

   client = Client("http://localhost:9997")
   common = dict(
       model_name="qwen3",
       model_size_in_billions="0_6",
       model_format="pytorch",
       model_engine="TRT-LLM",
       quantization="none",
       n_gpu=1,
       context_length=4096,
       enable_virtual_env=False,
   )
   client.launch_model(
       model_uid="qwen-ngram",
       **common,
       speculative_config={
           "decoding_type": "NGram",
           "max_draft_len": 3,
           "max_matching_ngram_size": 4,
           "is_public_pool": False,
       },
   )

Draft/Target loads a separate draft model. Provide its local checkpoint path
with ``draft_model_path``; ``num_speculative_tokens`` maps to TRT's
``max_draft_len`` and defaults to ``3``. This counts draft tokens and excludes
the target's bonus token. The draft and target must have compatible tokenizers
and vocabularies, and both must be supported by TRT-LLM 1.2.1. Use a smaller
dense Qwen3 checkpoint for the draft when serving a larger dense Qwen3 target.
This mode does not require the Gemma-style paired assistant or a shared KV cache.

.. code-block:: python

   client.launch_model(
       model_uid="qwen-draft-target",
       **dict(common, model_size_in_billions="1_7"),
       draft_model_path="/models/Qwen3-0.6B",
       num_speculative_tokens=3,
   )

The examples require a compatible TRT-LLM environment. Change the model sizes
and draft path to checkpoints available on your worker. A draft consumes
additional GPU memory and compute, so it may reduce throughput at high load.

An explicit ``speculative_config`` takes precedence over ``draft_model_path``
and ``num_speculative_tokens``. The adapter defaults
``disable_overlap_scheduler=True`` when speculation is configured. Use native
TRT options only with combinations supported by the pinned runtime. EAGLE3
and model-specific MTP require the corresponding architecture and speculative
checkpoint configuration. ``enable_mtp=True`` only downloads a drafter declared
in the Xinference model spec; it does not configure a TRT speculative method.

When it pays off
----------------
Speculation is only worth it when a drafting step is cheap *relative to* a
target decoding step. The drafter is a small dense model whose cost does not
change with the target, so the ratio is what decides the outcome — and a
mixture-of-experts target can land on the wrong side of it, because only its
activated slice is read per token:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - gemma-4 MLX, 4bit, M5 Pro
     - Without a drafter
     - With MTP
     - Accepted per round
   * - 31B (dense, 18.4 GB read per token)
     - 14.9 tok/s
     - **31.0 tok/s** (2.1x)
     - 2.08 of 4
   * - 26B-A4B (MoE, ~2.2 GB read per token)
     - 73.2 tok/s
     - 65.9 tok/s (0.9x)
     - 1.40 of 4

The 0.83 GB drafter costs about 5% of a 31B decoding step but 39% of a
26B-A4B one, so on the MoE the three drafting steps of a round already exceed
one plain decoding step — and the round has to win that back from a lower
acceptance rate. The MoE is still the faster model here in absolute terms; it
simply has no headroom left for speculation.

So measure before leaving it on, and note that the verification step itself is
not the problem: a four-token forward costs only ~40% more than a single-token
one on either model.
