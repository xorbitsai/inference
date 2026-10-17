.. _anthropic_api:

Anthropic API compatibility
===========================

Xinference exposes Anthropic-compatible Messages APIs for already launched
models. Use the model UID in requests. The available features depend on the
model and engine.

Endpoints and authentication
----------------------------

Send message requests to ``/v1/messages`` and token-count requests to
``/v1/messages/count_tokens``. The standard routes require
``anthropic-version: 2023-06-01``. The ``/anthropic/v1/messages`` and
``/anthropic/v1/messages/count_tokens`` aliases remain available.

Use your Xinference API key when authentication is enabled. An arbitrary
non-empty key is sufficient when authentication is disabled.

Claude Code
-----------

Launch a model with chat and tool-calling support first (see :ref:`launch`),
then configure Claude Code to use the Xinference server and model UID:

.. code-block:: bash

    export ANTHROPIC_BASE_URL="http://localhost:9997"
    export ANTHROPIC_API_KEY="YOUR_XINFERENCE_API_KEY"
    export ANTHROPIC_MODEL="YOUR_LAUNCHED_MODEL_UID"
    export ANTHROPIC_DEFAULT_HAIKU_MODEL="$ANTHROPIC_MODEL"
    export ANTHROPIC_DEFAULT_SONNET_MODEL="$ANTHROPIC_MODEL"
    export ANTHROPIC_DEFAULT_OPUS_MODEL="$ANTHROPIC_MODEL"
    claude --model "$ANTHROPIC_MODEL"

Python SDK
----------

.. code-block:: python

    import anthropic

    model_uid = "YOUR_LAUNCHED_MODEL_UID"
    client = anthropic.Anthropic(
        api_key="YOUR_XINFERENCE_API_KEY",
        base_url="http://localhost:9997",
    )
    message = client.messages.create(
        model=model_uid,
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, Claude"}],
    )
    print(message.content)

Token counting
--------------

The standard ``/v1/messages/count_tokens`` endpoint and its
``/anthropic/v1/messages/count_tokens`` alias use the loaded model's chat
template and tokenizer, including tool definitions and images where supported.
Both routes use the same model permissions and Xinference API key as message
generation. SGLang supports text and tool token counting; image counting returns
501.

.. code-block:: python

    token_count = client.messages.count_tokens(
        model=model_uid,
        messages=[{"role": "user", "content": "Hello, Claude"}],
    )
    print(token_count.input_tokens)

Images and documents
--------------------

Messages support text, base64 or URL images, tool results containing images,
and tool references to tools declared in the request. Plain-text documents are
supported; PDF pages are rendered as images with ``pypdfium2`` (install the
``image`` extra), require a vision model, and are limited to 20 pages per
request. File IDs and document citations are not supported.

The request-wide safety limit is 64 rendered images, counting direct images,
tool-result images and PDF pages together. A lower deployment limit still
applies. Exceeding either limit returns HTTP 400 before generation, including
for streaming requests.

Xinference's vLLM vision models default to two images per prompt. The limit
includes images in conversation history and tool results, so successive
single-image messages can also exceed it. Set ``limit_mm_per_prompt`` when
launching the model, for example ``{"image": 8, "video": 0}``, to allow more
images within the model's context length and available GPU memory.

PDF limits apply to the entire request: at most 20 pages and 20 million
rendered pixels. Decoded PDF bytes and rendered PNG bytes each use
``XINFERENCE_MEDIA_MAX_BYTES`` (64 MiB by default). Each PDF block is rendered
once per request, and model access is checked before fetching or rendering content.

Thinking
--------

Adaptive thinking enables the model's reasoning mode. The ``effort`` option
maps to prompt guidance, with ``medium`` preserving the default prompt.
Explicit thinking responses carry Xinference signatures, which can be replayed
with the same model UID and signing key; configure
``XINFERENCE_ANTHROPIC_THINKING_SECRET`` consistently across API servers.
SGLang enforces manual ``budget_tokens`` with its request-level
``max_thinking_tokens`` control when strict thinking is enabled. Unsupported
versions or disabled strict thinking return HTTP 501 for a manual budget.
MLX and vLLM do not currently enforce a separate reasoning-token limit.

Stale, unsigned or foreign thinking blocks are ignored during replay;
accompanying answers and tool calls are preserved.

On physical vLLM models, ``stop_sequences`` applies to streaming and non-streaming
requests, including when thinking is disabled. A stop match during thinking
can end the response before a final answer is produced. A matching
caller-provided string returns ``stop_reason: "stop_sequence"`` and the exact
``stop_sequence``. EOS and token-ID stops return ``end_turn`` with a null
``stop_sequence``.

For physical vLLM models, disconnecting from a streaming or non-streaming
Messages request aborts generation and releases request capacity.

Context management
------------------

``context_management`` supports ``clear_thinking_20251015``. Set ``keep`` to
``"all"`` or ``{"type": "all"}`` to retain all thinking (the default), or use
a positive ``thinking_turns`` value to retain recent thinking turns:

.. code-block:: json

    {
      "context_management": {
        "edits": [{
          "type": "clear_thinking_20251015",
          "keep": {"type": "thinking_turns", "value": 1}
        }]
      }
    }

User messages containing only ``tool_result`` blocks do not start a new
assistant turn. Additional user content starts a new turn. Thinking attached
to unresolved tool calls is retained. Generation, cache-prefix processing
and token counting use the same edited history. Omitting ``edits`` is a no-op;
unsupported edit types return HTTP 400.

Tool choice and structured output
---------------------------------

The vLLM, SGLang and MLX engines support structured JSON,
named and required tool choice through constrained JSON decoding. Disabled
parallel tool use returns at most one tool per response. Forced tool arguments
are buffered until complete.

Xinference automatically connects its existing reasoning parser to
vLLM and SGLang structured decoding for supported reasoning markers and engines with parser
plugin support. Qwen3.5 needs no extra parser configuration. Explicit engine
parser settings are preserved. Manual thinking cannot be combined with forced
tool choice. P/D models return 501 for forced tool choice. Other engines return
501 for unsupported controls. MLX uses ``llguidance`` (included in the
``mlx`` extra) with a fast tokenizer. Requests with constrained output or explicit cache checkpoints use
request-local decoding; other text requests retain continuous batching.

SGLang counts image tokens through a zero-output native prefill, using the
same image processing as generation. This performs image processing and
prefill work and can warm the engine cache.

With compatible SGLang versions, Xinference enables strict thinking for its
reasoning adapter on the default XGrammar backend, including ordinary requests.
This adds token-filtering overhead, requires a compatible tokenizer and does not
support UNO speculative decoding. Set ``enable_strict_thinking=False`` in model
launch options to disable it; EOS can then end thinking before structured output
begins.

Backend capabilities
--------------------

Xinference checks explicit feature requests against the loaded model and
engine before generation. Unsupported controls return HTTP 501; image input
to a text-only model returns HTTP 400. Translated, emulated, degraded and
rejected compatibility decisions are logged with their reasons. Capability
checks do not add features that the engine cannot implement.

The following limitations still apply. Supported controls also require a
compatible model and engine version, as described above.

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18

   * - Feature
     - vLLM
     - SGLang
     - MLX
   * - Forced tool choice / disabling parallel tools
     - Supported
     - Supported
     - Supported
   * - Structured JSON
     - Supported
     - Supported
     - Supported
   * - Image token counting
     - Supported
     - Supported
     - Supported
   * - Explicit cache lifetime
     - Automatic caching
     - Automatic caching
     - Text requests (LM and VLM)
   * - Separate reasoning-token budget
     - Not enforced
     - Supported with strict thinking
     - Not enforced

Prompt caching
--------------

For text requests, both MLX text and vision models support up to four explicit
``cache_control`` breakpoints. Each checkpoint has its own ``5m`` or ``1h``
lifetime, refreshed on a cache hit. Put longer lifetimes before shorter ones.
Expired checkpoints are removed independently; a live earlier checkpoint can
still be reused after a later one expires. Cache storage obeys the configured
``prompt_cache_size`` and ``prompt_cache_bytes`` capacity limits, so memory
pressure can evict checkpoints before their lifetime ends. Usage reports
actual cache writes and reads.

Image requests, vLLM and SGLang retain automatic engine caching, indicated by
the ``x-xinference-cache-control: automatic`` response header. Their engine
caches do not expose independent per-breakpoint lifetime control.
