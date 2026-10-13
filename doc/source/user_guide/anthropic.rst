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
generation.

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
document. File IDs and document citations are not supported.

Thinking
--------

Adaptive thinking enables the model's reasoning mode. The ``effort`` option
maps to prompt guidance, with ``medium`` preserving the default prompt.
Explicit thinking responses carry Xinference signatures, which can be replayed
with the same model UID and signing key; configure
``XINFERENCE_ANTHROPIC_THINKING_SECRET`` consistently across API servers.

Tool choice and structured output
---------------------------------

Structured JSON output requires a supporting engine. The vLLM engine supports
named and required tool choice through constrained JSON decoding, and disabled
parallel tool use returns at most one tool per response. Forced tool arguments
are buffered until complete. Combining adaptive thinking with forced tools
requires a matching ``structured_outputs_config.reasoning_parser`` at model
launch (``qwen3`` for Qwen3.5); manual thinking cannot be combined with forced
tool choice. Other engines return 501 for unsupported controls.

Prompt caching
--------------

For text requests, the MLX vision engine reuses the furthest ``cache_control``
boundary with a ``5m`` or ``1h`` lifetime and reports actual cache writes and
reads. Other paths retain automatic engine caching, indicated by the
``x-xinference-cache-control: automatic`` response header.
