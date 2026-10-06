.. _user_guide_vllm_enhancement:

############################################
Xavier: Share KV Cache between vllm replicas
############################################
For scenarios such as long document queries and multi-round conversations,
the computation during the inference prefill phase can be particularly heavy,
which affects overall throughput and the latency of individual inferences.
Xinference enhances the vllm engine by introducing the ``Xavier`` framework,
enabling KV cache sharing across multiple vllm instances.
This allows KV cache computed by other replicas to be directly reused, avoiding redundant computations.

*****
Usage
*****
Simply add the parameter ``enable_xavier=True`` when starting the vllm model.

***********
Limitations
***********
* Xavier retains the V0 adapter for vLLM >= ``0.7.0`` and < ``0.11.0``.
  The V1 connector supports vLLM >= ``0.21.0``. See :ref:`user_guide_pd_separation`
  for community Prefill/Decode deployment.
* Due to the underlying communication not recognizing ``0.0.0.0``, the actual IP address needs to be passed when starting Xinference, for example: ``xinference-local -H 192.168.xx.xx``.
* Xavier only works for Nvidia product.

SGLang replicas
===============

SGLang >= ``0.5.21`` can share prefix KV cache through Xavier's HiCache storage backend. Launch multiple replicas with ``model_engine="SGLang"`` and ``enable_xavier=True``. The initial adapter supports unquantized FP16 Qwen2, Qwen3 and Llama text models with full attention and TP=PP=DP=1; LoRA, speculation, hybrid attention and multimodal inputs are unsupported.

Xavier loads BF16 checkpoints as FP16 and emits a warning. Both weights and KV caches use FP16.

``xavier_cache_bytes`` sets the deployment's shared CPU cache budget (default: ``536870912``, 512 MiB). Each SGLang replica also allocates its own GPU and HiCache host pools. Eviction or unavailable storage becomes a cache miss for ordinary replicas. Cache namespaces include the actual weights, tokenizer, model configuration, KV geometry and engine format; caches cannot yet be exchanged between vLLM and SGLang.

Ordinary SGLang replicas share CPU HiCache pages. Explicit prefill/decode roles use Xavier GPU-to-GPU transfer instead; see :ref:`user_guide_pd_separation`. Cross-engine P/D is not yet supported.
