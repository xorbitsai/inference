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
* Xavier requires vllm version >= ``0.7.0``, and currently not supports for vllm version >= ``0.11.0`` due to vllm reconstruction.
* Due to the underlying communication not recognizing ``0.0.0.0``, the actual IP address needs to be passed when starting Xinference, for example: ``xinference-local -H 192.168.xx.xx``.
* Xavier only works for Nvidia product. 