# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Text architectures and checkpoint formats in the pinned PyTorch backend."""

# Keep this aligned with tensorrt_llm._torch.models in TRT-LLM 1.2.1.
# Model sizes, derivatives and MoE variants do not need separate allowlists.
SUPPORTED_ARCHITECTURES = [
    "DeciLMForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "Exaone4ForCausalLM",
    "ExaoneMoEForCausalLM",
    "Gemma3ForCausalLM",
    "Glm4MoeForCausalLM",
    "GptOssForCausalLM",
    "HunYuanDenseV1ForCausalLM",
    "HunYuanMoEV1ForCausalLM",
    "LlamaForCausalLM",
    "MiniMaxM2ForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "NemotronForCausalLM",
    "NemotronHForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "SeedOssForCausalLM",
    "Starcoder2ForCausalLM",
]

# Quantization is read from the checkpoint by TRT, including ModelOpt exports
# and native MXFP4 checkpoints registered as pytorch/none. Generic HF AWQ/GPTQ
# checkpoints are not accepted by the pinned PyTorch loader.
SUPPORTED_FORMATS = {
    "pytorch": {"none"},
    "fp8": {"none", "fp8"},
    "fp4": {"none", "fp4", "nvfp4", "mxfp4"},
}


def match_model(llm_family, llm_spec, quantization):
    if set(llm_family.model_ability) & {"vision", "audio", "omni", "embed"}:
        return False, "TRT-LLM currently supports text generation models only"
    if not llm_family.matches_supported_architectures(SUPPORTED_ARCHITECTURES):
        return False, "Model architecture is not supported by TRT-LLM 1.2.1"
    allowed = SUPPORTED_FORMATS.get(llm_spec.model_format, set())
    if (quantization or "none").lower() not in allowed:
        return (
            False,
            "TRT-LLM supports pytorch/fp8/fp4 checkpoints; use ModelOpt exports for AWQ",
        )
    return True
