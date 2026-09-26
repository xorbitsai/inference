"""Runtime patches that make vLLM serve the Confucius4-TTS T2S model.

Importing this module has two side effects, both required before an engine is
created:

1. Registers the custom ``Text2SemanticVLLM`` architecture in vLLM's model
   registry so it can be loaded by name.
2. Monkeypatches ``GPUModelRunner._prepare_inputs`` so that, for this model,
   position ids are shifted to be relative to the start of the *semantic*
   sequence (excluding the speaker/text/BOS prefix). The T2S positional
   embeddings are trained on that convention, so without this correction the
   generated audio would be wrong.
"""

from functools import wraps
from inspect import signature

import vllm
from vllm import ModelRegistry
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from confuciustts.llm.llm_vllm import Text2SemanticVLLM
from confuciustts.llm.vllm_compat import correct_confucius_positions

ModelRegistry.register_model("Text2SemanticVLLM", Text2SemanticVLLM)
print("Registered Text2SemanticVLLM into vLLM ModelRegistry")


def register_models():
    # Kept as an explicit no-op entry point: importing this module already does
    # the registration, but callers can reference this to force the import.
    pass


if not getattr(
    GPUModelRunner._prepare_inputs, "_confucius4_tts_position_patch", False
):
    _original_prepare_inputs = GPUModelRunner._prepare_inputs
    _prepare_inputs_signature = signature(_original_prepare_inputs)
    _required_parameters = {"scheduler_output", "num_scheduled_tokens"}
    if not _required_parameters.issubset(_prepare_inputs_signature.parameters):
        raise RuntimeError(
            "Confucius4-TTS requires vLLM GPUModelRunner._prepare_inputs to accept "
            "scheduler_output and num_scheduled_tokens "
            f"(vLLM {getattr(vllm, '__version__', 'unknown')})."
        )

    @wraps(_original_prepare_inputs)
    def _prepare_inputs(self, *args, **kwargs):
        bound_arguments = _prepare_inputs_signature.bind(self, *args, **kwargs)
        result = _original_prepare_inputs(self, *args, **kwargs)

        if isinstance(self.get_model(), Text2SemanticVLLM):
            correct_confucius_positions(
                self,
                bound_arguments.arguments["scheduler_output"],
                bound_arguments.arguments["num_scheduled_tokens"],
            )

        return result

    _prepare_inputs._confucius4_tts_position_patch = True
    GPUModelRunner._prepare_inputs = _prepare_inputs
    print("GPUModelRunner._prepare_inputs wrapped for Confucius4-TTS position correction")
