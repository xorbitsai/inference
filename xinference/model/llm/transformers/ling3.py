from threading import Event, Thread
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    PytorchGenerateConfig,
    PytorchModelConfig,
)
from ...utils import allow_trust_remote_code
from ..llm_family import LLMFamilyV2, LLMSpecV1, register_transformer
from ..media import validate_messages_media
from .core import PytorchChatModel, register_non_default_model
from .direct_chat import PytorchDirectChatMixin


@register_transformer
@register_non_default_model("BailingMoeV3ForCausalLM")
class Ling3PytorchChatModel(PytorchDirectChatMixin, PytorchChatModel):
    """Transformers adapter for the remote-code Ling-3.0 architecture."""

    _ARCHITECTURE = "BailingMoeV3ForCausalLM"

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        config = super()._sanitize_model_config(pytorch_model_config)
        config["trust_remote_code"] = allow_trust_remote_code(self.model_family)
        config.setdefault("torch_dtype", "auto")
        if self.model_family.model_name == "Ling-3.0-flash":
            # KDA stores recurrent state and a tuple of convolution states in
            # DynamicCache. The generic batching path expects two KV tensors.
            self.allow_batch = False
            config.setdefault("enable_thinking", True)
        return config  # type: ignore

    def _should_use_batching(self) -> bool:
        return self.allow_batch and super()._should_use_batching()

    def build_inputs_from_messages(self, messages: List[Dict], generate_config: Dict):
        prompt = self._get_full_prompt(
            messages, generate_config.get("tools"), generate_config
        )
        return self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self._model.get_input_embeddings().weight.device)

    def build_generate_kwargs(self, generate_config: Dict) -> Dict[str, Any]:
        if generate_config.get("n", 1) != 1:
            raise ValueError("Ling-3.0 Transformers direct chat supports n=1 only")
        temperature = generate_config.get("temperature", 0.6)
        kwargs = {
            "max_new_tokens": generate_config.get("max_tokens") or 512,
            "do_sample": temperature > 0,
            "use_cache": True,
            "eos_token_id": self.model_family.stop_token_ids,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs.update(
                temperature=temperature,
                top_p=generate_config.get("top_p", 0.95),
                top_k=generate_config.get("top_k", 20),
            )
        if "repetition_penalty" in generate_config:
            kwargs["repetition_penalty"] = generate_config["repetition_penalty"]
        if generate_config.get("stop"):
            kwargs.update(
                stop_strings=generate_config["stop"], tokenizer=self._tokenizer
            )
        return kwargs

    def build_streaming_iter(
        self, messages: List[Dict], generate_config: Dict
    ) -> Tuple[Iterator, int]:
        from transformers import (
            StoppingCriteria,
            StoppingCriteriaList,
            TextIteratorStreamer,
        )

        inputs = self.build_inputs_from_messages(messages, generate_config)
        kwargs = self.build_generate_kwargs(generate_config)
        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        cancelled = Event()
        errors: List[Exception] = []

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set()

        def generate():
            try:
                self._model.generate(
                    **inputs,
                    **kwargs,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([Cancelled()]),
                )
            except Exception as exc:
                errors.append(exc)
                streamer.end()

        def iterate():
            thread = Thread(target=generate, daemon=True)
            thread.start()
            try:
                yield from streamer
                if errors:
                    raise errors[0]
            finally:
                cancelled.set()

        return iterate(), inputs.input_ids.shape[-1]

    def get_stop_strs(self) -> List[str]:
        return self.model_family.stop or []

    async def _direct_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[PytorchGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        validate_messages_media(messages)
        if generate_config and generate_config.get("stream"):
            return self._to_chat_completion_chunks(
                self.generate_streaming(messages, generate_config),
                self.reasoning_parser,
            )
        return self.generate_non_streaming(messages, generate_config)

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ("pytorch", "fp8"):
            return (
                False,
                "Ling-3.0 Transformers supports BF16, FP8, and compressed-tensors INT4 checkpoints only",
            )
        normalized_quantization = (quantization or "none").lower()
        if llm_spec.model_format == "pytorch" and normalized_quantization not in (
            "none",
            "int4",
        ):
            return False, "Ling-3.0 Transformers only supports none/Int4 quantization"
        if llm_spec.model_format == "fp8" and normalized_quantization != "fp8":
            return False, "Ling-3.0 FP8 checkpoints require FP8 quantization"
        if not llm_family.has_architecture(cls._ARCHITECTURE):
            return False, "Model architecture is not BailingMoeV3ForCausalLM"
        if "chat" not in llm_family.model_ability:
            return False, "Ling-3.0 Transformers requires chat ability"
        return True
