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
from .core import PytorchChatModel, register_non_default_model
from .direct_chat import PytorchDirectChatMixin


@register_transformer
@register_non_default_model("DeepseekV41ForCausalLM")
class DeepSeekV41PytorchChatModel(PytorchDirectChatMixin, PytorchChatModel):
    """Transformers adapter for the native DeepSeek V4.1 architecture."""

    _ARCHITECTURE = "DeepseekV41ForCausalLM"

    @classmethod
    def check_lib(cls):
        result = super().check_lib()
        if result is not True:
            return result
        from importlib.metadata import version

        from packaging.version import Version

        if Version(version("transformers")) < Version("5.6.0"):
            return False, "DeepSeek V4.1 requires transformers>=5.6.0"
        return True

    def _sanitize_model_config(
        self, pytorch_model_config: Optional[PytorchModelConfig]
    ) -> PytorchModelConfig:
        config = super()._sanitize_model_config(pytorch_model_config)
        config["trust_remote_code"] = allow_trust_remote_code(self.model_family)
        config.setdefault("torch_dtype", "auto")
        self.allow_batch = False
        config.setdefault("enable_thinking", True)
        return config  # type: ignore

    def _should_use_batching(self) -> bool:
        return self.allow_batch and super()._should_use_batching()

    def _get_full_prompt(self, messages, tools, generate_config):
        normalized = []
        for message in self.prefill_messages(messages):
            content = message.get("content")
            if isinstance(content, list):
                if any(part.get("type") != "text" for part in content):
                    raise ValueError(
                        "DeepSeek V4.1 Transformers currently supports text inputs only"
                    )
                message = {
                    **message,
                    "content": "".join(part.get("text", "") for part in content),
                }
            normalized.append(message)
        kwargs = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, self.reasoning_parser
            )
            or {}
        )
        if tools:
            kwargs["tools"] = tools
        return self.get_full_context(
            normalized,
            self.model_family.chat_template or None,
            tokenizer=self._tokenizer,
            **kwargs,
        )

    def build_inputs_from_messages(self, messages: List[Dict], generate_config: Dict):
        prompt = self._get_full_prompt(
            messages, generate_config.get("tools"), generate_config
        )
        return self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self._model.get_input_embeddings().weight.device)

    def build_generate_kwargs(self, generate_config: Dict) -> Dict[str, Any]:
        if generate_config.get("n", 1) != 1:
            raise ValueError("DeepSeek V4.1 Transformers direct chat supports n=1 only")
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
        return self.build_direct_chat_result(messages, generate_config)

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if not llm_family.has_architecture(cls._ARCHITECTURE):
            return False, "Model architecture is not DeepseekV41ForCausalLM"
        if llm_spec.model_format not in ("pytorch", "fp8"):
            return False, "DeepSeek V4.1 Transformers supports pytorch/fp8 only"
        if llm_spec.model_format == "fp8" and (quantization or "").lower() != "fp8":
            return False, "DeepSeek V4.1 FP8 requires fp8 quantization"
        if (
            llm_spec.model_format == "pytorch"
            and (quantization or "none").lower() != "none"
        ):
            return False, "DeepSeek V4.1 PyTorch requires unquantized weights"
        if "chat" not in llm_family.model_ability:
            return False, "DeepSeek V4.1 Transformers requires chat ability"
        return True
