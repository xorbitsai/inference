# Copyright 2022-2024 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, TypedDict, Union

import numpy as np

from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
)
from ..core import LLM
from ..llm_family import BUILTIN_LLM_FAMILIES
from ..utils import ChatModelMixin

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from torch import Tensor

try:
    import tensorrt_llm
    import torch
    from tensorrt_llm.quantization import QuantMode
    from tensorrt_llm.runtime import ModelConfig, SamplingConfig

    TRTLLM_INSTALLED = True
except ImportError:
    TRTLLM_INSTALLED = False


class TRTModelConfig(TypedDict, total=False):
    tokenizer_dir: str


class TRTGenerateConfig(TypedDict, total=False):
    max_tokens: int
    end_id: int
    pad_id: int
    num_beams: int
    temperature: float
    top_k: int
    top_p: int
    length_penalty: float
    repetition_penalty: float
    min_length: int
    presence_penalty: float
    use_beam_hyps: bool

    stream: bool
    stream_interval: int


MODEL_SPECIAL_TOKENS = {
    "llama-2": {"EOS_TOKEN": 2, "PAD_TOKEN": 2},
    "llama-2-chat": {"EOS_TOKEN": 2, "PAD_TOKEN": 2},
}
MODEL_NAME_TO_FAMILY = dict(
    (family.model_name, family) for family in BUILTIN_LLM_FAMILIES
)


def read_config(config_path: Path):
    with open(config_path, "r") as f:
        config = json.load(f)
    use_gpt_attention_plugin = config["plugin_config"]["gpt_attention_plugin"]
    remove_input_padding = config["plugin_config"]["remove_input_padding"]
    dtype = config["builder_config"]["precision"]
    tp_size = config["builder_config"]["tensor_parallel"]
    pp_size = config["builder_config"]["pipeline_parallel"]
    world_size = tp_size * pp_size
    assert (
        world_size == tensorrt_llm.mpi_world_size()
    ), f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"
    num_heads = config["builder_config"]["num_heads"] // tp_size
    hidden_size = config["builder_config"]["hidden_size"] // tp_size
    vocab_size = config["builder_config"]["vocab_size"]
    num_layers = config["builder_config"]["num_layers"]
    num_kv_heads = config["builder_config"].get("num_kv_heads", num_heads)
    paged_kv_cache = config["plugin_config"]["paged_kv_cache"]
    tokens_per_block = config["plugin_config"]["tokens_per_block"]
    quant_mode = QuantMode(config["builder_config"]["quant_mode"])
    if config["builder_config"].get("multi_query_mode", False):
        tensorrt_llm.logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
        num_kv_heads = 1
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
    use_custom_all_reduce = config["plugin_config"].get("use_custom_all_reduce", False)

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        paged_kv_cache=paged_kv_cache,
        tokens_per_block=tokens_per_block,
        remove_input_padding=remove_input_padding,
        dtype=dtype,
        quant_mode=quant_mode,
        use_custom_all_reduce=use_custom_all_reduce,
    )

    return model_config, tp_size, pp_size, dtype


def get_engine_name(model, dtype, tp_size, pp_size, rank):
    if pp_size == 1:
        return "{}_{}_tp{}_rank{}.engine".format(model, dtype, tp_size, rank)
    return "{}_{}_tp{}_pp{}_rank{}.engine".format(model, dtype, tp_size, pp_size, rank)


class TRTModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_name: str,
        model_path: str,
        tokenizer_path: str,
    ):
        if model_name not in MODEL_SPECIAL_TOKENS:
            raise ValueError(
                f"Model name must be one of follows: {MODEL_SPECIAL_TOKENS.keys()}"
            )
        self._model_uid: str = model_uid
        self._model_name: str = model_name
        self._model_path: str = model_path
        self._tokenizer_path: str = tokenizer_path
        self._model_config: "ModelConfig" = None
        self._decoder: Any = None
        self._tokenizer: Any = None

        self.runtime_rank = tensorrt_llm.mpi_rank()

    def load(
        self,
        use_py_session: bool = False,  # Whether or not to use Python runtime session
    ):
        try:
            import tensorrt_llm
            from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

            if PYTHON_BINDINGS:
                from tensorrt_llm.runtime import ModelRunnerCpp
        except ImportError:
            error_message = "Failed to import module 'tensorrt_llm'"
            installation_guide = ["Please make sure 'tensorrt_llm' is installed. "]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        try:
            from transformers import AutoTokenizer
        except ImportError:
            error_message = "Failed to import module 'transformers'"
            installation_guide = [
                "Please make sure 'transformers' is installed. ",
                "You can install it by `pip install transformers`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        runner_cls = ModelRunner if use_py_session else ModelRunnerCpp
        engine_dir = Path(self._model_path)
        runner_kwargs = dict(
            engine_dir=engine_dir,
            rank=self.runtime_rank,
        )
        self._runner = runner_cls.from_dir(**runner_kwargs)
        config_path = engine_dir / "config.json"
        model_config, tp_size, pp_size, dtype = read_config(config_path)
        logger.info(
            f"Loading {self._model_uid} with following model config: {model_config}"
        )
        runtime_mapping = tensorrt_llm.Mapping(1, 0, tp_size=tp_size, pp_size=pp_size)
        engine_name = get_engine_name(self._model_name, dtype, tp_size, pp_size, 0)
        serialize_path = engine_dir / engine_name
        with open(serialize_path, "rb") as f:
            engine_buffer = f.read()
        self._model_config = model_config
        self._decoder = tensorrt_llm.runtime.GenerationSession(
            model_config, engine_buffer, runtime_mapping
        )

    def _sanitize_generate_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> TRTGenerateConfig:
        if not generate_config:
            generate_config = {}

        sanitized = TRTGenerateConfig()
        default_eos_token = MODEL_SPECIAL_TOKENS[self._model_name]["EOS_TOKEN"]
        default_pad_token = MODEL_SPECIAL_TOKENS[self._model_name]["PAD_TOKEN"]
        sanitized.setdefault("end_id", generate_config.get("end_id", default_eos_token))
        sanitized.setdefault("pad_id", generate_config.get("pad_id", default_pad_token))

        sanitized.setdefault("max_tokens", generate_config.get("max_tokens", 512))
        sanitized.setdefault("num_beams", generate_config.get("num_beams", 1))
        sanitized.setdefault("temperature", generate_config.get("temperature", 1.0))
        sanitized.setdefault("top_k", generate_config.get("top_k", 1))
        sanitized.setdefault("top_p", generate_config.get("top_p", 0.0))
        sanitized.setdefault(
            "length_penalty", generate_config.get("length_penalty", 1.0)
        )
        sanitized.setdefault(
            "repetition_penalty", generate_config.get("repetition_penalty", 1.0)
        )
        sanitized.setdefault("min_length", generate_config.get("min_length", 1))
        sanitized.setdefault(
            "presence_penalty", generate_config.get("presence_penalty", 0.0)
        )
        sanitized.setdefault(
            "use_beam_hyps", generate_config.get("use_beam_hyps", True)
        )
        sanitized.setdefault("stream", generate_config.get("stream", None))
        sanitized.setdefault(
            "stream_interval", generate_config.get("stream_interval", 5)
        )
        return sanitized

    def _gen_completion_chunk(
        self, out_ids: "Tensor", num_beams: int, out_start: int, out_end: int
    ):
        choices = []
        for beam in range(num_beams):
            ids = out_ids[0][beam][out_start:out_end].tolist()
            out_text = self._tokenizer.decode(ids)
            completion_choice = CompletionChoice(
                text=out_text, index=beam, logprobs=None, finish_reason=None
            )
            choices.append(completion_choice)
        completion_chunk = CompletionChunk(
            id=str(uuid.uuid1()),
            object="text_completion",
            created=int(time.time()),
            model=self._model_uid,
            choices=choices,
        )
        return completion_chunk

    def generate(
        self, prompt: str, generate_config: Optional[Dict] = None
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        if generate_config is None:
            generate_config = dict()
        sanitized_generate_config = self._sanitize_generate_config(generate_config)
        max_tokens = sanitized_generate_config.pop("max_tokens")
        stream = sanitized_generate_config.pop("stream")
        stream_interval = sanitized_generate_config.pop("stream_interval")
        num_beams = sanitized_generate_config.pop("num_beams")
        end_id = sanitized_generate_config.pop("end_id")
        pad_id = sanitized_generate_config.pop("pad_id")
        sampling_config = SamplingConfig(**sanitized_generate_config)

        input_tokens = [self._tokenizer.encode(prompt, add_special_tokens=False)]
        input_lengths = torch.tensor(
            [len(x) for x in input_tokens], dtype=torch.int32, device="cuda"
        )
        if self._model_config.remove_input_padding:
            input_ids = np.concatenate(input_tokens)
            input_ids = torch.tensor(
                input_ids, dtype=torch.int32, device="cuda"
            ).unsqueeze(0)
        else:
            input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
                sampling_config["end_id"],
            ).cuda()

        max_input_length = torch.max(input_lengths).item()
        self._decoder.setup(
            input_lengths.size(0),
            max_input_length,
            max_tokens,
            num_beams,
        )

        # # An example to stop generation when the model generate " London" on first sentence, " eventually became" on second sentence
        # stop_words_list = [[" London"], ["eventually became"]]
        # stop_words_list = tensorrt_llm.runtime.to_word_list_format(stop_words_list, tokenizer)
        # stop_words_list = torch.Tensor(stop_words_list).to(torch.int32).to("cuda").contiguous()
        stop_words_list = None

        # # An example to prevent generating " chef" on first sentence, " eventually" and " chef before" on second sentence
        # bad_words_list = [[" chef"], [" eventually, chef before"]]
        # bad_words_list = tensorrt_llm.runtime.to_word_list_format(bad_words_list, tokenizer)
        # bad_words_list = torch.Tensor(bad_words_list).to(torch.int32).to("cuda").contiguous()
        bad_words_list = None

        with torch.no_grad():
            outputs = self._runner.generate(
                batch_input_ids=input_ids,
                max_new_tokens=max_tokens,
                end_id=end_id,
                pad_id=pad_id,
                num_beams=num_beams,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                streaming=stream,
                output_sequence_lengths=True,
                return_dict=True,
            )
            torch.cuda.synchronize()

        if stream:
            for i, out in enumerate(outputs):
                if not i % stream_interval:
                    if self.runtime_rank == 0:
                        output_ids = out["output_ids"]
                        sequence_lengths = out["sequence_lengths"]

                        batch_size, num_beams, _ = output_ids.size()
                        for batch_idx in range(batch_size):
                            for beam in range(num_beams):
                                output_begin = input_lengths[batch_idx]
                                output_end = sequence_lengths[batch_idx][beam]
                                outputs = output_ids[batch_idx][beam][
                                    output_begin:output_end
                                ].tolist()
                                output_text = self._tokenizer.decode(outputs)
                                print(
                                    f'Output [Text {batch_idx} Beam {beam}]: "{output_text}"'
                                )
                    yield out

            if i % stream_interval:
                if self.runtime_rank == 0:
                    output_ids = out["output_ids"]
                    sequence_lengths = out["sequence_lengths"]

                    batch_size, num_beams, _ = output_ids.size()
                    for batch_idx in range(batch_size):
                        for beam in range(num_beams):
                            output_begin = input_lengths[batch_idx]
                            output_end = sequence_lengths[batch_idx][beam]
                            outputs = output_ids[batch_idx][beam][
                                output_begin:output_end
                            ].tolist()
                            output_text = self._tokenizer.decode(outputs)
                            print(
                                f'Output [Text {batch_idx} Beam {beam}]: "{output_text}"'
                            )
                yield out
        else:
            if self.runtime_rank == 0:
                output_ids = outputs["output_ids"]
                sequence_lengths = outputs["sequence_lengths"]

                batch_size, num_beams, _ = output_ids.size()
                for batch_idx in range(batch_size):
                    for beam in range(num_beams):
                        output_begin = input_lengths[batch_idx]
                        output_end = sequence_lengths[batch_idx][beam]
                        outputs = output_ids[batch_idx][beam][
                            output_begin:output_end
                        ].tolist()
                        output_text = self._tokenizer.decode(outputs)
                        print(f'Output [Text {batch_idx} Beam {beam}]: "{output_text}"')

            completion = self._gen_completion_chunk(
                output_ids, num_beams, len(input_lengths), len(output_ids)
            )
            choices = completion["choices"]
            completion_tokens = 0
            for beam in range(num_beams):
                completion_tokens += int(
                    (
                        output_ids[0][beam] == sanitized_generate_config["end_id"]
                    ).nonzero(as_tuple=True)[0][0]
                )
            usage = CompletionUsage(
                prompt_tokens=len(input_lengths),
                completion_tokens=completion_tokens,
                total_tokens=len(input_lengths) + completion_tokens,
            )
            return Completion(
                id=str(uuid.uuid1()),
                object="text_completion",
                created=int(time.time()),
                model=self._model_uid,
                choices=choices,
                usage=usage,
            )


class TRTChatModel(TRTModel, ChatModelMixin):
    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[ChatCompletionMessage]] = None,
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        model_family = MODEL_NAME_TO_FAMILY[self._model_name]
        assert model_family.prompt_style is not None
        prompt_style = model_family.prompt_style.copy()
        if system_prompt:
            prompt_style.system_prompt = system_prompt
        chat_history = chat_history or []
        full_prompt = self.get_prompt(prompt, chat_history, prompt_style)
        if not generate_config:
            generate_config = dict()
        stream = generate_config.get("stream", None)
        if stream:
            it = self.generate(full_prompt, generate_config)
            assert isinstance(it, Iterator)
            return self._to_chat_completion_chunks(it)
        else:
            c = self.generate(full_prompt, generate_config)
            assert not isinstance(c, Iterator)
            return self._to_chat_completion(c)
