# Copyright 2022-2023 XProbe Inc.
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
import logging
import uuid
from typing import AsyncGenerator, Dict, Iterator, List, Optional, TypedDict, Union

import torch

from ....types import ChatCompletion, ChatCompletionChunk, Completion, LoRA
from ..core import LLM
from ..llm_family import LLMFamilyV1, LLMSpecV1
from ..utils import ChatModelMixin, generate_chat_completion, generate_completion_chunk

logger = logging.getLogger(__name__)

try:
    import lmdeploy  # noqa: F401

    LMDEPLOY_INSTALLED = True
except ImportError:
    LMDEPLOY_INSTALLED = False

LMDEPLOY_SUPPORTED_CHAT_MODELS = ["internvl2"]
LMDEPLOY_MODEL_CHAT_TEMPLATE_NAME = {
    "internvl2": "internvl-internlm2",
}


class LMDeployModelConfig(TypedDict, total=False):
    model_format: Optional[str]
    tp: Optional[int]
    session_len: Optional[int]
    max_batch_size: Optional[int]
    cache_max_entry_count: Optional[float]
    cache_block_seq_len: Optional[int]
    enable_prefix_caching: Optional[bool]
    quant_policy: Optional[int]
    rope_scaling_factor: Optional[float]
    use_logn_attn: Optional[bool]
    download_dir: Optional[str]
    revision: Optional[str]
    max_prefill_token_num: Optional[int]
    num_tokens_per_iter: Optional[int]
    max_prefill_iters: Optional[int]


class LMDeployGenerateConfig(TypedDict, total=False):
    n: Optional[int]
    max_new_tokens: Optional[int]
    top_p: Optional[float]
    top_k: Optional[int]
    temperature: Optional[float]
    repetition_penalty: Optional[float]
    ignore_eos: Optional[bool]
    random_seed: Optional[int]
    stop_words: Optional[List[int]]
    bad_words: Optional[List[int]]
    min_new_tokens: Optional[int]
    skip_special_tokens: Optional[bool]
    logprobs: Optional[int]


class LMDeployModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV1",
        model_spec: "LLMSpecV1",
        quantization: str,
        model_path: str,
        model_config: Optional[LMDeployModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(model_uid, model_family, model_spec, quantization, model_path)
        self._model_config: LMDeployModelConfig = self._sanitize_model_config(
            model_config
        )
        if peft_model is not None:
            raise ValueError("LMDEPLOY engine has not supported lora yet.")

    def _sanitize_model_config(
        self, model_config: Optional[LMDeployModelConfig]
    ) -> LMDeployModelConfig:
        if model_config is None:
            model_config = LMDeployModelConfig()
        model_config.setdefault("session_len", 8192)
        if self.model_spec.model_format == "awq":
            model_config.setdefault("model_format", "awq")
        return model_config

    def load(self):
        try:
            import lmdeploy  # noqa: F401, F811
        except ImportError:
            error_message = "Failed to import module 'lmdeploy'"
            installation_guide = [
                "Please make sure 'lmdeploy' is installed. ",
                "You can install it by `pip install lmdeploy`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        raise ValueError("LMDEPLOY engine has not supported generate yet.")

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        return False

    def generate(
        self,
        prompt: str,
        generate_config: Optional[Dict] = None,
    ) -> Union[Completion, Iterator[ChatCompletionChunk]]:
        raise NotImplementedError("LMDeploy generate ablility does not support now.")


class LMDeployChatModel(LMDeployModel, ChatModelMixin):
    def load(self):
        try:
            from lmdeploy import (
                ChatTemplateConfig,
                TurbomindEngineConfig,
                VisionConfig,
                pipeline,
            )
        except ImportError:
            error_message = "Failed to import module 'lmdeploy'"
            installation_guide = [
                "Please make sure 'lmdeploy' is installed. ",
                "You can install it by `pip install lmdeploy`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        chat_temp_name = ""
        family = self.model_family.model_family or self.model_family.model_name
        for key in LMDEPLOY_MODEL_CHAT_TEMPLATE_NAME.keys():
            if family in key:
                chat_temp_name = LMDEPLOY_MODEL_CHAT_TEMPLATE_NAME[key]
                break
        if chat_temp_name == "":
            raise ValueError(f"Can not find correct chat template.")

        chat_template_config = ChatTemplateConfig(chat_temp_name)
        count = torch.cuda.device_count()
        if count > 1:
            self._model_config.setdefault("tp", torch.cuda.device_count())

        self._model = pipeline(
            self.model_path,
            chat_template_config=chat_template_config,
            backend_config=TurbomindEngineConfig(**self._model_config),
            vision_config=VisionConfig(thread_safe=True),
        )

    @classmethod
    def match(
        cls, llm_family: "LLMFamilyV1", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if llm_family.model_name not in LMDEPLOY_SUPPORTED_CHAT_MODELS:
            return False
        return LMDEPLOY_INSTALLED

    async def async_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[Dict] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        stream = (
            generate_config.get("stream", False)
            if isinstance(generate_config, dict)
            else False
        )
        stream_options = (
            generate_config.get("stream_options", None)
            if isinstance(generate_config, dict)
            else False
        )
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        if stream:
            chunk = self._chat_stream(messages, include_usage)
            return self._async_to_chat_completion_chunks(chunk)
        else:
            return await self._chat(messages)

    async def _chat_stream(self, messages, include_usage):
        from lmdeploy.messages import Response

        prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
        completion_id = str(uuid.uuid1())
        finish_reason = None
        async for output in self._generate(
            messages,
            session_id=-1,
            stream_response=True,
        ):
            new_text = output.text if isinstance(output, Response) else output.response
            prompt_tokens = output.input_token_len
            completion_tokens = output.generate_token_len
            total_tokens = prompt_tokens + completion_tokens
            finish_reason = output.finish_reason
            yield generate_completion_chunk(
                chunk_text=new_text,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        yield generate_completion_chunk(
            chunk_text=None,
            finish_reason=finish_reason,
            chunk_id=completion_id,
            model_uid=self.model_uid,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            has_choice=True,
            has_content=False,
        )
        if include_usage:
            yield generate_completion_chunk(
                chunk_text=None,
                finish_reason=None,
                chunk_id=completion_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                has_choice=False,
                has_content=False,
            )

    async def _chat(self, messages) -> ChatCompletion:
        from lmdeploy.messages import Response

        response, finish_reason = "", None
        prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
        async for output in self._generate(
            messages,
            session_id=-1,
            stream_response=False,
        ):
            response += output.text if isinstance(output, Response) else output.response
            prompt_tokens = output.input_token_len
            completion_tokens = output.generate_token_len
            total_tokens = output.input_token_len + output.generate_token_len
            finish_reason = output.finish_reason

        return generate_chat_completion(
            self.model_uid,
            response,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=finish_reason,
        )

    # copy from lmdeploy
    # Reference: lmdeploy.serve.async_engine.py
    async def _generate(
        self,
        messages: List[Dict],
        session_id: int,
        generate_config: Optional[Dict] = None,
        tools: Optional[List[object]] = None,
        stream_response: bool = True,
        sequence_start: bool = True,
        sequence_end: bool = True,  # no interactive mode by default
        step: int = 0,
        do_preprocess: bool = False,
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
        import random

        from lmdeploy.messages import EngineGenerationConfig, GenerationConfig
        from lmdeploy.serve.async_engine import GenOut
        from lmdeploy.tokenizer import DetokenizeState

        from ..utils import get_stop_token_ids_from_config_file

        session_id = -1

        if str(session_id) not in self._model.id2step:
            self._model.id2step[str(session_id)] = 0
        if generate_config is None:
            generate_config = GenerationConfig()
        if type(generate_config) is GenerationConfig:
            generate_config = EngineGenerationConfig.From(
                generate_config, self._model.tokenizer
            )
        if generate_config.stop_words is None:  # type: ignore
            stop_token_ids = get_stop_token_ids_from_config_file(self.model_path)
            if stop_token_ids is not None:
                generate_config.stop_words = stop_token_ids  # type: ignore
        if generate_config.random_seed is None and sequence_start:  # type: ignore
            generate_config.random_seed = random.getrandbits(64)  # type: ignore
        if generate_config.n > 1:  # type: ignore
            logger.warning(
                f"n({generate_config.n}) > 1 hasn't been supported yet. "  # type: ignore
                f"Fallback to 1"
            )
            generate_config.n = 1  # type: ignore

        prompt_input = await self._get_prompt_input(messages)
        prompt = prompt_input["prompt"]
        input_ids = prompt_input["input_ids"]
        finish_reason = None
        logger.info(
            f"prompt={prompt!r}, "
            f"gen_config={generate_config}, "
            f"prompt_token_id={input_ids}, "
            f"adapter_name={adapter_name}."
        )
        logger.info(
            f"session_id={session_id}, "  # type: ignore
            f"history_tokens={self._model.id2step[str(session_id)]}, "
            f"input_tokens={len(input_ids)}, "
            f"max_new_tokens={generate_config.max_new_tokens}, "
            f"seq_start={sequence_start}, seq_end={sequence_end}, "
            f"step={step}, prep={do_preprocess}"
        )

        if generate_config.max_new_tokens is None:  # type: ignore
            # for interactive endpoint, will try maximum possible token num
            generate_config.max_new_tokens = max(  # type: ignore
                128,
                self._model.session_len
                - self._model.id2step[str(session_id)]
                - len(input_ids),
            )
        elif (
            self._model.id2step[str(session_id)]
            + len(input_ids)
            + generate_config.max_new_tokens  # type: ignore
            > self._model.session_len
        ):
            generate_config.max_new_tokens = max(  # type: ignore
                self._model.session_len
                - self._model.id2step[str(session_id)]
                - len(input_ids),
                128,
            )
            logger.error(f"Truncate max_new_tokens to {generate_config.max_new_tokens}")  # type: ignore

        if (
            self._model.id2step[str(session_id)]
            + len(input_ids)
            + generate_config.max_new_tokens  # type: ignore
            > self._model.session_len
        ):
            logger.error(f"run out of tokens. session_id={session_id}.")
            yield GenOut(
                "", self._model.id2step[str(session_id)], len(input_ids), 0, "length"
            )
            if sequence_end is True and sequence_start is False:
                await self._model.end_session(session_id)
        else:
            generator = await self._model.get_generator(False, session_id)
            async with self._model.safe_run(session_id):
                state = DetokenizeState(len(input_ids))
                start_ids_offset = state.ids_offset
                response = ""
                async for outputs in generator.async_stream_infer(
                    session_id=session_id,
                    **prompt_input,
                    gen_config=generate_config,
                    adapter_name=adapter_name,
                    stream_output=stream_response,
                    sequence_start=sequence_start,
                    sequence_end=sequence_end,
                    step=self._model.id2step[str(session_id)],
                ):
                    # decode res
                    res, tokens = (
                        input_ids + outputs.token_ids,
                        outputs.num_token,
                    )  # noqa
                    if len(res) <= state.ids_offset:
                        continue

                    ids_offset = state.ids_offset
                    response, state = self._model.tokenizer.detokenize_incrementally(
                        res,
                        state,
                        skip_special_tokens=generate_config.skip_special_tokens,  # type: ignore
                    )

                    res = res[ids_offset:]
                    logprobs = None
                    if outputs.logprobs:
                        log_offset = ids_offset - start_ids_offset
                        logprobs = outputs.logprobs[log_offset:]

                    # response, history token len,
                    # input token len, gen token len
                    yield GenOut(
                        response,
                        self._model.id2step[str(session_id)],
                        len(input_ids),
                        tokens,
                        finish_reason,
                        res,
                        logprobs,
                    )

                finish_reason = (
                    "length" if tokens >= generate_config.max_new_tokens else "stop"  # type: ignore
                )
                # utf-8 char at the end means it's a potential unfinished
                # byte sequence
                if not response.endswith("�"):
                    response = ""  # avaid returning the last response twice
                yield GenOut(
                    response,
                    self._model.id2step[str(session_id)],
                    len(input_ids),
                    tokens,
                    finish_reason,
                )
                # update step
                self._model.id2step[str(session_id)] += len(input_ids) + tokens
                if sequence_end:
                    self._model.id2step[str(session_id)] = 0
                # manually end pytorch session
                # TODO modify pytorch or turbomind api
                if self._model.backend == "pytorch" and sequence_end:
                    await self._model.end_session(session_id)

    # copy from lmdeploy
    # Reference: lmdeploy.serve.vl_async_engine.py
    async def _get_prompt_input(
        self,
        messages: List[Dict],
        sequence_start: bool = True,
        tools: Optional[List[object]] = None,
        **kwargs,
    ):
        """get input_ids, embeddings and offsets."""
        IMAGE_TOKEN = "<IMAGE_TOKEN>"
        IMAGE_DUMMY_TOKEN_INDEX = 0
        import numpy as np

        model_family = self.model_family.model_family or self.model_family.model_name
        decorated, _ = self.get_specific_prompt(model_family, messages)  # type: ignore
        prompt = messages  # type: ignore

        decorated = decorated.replace("<image>", "<img><IMAGE_TOKEN></img>")

        segs = decorated.split(IMAGE_TOKEN)

        results = {}
        input_ids = []  # type: ignore
        if len(segs) > 1:
            images = await self._model.vl_prompt_template.async_collect_pil_images(
                prompt
            )

            features = await self._model.vl_encoder.async_infer(images)

            from lmdeploy.vl.templates import MiniCPMVTempateWrapper

            if isinstance(self._model.vl_prompt_template, MiniCPMVTempateWrapper):
                (
                    decorated,
                    features,
                ) = self._model.vl_prompt_template.update_image_token(  # noqa: E501
                    decorated, features
                )
                segs = decorated.split(IMAGE_TOKEN)

            features = [x.cpu().numpy() for x in features]
            input_ids = []
            begins = []
            ends = []
            if len(segs) != len(features) + 1:
                logger.error(
                    f"the number of {IMAGE_TOKEN} is not equal "
                    f"to input images, {len(segs) - 1} vs {len(features)}"
                )
                features = features[: len(segs) - 1]
            for i, seg in enumerate(segs):
                if i > 0 and i <= len(features):
                    image_dim = features[i - 1].shape[0]
                    begins.append(len(input_ids))
                    ends.append(begins[-1] + image_dim)
                    input_ids.extend([IMAGE_DUMMY_TOKEN_INDEX] * image_dim)
                seg_ids = self._model.tokenizer.encode(
                    seg, add_bos=((i == 0) and sequence_start)
                )
                input_ids.extend(seg_ids)
            ranges = np.stack([begins, ends], axis=1).tolist()
            results["input_embeddings"] = features
            results["input_embedding_ranges"] = ranges
        else:
            input_ids = self._model.tokenizer.encode(decorated, add_bos=sequence_start)

        results["input_ids"] = input_ids
        results["prompt"] = decorated

        return results
