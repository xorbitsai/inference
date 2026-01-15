# Copyright 2022-2026 XProbe Inc.
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

import asyncio
import concurrent.futures
import importlib
import logging
import pathlib
import platform
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import xoscar as xo

from ....constants import XINFERENCE_MAX_TOKENS
from ....fields import max_tokens_field
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    Completion,
    CompletionChunk,
    CompletionUsage,
    LoRA,
)
from ...utils import check_dependency_available
from ..core import LLM, chat_context_var
from ..llm_family import LLMFamilyV2, LLMSpecV1
from ..utils import (
    DEEPSEEK_TOOL_CALL_FAMILY,
    QWEN_TOOL_CALL_FAMILY,
    ChatModelMixin,
    generate_completion_chunk,
)

logger = logging.getLogger(__name__)


class MLXBatchModel:
    """Wrapper around MLX-LM BatchGenerator for continuous batching."""

    # Class-level storage for multiple batch generators keyed by (temperature, top_p)
    _batch_generators: Dict[Tuple[float, float], Dict[str, Any]] = (
        {}
    )  # (temp, top_p) -> {'generator': BatchGenerator, 'queues': dict, 'pending': dict, 'active': set, 'task': task}
    _model_ref = None
    _tokenizer_ref = None
    _batch_size = 4
    _max_context_length = 2048
    _stop_tokens: set = set()
    _lock: Optional[asyncio.Lock] = None  # Will be initialized lazily

    def __init__(
        self, model, tokenizer, batch_size: int = 4, max_context_length: int = 2048
    ):
        # Store references for creating new generators on demand
        MLXBatchModel._model_ref = model
        MLXBatchModel._tokenizer_ref = tokenizer
        MLXBatchModel._batch_size = batch_size
        MLXBatchModel._max_context_length = max_context_length
        eos_token_ids = tokenizer.eos_token_ids
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        MLXBatchModel._stop_tokens = set(eos_token_ids or [])

    @staticmethod
    def _get_lock() -> asyncio.Lock:
        """Get or create the async lock."""
        if MLXBatchModel._lock is None:
            MLXBatchModel._lock = asyncio.Lock()
        return MLXBatchModel._lock

    def _get_or_create_generator(self, temperature: float, top_p: float):
        """Get or create a BatchGenerator for the given sampling parameters."""
        key = (round(temperature, 6), round(top_p, 6))

        if key not in MLXBatchModel._batch_generators:
            logger.info(
                f"Creating new BatchGenerator for temperature={temperature}, top_p={top_p}"
            )
            from mlx_lm.generate import BatchGenerator
            from mlx_lm.sample_utils import make_sampler

            # Create sampler with specific settings
            sampler = make_sampler(temp=temperature, top_p=top_p)

            # Create batch generator
            batch_generator = BatchGenerator(
                model=MLXBatchModel._model_ref,
                max_tokens=MLXBatchModel._max_context_length,
                stop_tokens=MLXBatchModel._stop_tokens,
                sampler=sampler,
                completion_batch_size=MLXBatchModel._batch_size,
                prefill_batch_size=1,  # Use 1 for lowest latency
            )

            MLXBatchModel._batch_generators[key] = {
                "generator": batch_generator,
                "queues": {},  # uid -> asyncio.Queue
                "pending": {},  # uid -> list of results
                "active": set(),  # active uids
                "task": None,
            }

        return MLXBatchModel._batch_generators[key]

    def _ensure_background_worker(self, gen_dict):
        """Ensure background worker is running for this generator."""
        if gen_dict["task"] is None:
            loop = asyncio.get_event_loop()
            gen_dict["task"] = loop.create_task(self._background_worker(gen_dict))

    async def _background_worker(self, gen_dict):
        """Background worker that continuously calls next() and distributes results."""
        key = f"temp={gen_dict['generator'].sampler}"
        logger.info(f"Starting BatchGenerator background worker for {key}")
        batch_generator = gen_dict["generator"]

        while True:
            try:
                # Get next batch of results for ALL active requests
                batch_results = batch_generator.next()

                if not batch_results:
                    # No active requests, sleep briefly
                    await asyncio.sleep(0.001)
                    continue

                # Distribute results to respective request queues
                async with MLXBatchModel._get_lock():
                    for result in batch_results:
                        if result.uid in gen_dict["queues"]:
                            queue = gen_dict["queues"][result.uid]
                            # Put result in queue
                            try:
                                queue.put_nowait(result)
                            except asyncio.QueueFull:
                                logger.warning(f"Queue full for uid {result.uid}")
                        else:
                            # Queue not ready yet, cache the result
                            if result.uid not in gen_dict["pending"]:
                                gen_dict["pending"][result.uid] = []
                            gen_dict["pending"][result.uid].append(result)
                            logger.debug(
                                f"Cached result for uid {result.uid}, queue not ready yet"
                            )

                        # Remove from active requests if finished
                        if result.finish_reason is not None:
                            if result.uid in gen_dict["active"]:
                                gen_dict["active"].remove(result.uid)
                                logger.debug(
                                    f"Request {result.uid} finished, removed from active set"
                                )

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.0001)

            except Exception as e:
                logger.error(f"Error in background worker: {e}", exc_info=True)
                await asyncio.sleep(0.01)

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        request_id: Optional[str] = None,
        skip_special_tokens: bool = True,
    ) -> AsyncGenerator[CompletionChunk, None]:
        """Async stream generate method using BatchGenerator."""
        # Get or create generator for this sampling configuration
        gen_dict = self._get_or_create_generator(temperature, top_p)
        batch_generator = gen_dict["generator"]

        # Ensure background worker is running for this generator
        self._ensure_background_worker(gen_dict)

        # Prepare request
        assert MLXBatchModel._tokenizer_ref is not None
        prompt_tokens = MLXBatchModel._tokenizer_ref.encode(prompt)
        input_echo_len = len(prompt_tokens)

        # Create queue first
        queue: asyncio.Queue = asyncio.Queue()

        # Insert prompt into batch to get the real uid
        request_ids = batch_generator.insert([prompt_tokens], max_tokens=max_tokens)
        inserted_uid = request_ids[0]

        logger.debug(
            f"Inserted request {inserted_uid} into batch (temp={temperature}, top_p={top_p})"
        )

        # Add to active requests set
        async with MLXBatchModel._get_lock():
            gen_dict["active"].add(inserted_uid)
            gen_dict["queues"][inserted_uid] = queue

            # Check if there are any pending results (early results that arrived before queue was ready)
            if inserted_uid in gen_dict["pending"]:
                logger.debug(
                    f"Found {len(gen_dict['pending'][inserted_uid])} pending results for uid {inserted_uid}"
                )
                for result in gen_dict["pending"][inserted_uid]:
                    try:
                        queue.put_nowait(result)
                    except asyncio.QueueFull:
                        logger.warning(
                            f"Queue full for uid {inserted_uid} while adding pending results"
                        )
                del gen_dict["pending"][inserted_uid]

        try:
            # Track generated text
            generated_tokens = []
            chunk_id = str(uuid.uuid4())
            token_count = 0

            # Wait for results from background worker
            while True:
                try:
                    # Wait for result with short timeout (so we can check if request is still active)
                    result = await asyncio.wait_for(queue.get(), timeout=0.5)

                    if result.token is not None:
                        generated_tokens.append(result.token)
                        token_count += 1

                        # Decode current token
                        assert MLXBatchModel._tokenizer_ref is not None
                        token_text = MLXBatchModel._tokenizer_ref.decode(
                            [result.token], skip_special_tokens=skip_special_tokens
                        )
                        token_text = token_text.strip("�")

                        # Generate completion chunk
                        yield generate_completion_chunk(
                            chunk_text=token_text,
                            finish_reason=None,
                            chunk_id=chunk_id,
                            model_uid=None,
                            prompt_tokens=input_echo_len,
                            completion_tokens=token_count,
                            total_tokens=input_echo_len + token_count,
                        )

                    # Check if generation is finished
                    if result.finish_reason is not None:
                        # Generate final chunk with finish reason
                        finish_reason = (
                            "length" if result.finish_reason == "length" else "stop"
                        )
                        yield generate_completion_chunk(
                            chunk_text="",
                            finish_reason=finish_reason,
                            chunk_id=chunk_id,
                            model_uid=None,
                            prompt_tokens=input_echo_len,
                            completion_tokens=token_count,
                            total_tokens=input_echo_len + token_count,
                        )
                        logger.debug(
                            f"Request {inserted_uid} finished with {finish_reason}"
                        )
                        return

                except asyncio.TimeoutError:
                    # Check if request is still in active set
                    async with MLXBatchModel._get_lock():
                        is_active = inserted_uid in gen_dict["active"]

                    if not is_active:
                        # Request has finished (removed from active set by background worker)
                        logger.debug(
                            f"Request {inserted_uid} is no longer active, finishing"
                        )
                        return
                    # Otherwise, continue waiting (request is still being processed)

        except Exception as e:
            logger.error(f"Error in generate_stream for request: {e}", exc_info=True)
            # Generate error chunk
            yield generate_completion_chunk(
                chunk_text="",
                finish_reason="error",
                chunk_id=chunk_id,
                model_uid=None,
                prompt_tokens=input_echo_len,
                completion_tokens=token_count,
                total_tokens=input_echo_len + token_count,
            )
        finally:
            # Clean up queue using the correct uid
            async with MLXBatchModel._get_lock():
                if inserted_uid in gen_dict["queues"]:
                    del gen_dict["queues"][inserted_uid]
                    logger.debug(f"Cleaned up queue for uid {inserted_uid}")
                # Also clean up any pending results
                if inserted_uid in gen_dict["pending"]:
                    del gen_dict["pending"][inserted_uid]
                # And remove from active set (in case it's still there)
                if inserted_uid in gen_dict["active"]:
                    gen_dict["active"].remove(inserted_uid)

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stream: bool = False,
        skip_special_tokens: bool = True,
    ) -> str:
        """Async generate method using BatchGenerator."""
        # Non-streaming: collect all tokens
        result_text = ""
        async for chunk in self.generate_stream(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            request_id=None,
            skip_special_tokens=skip_special_tokens,
        ):
            if chunk.get("choices") and len(chunk["choices"]) > 0:
                result_text += chunk["choices"][0].get("text", "")
        return result_text


class MLXModelConfig(TypedDict, total=False):
    revision: Optional[str]
    max_gpu_memory: str
    trust_remote_code: bool
    reasoning_content: bool
    # distributed
    address: Optional[str]
    shard: Optional[int]
    n_worker: Optional[int]
    # batch configuration
    allow_batch: bool
    batch_size: int


class MLXGenerateConfig(TypedDict, total=False):
    max_tokens: int
    temperature: float
    repetition_penalty: Optional[float]
    repetition_context_size: Optional[float]
    top_p: float
    logit_bias: Optional[Dict[int, float]]
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[Union[int, List[int]]]
    stream: bool
    stream_options: Optional[Union[dict, None]]
    tools: Optional[List[Dict]]
    lora_name: Optional[str]


@dataclass
class PromptCache:
    cache: List[Any] = field(default_factory=list)
    model_key: Tuple[str, Optional[str]] = ("", None)
    tokens: List[int] = field(default_factory=list)


def get_context_length(config: dict) -> int:
    """Get the context length of a model from model config."""
    if config.get("max_sequence_length") is not None:
        max_sequence_length = config["max_sequence_length"]
    else:
        max_sequence_length = 2048
    if config.get("seq_length") is not None:
        seq_length = config["seq_length"]
    else:
        seq_length = 2048
    if config.get("max_position_embeddings") is not None:
        max_position_embeddings = config["max_position_embeddings"]
    else:
        max_position_embeddings = 2048
    return max(max_sequence_length, seq_length, max_position_embeddings)


class MLXModel(LLM):
    _rank_to_addresses: Optional[Dict[int, str]]
    allow_batch: bool = False

    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        model_config: Optional[MLXModelConfig] = None,
        peft_model: Optional[List[LoRA]] = None,
    ):
        super().__init__(model_uid, model_family, model_path)
        self._use_fast_tokenizer = True
        self._model_config: MLXModelConfig = self._sanitize_model_config(model_config)
        self._context_length: Optional[int] = None
        # for distributed
        assert model_config is not None
        self._address = model_config.pop("address", None)
        self._n_worker = model_config.pop("n_worker", 1)
        self._shard = model_config.pop("shard", 0)
        self._driver_info = model_config.pop("driver_info", None)  # type: ignore
        self._rank_to_addresses = None
        self._loading_thread = None
        self._loading_error = None
        self._all_worker_started = asyncio.Event()
        self._max_kv_size = None
        self._prompt_cache = None
        if peft_model is not None:
            raise ValueError("MLX engine has not supported lora yet")
        # used to call async
        self._loop = None
        self._batch_model = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        # loop will be passed into ModelWrapper,
        # to call aynsc method with asyncio.run_coroutine_threadsafe
        self._loop = loop  # type: ignore

    def _cleanup_memory(self):
        import gc

        import mlx.core as mx

        # mandatory recycling
        gc.collect()
        # clear the MLX cache
        mx.clear_cache()

    @property
    def driver_info(self) -> Optional[dict]:
        return self._driver_info

    def set_shard_info(self, shard: int, address: str):
        # set shard info to rank 0
        if self._rank_to_addresses is None:
            self._rank_to_addresses = {}
        self._rank_to_addresses[shard] = address
        if len(self._rank_to_addresses) == self._n_worker:
            self._all_worker_started.set()

    async def get_rank_addresses(self) -> Optional[Dict[int, str]]:
        await self._all_worker_started.wait()
        return self._rank_to_addresses

    def _sanitize_model_config(
        self, model_config: Optional[MLXModelConfig]
    ) -> MLXModelConfig:
        if model_config is None:
            model_config = MLXModelConfig()
        model_config.setdefault("revision", self.model_spec.model_revision)
        model_config.setdefault("trust_remote_code", True)
        model_config.setdefault("reasoning_content", False)
        return model_config

    def _sanitize_generate_config(
        self,
        generate_config: Optional[MLXGenerateConfig],
    ) -> MLXGenerateConfig:
        if generate_config is None:
            generate_config = MLXGenerateConfig()

        # default config is adapted from
        # https://github.com/ml-explore/mlx-examples/blob/f212b770d8b5143e23102eda20400ae43340f844/llms/mlx_lm/utils.py#L129
        generate_config.setdefault("temperature", 0.0)
        generate_config.setdefault("logit_bias", None)
        generate_config.setdefault("repetition_penalty", None)
        generate_config.setdefault("repetition_context_size", 20)
        generate_config.setdefault("top_p", 1.0)

        max_tokens = max_tokens_field.default or XINFERENCE_MAX_TOKENS
        if not generate_config.get("max_tokens") and max_tokens:
            generate_config["max_tokens"] = max_tokens  # type: ignore
        return generate_config

    def _load_model(self, **kwargs):
        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError:
            error_message = "Failed to import module 'mlx_lm'"
            installation_guide = [
                "Please make sure 'mlx_lm' is installed. ",
                "You can install it by `pip install mlx_lm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        tokenizer_config = dict(
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
        )
        logger.debug(
            "loading model with tokenizer config: %s, model config: %s",
            tokenizer_config,
            self._model_config,
        )

        cache_limit_gb = kwargs.get("cache_limit_gb", None)
        if cache_limit_gb:
            logger.debug(f"Setting cache limit to {cache_limit_gb} GB")
            mx.metal.set_cache_limit(cache_limit_gb * 1024 * 1024 * 1024)

        self._max_kv_size = kwargs.get("max_kv_size", None)
        self._prompt_cache = PromptCache()

        model, tokenizer = load(
            self.model_path,
            tokenizer_config=tokenizer_config,
            model_config=self._model_config,
        )
        if stop_token_ids := self.model_family.stop_token_ids:
            for stop_token_id in stop_token_ids:
                tokenizer.add_eos_token(stop_token_id)

        # Store model and tokenizer for batch model creation later
        self._model = model
        self._tokenizer = tokenizer

        return model, tokenizer

    def _load_model_shard(self, **kwargs):
        try:
            import mlx.core as mx
            from mlx_lm.utils import load_model, load_tokenizer
        except ImportError:
            error_message = "Failed to import module 'mlx_lm'"
            installation_guide = [
                "Please make sure 'mlx_lm' is installed. ",
                "You can install it by `pip install mlx_lm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        # Ensure some attributes correctly inited by model actor
        assert (
            self._loop is not None and self._rank_to_addresses is not None
        ), "Service not started correctly"

        tokenizer_config = dict(
            use_fast=self._use_fast_tokenizer,
            trust_remote_code=kwargs["trust_remote_code"],
            revision=kwargs["revision"],
        )
        logger.debug(
            "loading model with tokenizer config: %s, model config: %s, shard: %d, n_worker: %d",
            tokenizer_config,
            self._model_config,
            self._shard,
            self._n_worker,
        )

        cache_limit_gb = kwargs.get("cache_limit_gb", None)
        if cache_limit_gb:
            logger.debug(f"Setting cache limit to {cache_limit_gb} GB")
            mx.metal.set_cache_limit(cache_limit_gb * 1024 * 1024 * 1024)

        self._max_kv_size = kwargs.get("max_kv_size", None)
        self._prompt_cache = PromptCache()

        self._model, config = load_model(
            pathlib.Path(self.model_path),
            lazy=True,
            get_model_classes=self._get_classes,
        )
        model = self._model.model
        model.rank = self._shard
        model.world_size = self._n_worker
        model.model_uid = self.model_uid
        model.loop = self._loop
        model.address = self._address
        model.rank_to_addresses = self._rank_to_addresses

        # create actors and so forth
        model.prepare()
        # real load the partial weights
        model.pipeline()
        mx.eval(model.parameters())

        self._tokenizer = load_tokenizer(
            pathlib.Path(self.model_path),
            tokenizer_config,
            eos_token_ids=config.get("eos_token_id", None),
        )

    @staticmethod
    def _get_classes(config: dict):
        """
        Retrieve the model and model args classes based on the configuration
        that supported distributed inference.

        Args:
            config (dict): The model configuration.

        Returns:
            A tuple containing the Model class and the ModelArgs class.
        """
        from mlx_lm.utils import MODEL_REMAPPING

        model_type = config["model_type"]
        model_type = MODEL_REMAPPING.get(model_type, model_type)
        try:
            arch = importlib.import_module(
                f"xinference.model.llm.mlx.distributed_models.{model_type}"
            )
        except ImportError:
            msg = f"Model type {model_type} not supported for distributed inference."
            logger.error(msg)
            raise ValueError(msg)

        return arch.Model, arch.ModelArgs

    def load(self):
        reasoning_content = self._model_config.pop("reasoning_content")
        enable_thinking = self._model_config.pop("enable_thinking", True)
        self.prepare_parse_reasoning_content(
            reasoning_content, enable_thinking=enable_thinking
        )
        self.prepare_parse_tool_calls()

        kwargs = {}
        kwargs["revision"] = self._model_config.get(
            "revision", self.model_spec.model_revision
        )
        kwargs["trust_remote_code"] = self._model_config.get("trust_remote_code")
        kwargs["cache_limit_gb"] = self._model_config.pop("cache_limit_gb", None)

        if self._n_worker <= 1:
            self._model, self._tokenizer = self._load_model(**kwargs)
        else:

            def _load():
                try:
                    if self._shard == 0:
                        self._driver_info = {"address": self._address}
                        self.set_shard_info(0, self._address)
                    else:
                        assert self._driver_info is not None
                        driver_address = self._driver_info["address"]

                        async def wait_for_all_shards():
                            model_ref = await xo.actor_ref(
                                address=driver_address, uid=self.raw_model_uid
                            )
                            # set shard info
                            await model_ref.set_shard_info(self._shard, self._address)
                            # wait for all shards
                            self._rank_to_addresses = (
                                await model_ref.get_rank_addresses()
                            )

                        asyncio.run_coroutine_threadsafe(
                            wait_for_all_shards(), self._loop
                        ).result()

                    self._load_model_shard(**kwargs)
                except:
                    logger.exception("Loading mlx shard model failed")
                    self._loading_error = sys.exc_info()

            # distributed inference
            self._loading_thread = threading.Thread(target=_load)
            self._loading_thread.start()

    def wait_for_load(self):
        from mlx_lm.utils import load_config

        if self._loading_thread:
            self._loading_thread.join()
            if self._loading_error:
                _, err, tb = self._loading_error
                raise err.with_traceback(tb)

        # get context length
        config = load_config(Path(self.model_path))
        config.update(self._model_config)
        self._context_length = get_context_length(config)

        # Update allow_batch based on distributed inference
        # Only enable continuous batching for non-distributed inference (single worker)
        n_worker = self._n_worker if self._n_worker is not None else 1
        if self.__class__.allow_batch and n_worker > 1:
            # Distributed inference: disable continuous batching
            self.allow_batch = False

        # Create MLXBatchModel for continuous batching after context length is available
        # Only enable continuous batching for non-distributed inference (single worker)
        if (
            self._batch_model is None
            and hasattr(self, "_model")
            and hasattr(self, "_tokenizer")
            and self.allow_batch  # Check instance-level allow_batch
        ):
            batch_size = self._model_config.get("batch_size", 4)
            self._batch_model = MLXBatchModel(
                model=self._model,
                tokenizer=self._tokenizer,
                batch_size=batch_size,
                max_context_length=self._context_length,
            )

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        dep_check = check_dependency_available("mlx_lm", "mlx_lm")
        if dep_check != True:
            return dep_check
        return True

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["mlx"]:
            return False, "MLX base engine only supports mlx format"
        if sys.platform != "darwin" or platform.processor() != "arm":
            return False, "MLX base engine only works on Apple silicon Macs"
        if "generate" not in llm_family.model_ability:
            return False, "MLX base engine requires generate ability"
        if "chat" in llm_family.model_ability or "vision" in llm_family.model_ability:
            return False, "MLX base engine does not handle chat or vision models"
        return True

    def _get_prompt_cache(
        self, prompt, lora_name: Optional[str] = None, model: Any = None
    ):
        from mlx_lm.models.cache import make_prompt_cache

        assert self._prompt_cache is not None
        cache_len = len(self._prompt_cache.tokens)
        model_key = (self.model_path, lora_name)
        if (
            self._prompt_cache.model_key != model_key
            or cache_len >= len(prompt)
            or self._prompt_cache.tokens != prompt[:cache_len]
        ):
            self._prompt_cache.model_key = model_key
            self._prompt_cache.cache = make_prompt_cache(
                model or self._model, self._max_kv_size
            )
            self._prompt_cache.tokens = []
            logger.debug("Making new prompt cache for %s", self.model_uid)
        else:
            prompt = prompt[cache_len:]
            logger.debug("Cache hit for %s", self.model_uid)
        self._prompt_cache.tokens.extend(prompt)
        return prompt

    def _generate_stream_inner(self, **kwargs):
        try:
            from mlx_lm.utils import (
                make_logits_processors,
                make_sampler,
                stream_generate,
            )
        except ImportError:
            # for mlx-lm >= 0.22.3
            from mlx_lm.generate import stream_generate
            from mlx_lm.sample_utils import make_logits_processors, make_sampler

        sampler = make_sampler(
            temp=kwargs.pop("temperature"), top_p=kwargs.pop("top_p")
        )
        prompt_token_ids = kwargs.pop("prompt_token_ids")
        logits_processors = make_logits_processors(
            logit_bias=kwargs.pop("logits_bias", None),
            repetition_penalty=kwargs.pop("repetition_penalty"),
            repetition_context_size=kwargs.pop("repetition_context_size"),
        )
        try:
            yield from stream_generate(
                self._model,
                self._tokenizer,
                prompt_token_ids,
                sampler=sampler,
                logits_processors=logits_processors,
                **kwargs,
            )
        finally:
            # after completing the inference, clear the memory.
            self._cleanup_memory()

    def _prepare_inputs(
        self, prompt: Union[str, Dict[str, Any]], kwargs
    ) -> Tuple[Any, int]:
        prompt_token_ids = self._tokenizer.encode(prompt)
        prompt_token_ids = self._get_prompt_cache(
            prompt_token_ids, kwargs.get("lora_name")
        )
        return prompt_token_ids, len(prompt_token_ids)

    def _generate_stream(
        self, prompt: Union[str, Dict[str, Any]], kwargs: MLXGenerateConfig
    ):
        model_uid = self.model_uid
        tokenizer = self._tokenizer
        max_tokens = kwargs["max_tokens"]
        chunk_id = str(uuid.uuid4())
        stop_token_ids = kwargs.get("stop_token_ids", [])
        stream = kwargs.get("stream", False)
        stream_options = kwargs.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        prompt_token_ids, input_echo_len = self._prepare_inputs(prompt, kwargs)

        if max_tokens is None:
            # not set max_tokens
            max_tokens = self._context_length - input_echo_len
            logger.debug("No max_tokens set, setting to: %s", max_tokens)

        i = 0
        start = time.time()
        output = ""
        tokens = []
        # For VLM models, let generate_step handle cache creation internally
        # to avoid incompatible cache types between mlx_lm and mlx_vlm
        stream_kwargs = {
            "prompt_token_ids": prompt_token_ids,
            "max_tokens": max_tokens,
            "temperature": kwargs["temperature"],
            "top_p": kwargs["top_p"],
            "repetition_penalty": kwargs["repetition_penalty"],
            "repetition_context_size": kwargs["repetition_context_size"],
            "stop_token_ids": stop_token_ids,
            # Don't pass prompt_cache for VLM models to let mlx_vlm handle it
        }

        for i, chunk_resp in enumerate(self._generate_stream_inner(**stream_kwargs)):
            token = chunk_resp.token
            tokens.append(token)

            out = chunk_resp.text
            if stream:
                # this special character is mainly for qwen
                out = out.strip("�")
                output = out
            else:
                output += out

            completion_usage = CompletionUsage(
                prompt_tokens=input_echo_len,
                completion_tokens=i,
                total_tokens=(input_echo_len + i),
            )

            yield generate_completion_chunk(
                chunk_text=output,
                finish_reason=None,
                chunk_id=chunk_id,
                model_uid=model_uid,
                prompt_tokens=input_echo_len,
                completion_tokens=i,
                total_tokens=(input_echo_len + i),
            ), completion_usage

            if token == tokenizer.eos_token_id or token in stop_token_ids:  # type: ignore
                break

        logger.info(
            f"Average generation speed: {i / (time.time() - start):.2f} tokens/s."
        )

        if self._prompt_cache:
            self._prompt_cache.tokens.extend(tokens)  # type: ignore

        if i == max_tokens - 1:
            finish_reason = "length"
        else:
            finish_reason = "stop"

        completion_usage = CompletionUsage(
            prompt_tokens=input_echo_len,
            completion_tokens=i,
            total_tokens=(input_echo_len + i),
        )
        if stream:
            yield generate_completion_chunk(
                "",
                finish_reason=finish_reason,
                chunk_id=chunk_id,
                model_uid=model_uid,
                prompt_tokens=input_echo_len,
                completion_tokens=i,
                total_tokens=(input_echo_len + i),
            ), completion_usage
        else:
            yield generate_completion_chunk(
                output,
                finish_reason=finish_reason,
                chunk_id=chunk_id,
                model_uid=model_uid,
                prompt_tokens=input_echo_len,
                completion_tokens=i,
                total_tokens=(input_echo_len + i),
            ), completion_usage

        if include_usage:
            completion_chunk = CompletionChunk(
                id=chunk_id,
                object="text_completion",
                created=int(time.time()),
                model=model_uid,
                choices=[],
            )
            yield completion_chunk, completion_usage

    def _run_non_drivers(
        self, method: str, stream: bool, *args, **kwargs
    ) -> Optional[concurrent.futures.Future]:
        assert self._n_worker is not None and self._shard is not None
        if self._n_worker == 1 or self._shard > 0:
            # only run for distributed driver
            return None

        async def run_other_shard(shard: int):
            assert self._rank_to_addresses is not None
            address = self._rank_to_addresses[shard]
            model_actor_ref = await xo.actor_ref(
                address=address, uid=self.raw_model_uid
            )
            # we don't actually need to get the result from shard >= 1
            if stream:
                async for _ in await getattr(model_actor_ref, method)(*args, **kwargs):
                    pass
            else:
                await getattr(model_actor_ref, method)(*args, **kwargs)

        async def run_non_driver_shards():
            logger.debug("Start to run non driver %s", method)
            coros = []
            for rank in range(1, self._n_worker):
                coros.append(run_other_shard(rank))
            await asyncio.gather(*coros)

        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(run_non_driver_shards(), self._loop)

    async def async_generate(
        self,
        prompt: Union[str, Dict[str, Any]],
        generate_config: Optional[MLXGenerateConfig] = None,
        request_id: Optional[str] = None,
    ) -> Union[Completion, AsyncGenerator[CompletionChunk, None]]:
        logger.debug(
            "Enter async_generate, prompt: %s, generate config: %s",
            prompt,
            generate_config,
        )

        generate_config = self._sanitize_generate_config(generate_config)

        assert self._model is not None
        assert self._tokenizer is not None

        stream = generate_config.get("stream", False)

        # If batch model is available (non-distributed inference), use continuous batching
        if self._batch_model is not None:
            # Extract prompt text
            if isinstance(prompt, dict):
                prompt_text = prompt.get("prompt", "")
            else:
                prompt_text = prompt

            # Handle max_tokens - use auto-calculation if not set
            max_tokens = generate_config.get("max_tokens")
            if max_tokens is None:
                # Calculate max_tokens automatically based on context length and prompt length
                prompt_tokens = self._tokenizer.encode(prompt_text)
                input_echo_len = len(prompt_tokens)
                max_tokens = self._context_length - input_echo_len
                logger.debug(
                    f"Auto-calculated max_tokens: {max_tokens} (context_length: {self._context_length}, input_echo_len: {input_echo_len})"
                )

            if stream:
                # Return async generator for streaming
                async def stream_generator():
                    async for chunk in self._batch_model.generate_stream(
                        prompt=prompt_text,
                        max_tokens=max_tokens,
                        temperature=generate_config.get("temperature", 1.0),
                        top_p=generate_config.get("top_p", 1.0),
                        request_id=request_id or str(uuid.uuid4()),
                        skip_special_tokens=generate_config.get(
                            "skip_special_tokens", True
                        ),
                    ):
                        # Set model_uid for each chunk
                        if hasattr(chunk, "get"):
                            chunk["model"] = self.model_uid
                            if chunk.get("choices") and len(chunk["choices"]) > 0:
                                chunk["choices"][0]["index"] = 0
                        yield chunk

                return stream_generator()
            else:
                # Non-streaming: get full result and return completion
                result = await self._batch_model.generate(
                    prompt=prompt_text,
                    max_tokens=max_tokens,
                    temperature=generate_config.get("temperature", 1.0),
                    top_p=generate_config.get("top_p", 1.0),
                    stream=False,
                    skip_special_tokens=generate_config.get(
                        "skip_special_tokens", True
                    ),
                )

                # Return completion object
                completion = Completion(
                    id=str(uuid.uuid4()),
                    object="text_completion",
                    created=int(time.time()),
                    model=self.model_uid,
                    choices=[
                        {
                            "text": str(result),
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    usage=CompletionUsage(
                        prompt_tokens=len(prompt_text),
                        completion_tokens=len(str(result)),
                        total_tokens=len(prompt_text) + len(str(result)),
                    ),
                )
                return completion

        # Fallback to original implementation for distributed inference
        # For distributed inference, use _generate_stream which doesn't use BatchGenerator
        if stream:

            async def stream_generator():
                loop = asyncio.get_event_loop()
                chunk_iterator = await loop.run_in_executor(
                    None, self._generate_stream, prompt, generate_config
                )
                for chunk, usage in chunk_iterator:
                    chunk["model"] = self.model_uid
                    yield chunk

            return stream_generator()
        else:
            # Non-streaming: collect all chunks and return completion
            loop = asyncio.get_event_loop()
            chunk_iterator = await loop.run_in_executor(
                None, self._generate_stream, prompt, generate_config
            )

            text = ""
            finish_reason = None
            usage: Optional[CompletionUsage] = None
            for chunk, chunk_usage in chunk_iterator:
                if chunk.get("choices") and len(chunk["choices"]) > 0:
                    text += chunk["choices"][0].get("text", "")
                    finish_reason = chunk["choices"][0].get("finish_reason")
                usage = chunk_usage

            return Completion(
                id=str(uuid.uuid4()),
                object="text_completion",
                created=int(time.time()),
                model=self.model_uid,
                choices=[
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                usage=(
                    usage
                    if usage is not None
                    else CompletionUsage(
                        prompt_tokens=0, completion_tokens=0, total_tokens=0
                    )
                ),
            )


class MLXChatModel(MLXModel, ChatModelMixin):
    allow_batch: bool = True

    def _sanitize_generate_config(
        self,
        generate_config: Optional[MLXGenerateConfig],
    ) -> MLXGenerateConfig:
        generate_config = super()._sanitize_generate_config(generate_config)
        if (not generate_config.get("stop")) and self.model_family.stop:
            generate_config["stop"] = self.model_family.stop.copy()
        if (
            generate_config.get("stop_token_ids", None) is None
            and self.model_family.stop_token_ids
        ):
            generate_config["stop_token_ids"] = self.model_family.stop_token_ids.copy()

        return generate_config

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["mlx"]:
            return False, "MLX chat engine only supports mlx format"
        if sys.platform != "darwin" or platform.processor() != "arm":
            return False, "MLX chat engine only works on Apple silicon Macs"
        if "chat" not in llm_family.model_ability:
            return False, "MLX chat engine requires chat ability"
        if "vision" in llm_family.model_ability:
            return False, "MLX chat engine does not support vision models"
        return True

    async def async_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[MLXGenerateConfig] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        model_family = self.model_family.model_family or self.model_family.model_name
        tools = generate_config.pop("tools", []) if generate_config else None
        chat_template_kwargs = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, self.reasoning_parser
            )
            or {}
        )
        chat_context_var.set(chat_template_kwargs)
        full_context_kwargs = chat_template_kwargs.copy()
        if tools:
            if (
                model_family in QWEN_TOOL_CALL_FAMILY
                or model_family in DEEPSEEK_TOOL_CALL_FAMILY
            ):
                full_context_kwargs["tools"] = tools
        assert self.model_family.chat_template is not None
        full_prompt = self.get_full_context(
            messages, self.model_family.chat_template, **full_context_kwargs
        )

        generate_config = self._sanitize_generate_config(generate_config)

        stream = generate_config.get("stream", False)
        if stream:
            # Use async_generate for streaming
            chunks = await self.async_generate(full_prompt, generate_config)
            assert hasattr(
                chunks, "__aiter__"
            ), "async_generate should return AsyncGenerator for streaming"
            # Type narrowing: we know chunks is an AsyncGenerator when stream=True
            return self._async_to_chat_completion_chunks(
                chunks,  # type: ignore[arg-type]
                self.reasoning_parser,
                chat_template_kwargs,
            )
        else:
            # Use async_generate for non-streaming
            c = await self.async_generate(full_prompt, generate_config)
            assert not hasattr(
                c, "__aiter__"
            ), "async_generate should return Completion for non-streaming"
            if tools:
                return self._post_process_completion(
                    self.model_family, self.model_uid, c
                )
            return self._to_chat_completion(c, self.reasoning_parser)


class MLXVisionModel(MLXModel, ChatModelMixin):
    allow_batch: bool = False

    @classmethod
    def check_lib(cls) -> Union[bool, Tuple[bool, str]]:
        dep_check = check_dependency_available("mlx_vlm", "mlx_vlm")
        if dep_check != True:
            return dep_check
        return True

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> Union[bool, Tuple[bool, str]]:
        if llm_spec.model_format not in ["mlx"]:
            return False, "MLX vision engine only supports mlx format"
        if sys.platform != "darwin" or platform.processor() != "arm":
            return False, "MLX vision engine only works on Apple silicon Macs"
        if "vision" not in llm_family.model_ability:
            return False, "MLX vision engine requires vision ability"
        return True

    def generate(
        self,
        prompt: Union[str, Dict[str, Any]],
        generate_config: Optional[MLXGenerateConfig] = None,
        from_chat: bool = False,
    ) -> Union[Completion, Iterator[CompletionChunk]]:
        """Generate method for vision models (not using continuous batching)."""
        generate_config = self._sanitize_generate_config(generate_config)

        assert self._model is not None
        assert self._tokenizer is not None

        stream = generate_config.get("stream", False)

        if stream:
            # Return generator for streaming
            return self._generate_stream(prompt, generate_config)
        else:
            # Non-streaming: collect all chunks and return completion
            text = ""
            finish_reason = None
            usage: Optional[CompletionUsage] = None
            for chunk, chunk_usage in self._generate_stream(prompt, generate_config):
                if chunk.get("choices") and len(chunk["choices"]) > 0:
                    text += chunk["choices"][0].get("text", "")
                    finish_reason = chunk["choices"][0].get("finish_reason")
                usage = chunk_usage

            return Completion(
                id=str(uuid.uuid4()),
                object="text_completion",
                created=int(time.time()),
                model=self.model_uid,
                choices=[
                    {
                        "text": text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                usage=(
                    usage
                    if usage is not None
                    else CompletionUsage(
                        prompt_tokens=0, completion_tokens=0, total_tokens=0
                    )
                ),
            )

    def wait_for_load(self):
        """Override parent wait_for_load to skip MLXBatchModel creation."""
        if self._loading_thread:
            self._loading_thread.join()
            if self._loading_error:
                _, err, tb = self._loading_error
                raise err.with_traceback(tb)

    def _load_model(self, **kwargs):
        try:
            from mlx_vlm import load
        except ImportError:
            error_message = "Failed to import module 'mlx_vlm'"
            installation_guide = [
                "Please make sure 'mlx_vlm' is installed. ",
                "You can install it by `pip install mlx_vlm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._prompt_cache = PromptCache()

        return load(self.model_path)

    def load(self):
        from mlx_lm.utils import load_config

        if self._n_worker > 1:
            raise NotImplementedError(
                "Distributed inference is not supported for vision models"
            )

        kwargs = {}
        kwargs["revision"] = self._model_config.get(
            "revision", self.model_spec.model_revision
        )
        kwargs["trust_remote_code"] = self._model_config.get("trust_remote_code")
        kwargs["cache_limit_gb"] = self._model_config.pop("cache_limit_gb", None)

        self._model, self._processor = self._load_model(**kwargs)
        self._tokenizer = self._processor.tokenizer

        # get context length
        config = load_config(Path(self.model_path))
        config.update(self._model_config)
        self._context_length = get_context_length(config)

    def _generate_stream_inner(self, **kwargs):
        import mlx.core as mx

        try:
            from mlx_lm.utils import GenerationResponse
        except ImportError:
            # for mlx-lm >= 0.22.3
            from mlx_lm.generate import GenerationResponse
        from mlx_vlm.generate import generate_step

        inputs = kwargs.pop("prompt_token_ids")
        stop_token_ids = kwargs.pop("stop_token_ids", [])

        extra_kwargs = kwargs.copy()
        input_ids, pixel_values, mask, kwargs = inputs
        kwargs.update(extra_kwargs)

        tokenizer = self._processor.tokenizer
        detokenizer = self._processor.detokenizer

        detokenizer.reset()
        tic = time.perf_counter()
        try:
            for n, (token, logprobs) in enumerate(
                generate_step(input_ids, self._model, pixel_values, mask, **kwargs),
            ):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    prompt_tps = len(input_ids) / prompt_time
                    tic = time.perf_counter()
                if token == tokenizer.eos_token_id or token in stop_token_ids:
                    break
                detokenizer.add_token(token)

                # Yield the last segment if streaming
                yield GenerationResponse(
                    text=detokenizer.last_segment,
                    token=token,
                    logprobs=logprobs,
                    from_draft=False,
                    prompt_tokens=len(input_ids),
                    prompt_tps=prompt_tps,
                    generation_tokens=n + 1,
                    generation_tps=(n + 1) / (time.perf_counter() - tic),
                    peak_memory=mx.metal.get_peak_memory() / 1e9,
                )

            detokenizer.finalize()
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=False,
                prompt_tokens=len(input_ids),
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.metal.get_peak_memory() / 1e9,
            )
        finally:
            # after completing the inference, clear the memory
            self._cleanup_memory()

    def _prepare_inputs(
        self, prompt: Union[str, Dict[str, Any]], kwargs
    ) -> Tuple[Any, int]:
        import mlx.core as mx
        from mlx_vlm import prepare_inputs

        prompt_str = prompt.get("prompt")  # type: ignore
        images = prompt.get("multi_modal_data", {}).get("image")  # type: ignore
        if images and not isinstance(images, list):
            images = [images]
        resize_shape = kwargs.pop("resize_shape", None)
        image_token_index = getattr(self._model.config, "image_token_index", None)

        processor = self._processor
        tokenizer = processor if hasattr(processor, "encode") else processor.tokenizer
        prompt_tokens = mx.array(tokenizer.encode(prompt_str))

        if not images:
            input_ids = prompt_tokens[None, :]
            pixel_values = mask = None
            kwargs = {}
            input_token_len = input_ids.size
        else:
            # Check if processor supports audio to avoid feature_extractor errors
            supports_audio = hasattr(processor, "feature_extractor") or (
                hasattr(processor, "audio_tokenizer")
                and processor.audio_tokenizer is not None
            )

            # For processors that don't support audio (like Qwen3VLProcessor),
            # explicitly set audio=None to prevent mlx-vlm from trying to process audio
            if not supports_audio:
                inputs = prepare_inputs(
                    processor=processor,
                    images=images,
                    audio=None,
                    prompts=prompt_str,
                    image_token_index=image_token_index,
                    resize_shape=resize_shape,
                )
            else:
                inputs = prepare_inputs(
                    processor=processor,
                    images=images,
                    prompts=prompt_str,
                    image_token_index=image_token_index,
                    resize_shape=resize_shape,
                )
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            mask = inputs["attention_mask"]
            kwargs = {
                k: v
                for k, v in inputs.items()
                if k not in ["input_ids", "pixel_values", "attention_mask"]
            }
            input_token_len = int(mask.sum())
        return (input_ids, pixel_values, mask, kwargs), input_token_len

    def chat(
        self,
        messages: List[Dict],
        generate_config: Optional[MLXGenerateConfig] = None,
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        messages = self._transform_messages(messages)  # type: ignore
        tools = generate_config.pop("tools", []) if generate_config else None

        model_family = self.model_family.model_family or self.model_family.model_name

        if "internvl2" not in model_family.lower():
            from qwen_vl_utils import process_vision_info

            chat_template_kwargs = (
                self._get_chat_template_kwargs_from_generate_config(
                    generate_config, self.reasoning_parser
                )
                or {}
            )
            chat_context_var.set(chat_template_kwargs)
            full_context_kwargs = chat_template_kwargs.copy()
            if tools and model_family in QWEN_TOOL_CALL_FAMILY:
                full_context_kwargs["tools"] = tools
            assert self.model_family.chat_template is not None
            prompt = self.get_full_context(
                messages, self.model_family.chat_template, **full_context_kwargs
            )
            images, video_inputs = process_vision_info(messages)
            if video_inputs:
                raise ValueError("Not support video input now.")
        else:
            prompt, images = self.get_specific_prompt(model_family, messages)  # type: ignore

        if not images:
            inputs = {
                "prompt": prompt,
            }
        elif len(images) == 1:
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images[-1]},  # type: ignore
            }
        else:
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images},  # type: ignore
            }
        generate_config = self._sanitize_generate_config(generate_config)

        stream = generate_config.get("stream", False)
        if stream:
            it = self.generate(inputs, generate_config)
            assert isinstance(it, Iterator)
            return self._to_chat_completion_chunks(it)
        else:
            c = self.generate(inputs, generate_config)
            assert not isinstance(c, Iterator)
            if tools:
                return self._post_process_completion(
                    self.model_family, self.model_uid, c
                )
            return self._to_chat_completion(c)
