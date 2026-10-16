import asyncio
from io import BytesIO
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams as VllmSamplingParams

from PIL import Image

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    RequestError,
    SamplingParams,
    ServerError,
    UnsupportedError,
    VlmClient,
)
from .utils import get_rgb_image, load_resource


class VllmEngineVlmClient(VlmClient):
    def __init__(
        self,
        vllm_llm,  # vllm.LLM instance
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        batch_size: int = 0,
        use_tqdm: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Please install vllm to use VllmEngineVlmClient.")

        if not vllm_llm:
            raise ValueError("vllm_llm is None.")
        if not isinstance(vllm_llm, LLM):
            raise ValueError("vllm_llm must be an instance of vllm.LLM.")

        self.vllm_llm = vllm_llm
        self.tokenizer = vllm_llm.get_tokenizer()
        self.model_max_length = vllm_llm.llm_engine.model_config.max_model_len
        self.VllmSamplingParams = SamplingParams
        self.batch_size = batch_size
        self.use_tqdm = use_tqdm
        self.debug = debug

    def build_messages(self, prompt: str) -> list[dict]:
        prompt = prompt or self.prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if "<image>" in prompt:
            prompt_1, prompt_2 = prompt.split("<image>", 1)
            user_messages = [
                *([{"type": "text", "text": prompt_1}] if prompt_1.strip() else []),
                {"type": "image"},
                *([{"type": "text", "text": prompt_2}] if prompt_2.strip() else []),
            ]
        elif self.text_before_image:
            user_messages = [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ]
        else:  # image before text, which is the default behavior.
            user_messages = [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": user_messages})
        return messages

    def build_vllm_sampling_params(self, sampling_params: SamplingParams | None):
        sp = self.build_sampling_params(sampling_params)

        vllm_sp_dict = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "presence_penalty": sp.presence_penalty,
            "frequency_penalty": sp.frequency_penalty,
            "repetition_penalty": sp.repetition_penalty,
            # max_tokens should smaller than model max length
            "max_tokens": sp.max_new_tokens if sp.max_new_tokens is not None else self.model_max_length,
        }

        if sp.no_repeat_ngram_size is not None:
            vllm_sp_dict["extra_args"] = {
                "no_repeat_ngram_size": sp.no_repeat_ngram_size,
                "debug": self.debug,
            }

        return self.VllmSamplingParams(
            **{k: v for k, v in vllm_sp_dict.items() if v is not None},
            skip_special_tokens=False,
        )

    def get_output_content(self, output: "RequestOutput") -> str:
        if not output.finished:
            raise ServerError("The output generation was not finished.")

        choices = output.outputs
        if not (isinstance(choices, list) and choices):
            raise ServerError("No choices found in the output.")

        finish_reason = choices[0].finish_reason
        if finish_reason is None:
            raise ServerError("Finish reason is None in the output.")
        if finish_reason == "length":
            if not self.allow_truncated_content:
                raise RequestError("The output was truncated due to length limit.")
            else:
                print("Warning: The output was truncated due to length limit.")
        elif finish_reason != "stop":
            raise RequestError(f"Unexpected finish reason: {finish_reason}")

        return choices[0].text

    def predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        return self.batch_predict(
            [image],  # type: ignore
            [prompt],
            [sampling_params],
        )[0]

    def batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        if not isinstance(prompts, str):
            assert len(prompts) == len(images), "Length of prompts and images must match."
        if isinstance(sampling_params, Sequence):
            assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        if isinstance(priority, Sequence):
            assert len(priority) == len(images), "Length of priority and images must match."

        image_objs: list[Image.Image] = []
        for image in images:
            if isinstance(image, str):
                image = load_resource(image)
            if not isinstance(image, Image.Image):
                image = Image.open(BytesIO(image))
            image = get_rgb_image(image)
            image_objs.append(image)

        if isinstance(prompts, str):
            chat_prompts: list[str] = [
                self.tokenizer.apply_chat_template(
                    self.build_messages(prompts),  # type: ignore
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ] * len(images)
        else:  # isinstance(prompts, Sequence[str])
            chat_prompts: list[str] = [
                self.tokenizer.apply_chat_template(
                    self.build_messages(prompt),  # type: ignore
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]

        if not isinstance(sampling_params, Sequence):
            vllm_sp_list = [self.build_vllm_sampling_params(sampling_params)] * len(images)
        else:
            vllm_sp_list = [self.build_vllm_sampling_params(sp) for sp in sampling_params]

        outputs = []
        batch_size = self.batch_size if self.batch_size > 0 else len(images)
        batch_size = max(1, batch_size)

        for i in range(0, len(images), batch_size):
            batch_image_objs = image_objs[i : i + batch_size]
            batch_chat_prompts = chat_prompts[i : i + batch_size]
            batch_sp_list = vllm_sp_list[i : i + batch_size]
            batch_outputs = self._predict_one_batch(
                batch_image_objs,
                batch_chat_prompts,
                batch_sp_list,
            )
            outputs.extend(batch_outputs)

        return outputs

    def _predict_one_batch(
        self,
        image_objs: list[Image.Image],
        chat_prompts: list[str],
        vllm_sampling_params: list["VllmSamplingParams"],
    ):
        vllm_prompts = [
            {"prompt": chat_prompt, "multi_modal_data": {"image": image}}
            for chat_prompt, image in zip(chat_prompts, image_objs)
        ]

        outputs = self.vllm_llm.generate(
            prompts=vllm_prompts,  # type: ignore
            sampling_params=vllm_sampling_params,
            use_tqdm=self.use_tqdm,
        )

        return [self.get_output_content(output) for output in outputs]

    async def aio_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        raise UnsupportedError(
            "Asynchronous aio_predict() is not supported in vllm-engine VlmClient(backend). "
            "Please use predict() instead. If you intend to use asynchronous client, "
            "please use vllm-async-engine VlmClient(backend)."
        )

    async def aio_batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm=False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        raise UnsupportedError(
            "Asynchronous aio_batch_predict() is not supported in vllm-engine VlmClient(backend). "
            "Please use batch_predict() instead. If you intend to use asynchronous client, "
            "please use vllm-async-engine VlmClient(backend)."
        )
