import asyncio
from io import BytesIO
from itertools import groupby
from typing import Sequence

from PIL import Image
from tqdm import tqdm

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    SamplingParams,
    VlmClient,
)
from .utils import get_rgb_image, load_resource


class TransformersVlmClient(VlmClient):
    def __init__(
        self,
        model,  # transformers model
        processor,  # transformers processor
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        batch_size: int = 0,
        use_tqdm: bool = True,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )
        if not model:
            raise ValueError("Model is None.")
        if not hasattr(model, "generate"):
            raise ValueError("Model does not have generate method.")
        if not processor:
            raise ValueError("Processor is None.")
        if not hasattr(processor, "apply_chat_template"):
            raise ValueError("Processor does not have apply_chat_template method.")
        self.model = model
        self.processor = processor
        self.model_max_length = model.config.max_position_embeddings

        skip_token_ids: set[int] = set()
        for field in ["bos_token_id", "eos_token_id", "pad_token_id"]:
            if hasattr(model.config, field):
                token_id = getattr(model.config, field)
                if isinstance(token_id, int):
                    skip_token_ids.add(token_id)
            if hasattr(processor.tokenizer, field):
                token_id = getattr(processor.tokenizer, field)
                if isinstance(token_id, int):
                    skip_token_ids.add(token_id)

        self.skip_token_ids = skip_token_ids
        self.batch_size = batch_size
        self.use_tqdm = use_tqdm

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

    def build_generate_kwargs(self, sampling_params: SamplingParams | None):
        sp = self.build_sampling_params(sampling_params)

        do_sample = ((sp.temperature or 0.0) > 0.0) and ((sp.top_k or 1) > 1)

        # write these three params anyway.
        generate_kwargs = {
            "temperature": sp.temperature if do_sample and sp.temperature is not None else None,
            "top_p": sp.top_p if do_sample and sp.top_p is not None else None,
            "top_k": sp.top_k if do_sample and sp.top_k is not None else None,
        }
        if sp.repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = sp.repetition_penalty
        if sp.no_repeat_ngram_size is not None:
            generate_kwargs["no_repeat_ngram_size"] = sp.no_repeat_ngram_size
        if sp.max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = sp.max_new_tokens
        else:  # set max_length when max_new_tokens is not set
            generate_kwargs["max_length"] = self.model_max_length
        generate_kwargs["do_sample"] = do_sample
        return generate_kwargs

    def predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
        **kwargs,
    ) -> str:
        return self.batch_predict(
            [image],  # type: ignore
            [prompt],
            [sampling_params],
            **kwargs,
        )[0]

    def batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        **kwargs,
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
                self.processor.apply_chat_template(
                    self.build_messages(prompts),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            ] * len(images)
        else:  # isinstance(prompts, Sequence[str])
            chat_prompts: list[str] = [
                self.processor.apply_chat_template(
                    self.build_messages(prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]

        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)

        inputs = [
            (args[0].width * args[0].height, idx, *args)
            for (idx, args) in enumerate(zip(image_objs, chat_prompts, sampling_params))
        ]

        outputs: list[str | None] = [None] * len(inputs)
        batch_size = max(1, self.batch_size)

        with tqdm(total=len(inputs), desc="Predict", disable=not self.use_tqdm) as pbar:

            # group inputs by sampling_params, because transformers
            # don't support different params in one batch.
            for params, group_inputs in groupby(inputs, key=lambda item: item[-1]):
                group_inputs = [input[:-1] for input in group_inputs]

                if (batch_size > 1) and (len(group_inputs) > batch_size):
                    group_inputs.sort(key=lambda item: item[0])

                for i in range(0, len(group_inputs), batch_size):
                    batch_inputs = group_inputs[i : i + batch_size]
                    batch_outputs = self._predict_one_batch(
                        image_objs=[item[2] for item in batch_inputs],
                        chat_prompts=[item[3] for item in batch_inputs],
                        sampling_params=params,
                        **kwargs,
                    )
                    for input, output in zip(batch_inputs, batch_outputs):
                        idx = input[1]
                        outputs[idx] = output
                    pbar.update(len(batch_outputs))

        assert all(output is not None for output in outputs)
        return outputs  # type: ignore

    def _predict_one_batch(
        self,
        image_objs: list[Image.Image],
        chat_prompts: list[str],
        sampling_params: SamplingParams | None,
        **kwargs,
    ):
        inputs = self.processor(
            text=chat_prompts,
            images=image_objs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device=self.model.device, dtype=self.model.dtype)

        generate_kwargs = self.build_generate_kwargs(sampling_params)

        output_ids = self.model.generate(
            **inputs,
            use_cache=True,
            **generate_kwargs,
            **kwargs,
        )

        output_ids = output_ids.cpu().tolist()
        output_ids = [ids[len(in_ids) :] for in_ids, ids in zip(inputs.input_ids, output_ids)]
        output_ids = [[id for id in ids if id not in self.skip_token_ids] for ids in output_ids]

        output_texts = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        return output_texts

    async def aio_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self.predict,
            image,
            prompt,
            sampling_params,
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
        return await asyncio.to_thread(
            self.batch_predict,
            images,
            prompts,
            sampling_params,
        )
