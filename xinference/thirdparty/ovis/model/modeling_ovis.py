import logging
import os

from packaging import version
from datetime import datetime
from importlib import import_module
from typing import List, Union, Callable, Optional, Dict

import PIL.Image
import deepspeed
import torch
import transformers
from torch import Tensor
from torch.nn import init
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import HybridCache
from transformers.generation.utils import GenerateOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled, deepspeed_config

from ovis.model.configuration_ovis import OvisConfig
from ovis.model.conversation_formatter import ConversationFormatter
from ovis.util.constants import IGNORE_ID, BEGIN_LINE, END_LINE, IMAGE_ATOM_ID, IMAGE_INDICATOR_IDS, \
    IMAGE_TOKEN_ID
from ovis.util.utils import rank0_print


class VisualEmbedding(torch.nn.Embedding):
    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.long]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)

    def reset_parameters(self, mean=0., std=1.) -> None:
        init.normal_(self.weight, mean=mean, std=std)
        self._fill_padding_idx_with_zero()


class OvisPreTrainedModel(PreTrainedModel):
    config_class = OvisConfig
    base_model_prefix = "ovis"


class Ovis(OvisPreTrainedModel):

    def __init__(self, config: OvisConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if kwargs.get('train_from_scratch'):
            self.llm = kwargs['llm']
            self.generation_config = self.llm.generation_config
            self.config.llm_config = self.llm.config
            self.config.hidden_size = self.llm.config.hidden_size  # for deepspeed auto configuration
            self.text_tokenizer = kwargs['text_tokenizer']
            self.visual_tokenizer = kwargs['visual_tokenizer']
            self.config.visual_tokenizer_config = self.visual_tokenizer.config
        else:
            attn_kwargs = dict()
            if self.config.llm_attn_implementation:
                attn_kwargs['attn_implementation'] = self.config.llm_attn_implementation
            self.llm = AutoModelForCausalLM.from_config(self.config.llm_config, **attn_kwargs)
            assert self.config.hidden_size == self.llm.config.hidden_size, "hidden size mismatch"
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.name_or_path)
            self.visual_tokenizer = AutoModel.from_config(self.config.visual_tokenizer_config,
                                                          image_processor_name_or_path=self.config.name_or_path)

        # initialize vte
        if is_deepspeed_zero3_enabled():
            with deepspeed.zero.Init(config_dict_or_path=deepspeed_config()):
                self.vte = VisualEmbedding(self.config.visual_tokenizer_config.vocab_size, self.config.hidden_size)
        else:
            self.vte = VisualEmbedding(self.config.visual_tokenizer_config.vocab_size, self.config.hidden_size,
                                       device=self.visual_tokenizer.device, dtype=self.visual_tokenizer.dtype)

        def _merge_modules(modules_list: tuple):
            merged_modules = []
            for modules in modules_list:
                merged_modules.extend(modules if modules else [])
            return merged_modules

        self._no_split_modules = _merge_modules((self.llm._no_split_modules, self.visual_tokenizer._no_split_modules))
        self._skip_keys_device_placement = self.llm._skip_keys_device_placement
        self._keep_in_fp32_modules = _merge_modules(
            (self.llm._keep_in_fp32_modules, self.visual_tokenizer._keep_in_fp32_modules))
        self.is_parallelizable = all((self.llm.is_parallelizable, self.visual_tokenizer.is_parallelizable))
        self.supports_gradient_checkpointing = all(
            (self.llm.supports_gradient_checkpointing, self.visual_tokenizer.supports_gradient_checkpointing))
        self._supports_flash_attn_2 = True
        self._supports_sdpa = all((self.llm._supports_sdpa, self.visual_tokenizer._supports_sdpa))

    def get_text_tokenizer(self):
        return self.text_tokenizer

    def get_visual_tokenizer(self):
        return self.visual_tokenizer

    def tie_weights(self):
        if not self.config.disable_tie_weight:
            self.get_llm().tie_weights()

    def re_init_vte(self, mean, std):
        vte = self.get_vte()
        rank0_print(BEGIN_LINE)
        rank0_print(f'[{datetime.now()}] Before re-initialization of vte: ')
        with deepspeed.zero.GatheredParameters([vte.weight]):
            rank0_print(f'vte.weight: {vte.weight}')
        with deepspeed.zero.GatheredParameters([vte.weight], modifier_rank=0):
            if not is_deepspeed_zero3_enabled() or deepspeed.comm.get_rank() == 0:
                vte.reset_parameters(mean, std)
        rank0_print(f'[{datetime.now()}] After re-initialization of vte:')
        with deepspeed.zero.GatheredParameters([vte.weight]):
            rank0_print(f'vte.weight: {vte.weight}')
        rank0_print(END_LINE)

    def get_monitor_tensors(self):
        monitor_tensors = dict(
            wte=self.get_wte().weight,
            lm_head=self.get_lm_head().weight,
            vte=self.get_vte().weight
        )
        monitor_tensors.update(
            {f'visual_tokenizer_{k}': v for k, v in self.get_visual_tokenizer().get_monitor_tensors().items()})
        return monitor_tensors

    def get_lm_head(self):
        return self.get_llm().get_output_embeddings()

    def get_llm(self):
        return self.llm

    def get_vte(self):
        return self.vte

    def get_wte(self):
        return self.llm.get_input_embeddings()

    def get_conversation_formatter(self) -> ConversationFormatter:
        if getattr(self, 'conversation_formatter', None) is None:
            self.conversation_formatter = getattr(import_module(".conversation_formatter", __package__),
                                                  self.config.conversation_formatter_class)(self.text_tokenizer)
        return self.conversation_formatter

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        pixel_values: List[Optional[torch.Tensor]],
        **kwargs
    ):
        # assert self.training, "`forward` can only be used in training. For inference, use `generate`."
        _, inputs_embeds, labels, attention_mask = self.merge_multimodal(
            text_input_ids=input_ids,
            text_attention_masks=attention_mask,
            text_labels=labels,
            pixel_values=pixel_values
        )
        return self.llm(inputs_embeds=inputs_embeds, labels=labels, attention_mask=attention_mask, **kwargs)

    def merge_multimodal(
        self,
        text_input_ids: torch.Tensor,
        text_attention_masks: torch.Tensor,
        text_labels: Optional[torch.Tensor],
        pixel_values: List[Optional[torch.Tensor]],
        left_padding: bool = False
    ):
        input_device = text_input_ids.device
        visual_vocab_szie = self.get_visual_tokenizer().config.vocab_size
        visual_indicator_embeds = self.get_vte()(
            torch.tensor(
                list(range(visual_vocab_szie - 5, visual_vocab_szie)),
                dtype=torch.long,
                device=self.get_visual_tokenizer().device
            )
        ).to(device=input_device)

        if self.training:
            # When training, to be compatible with deepspeed zero, each sample has to include pixel_value tensor.
            # For text-only sample, one can simply use a full zero tensor as pixel_value, which will be ignored
            # (see below in this function); so, the gradient will not be affected.
            num_images = [x.shape[0] for x in pixel_values]
            visual_tokens = self.visual_tokenizer(torch.cat([x for x in pixel_values], dim=0))
            visual_embeds = torch.split(self.get_vte()(visual_tokens).to(dtype=self.dtype, device=input_device),
                                        split_size_or_sections=num_images, dim=0)
            visual_input_ids = torch.split(torch.argmax(visual_tokens, dim=-1).to(device=input_device),
                                           split_size_or_sections=num_images, dim=0)
            visual_labels = [torch.full(x.shape, IGNORE_ID, dtype=torch.long, device=input_device) for x in
                             visual_input_ids]
        else:
            # When inference, sample can include only text with `None` pixel_value
            num_images = [x.shape[0] if x is not None else 0 for x in pixel_values]
            if sum(num_images) > 0:
                visual_tokens = self.visual_tokenizer(torch.cat([x for x in pixel_values if x is not None], dim=0))
                visual_embeds = torch.split(self.get_vte()(visual_tokens).to(dtype=self.dtype, device=input_device),
                                            split_size_or_sections=num_images, dim=0)
                visual_input_ids = torch.split(torch.argmax(visual_tokens, dim=-1).to(device=input_device),
                                               split_size_or_sections=num_images, dim=0)
                visual_labels = [torch.full(x.shape, IGNORE_ID, dtype=torch.long, device=input_device) for x in
                                 visual_input_ids]
            else:
                # just placeholders
                visual_embeds = [None] * len(num_images)
                visual_input_ids = [None] * len(num_images)
                visual_labels = [None] * len(num_images)
        # just placeholders
        if text_labels is None:
            text_labels = torch.full(text_input_ids.shape, IGNORE_ID, dtype=torch.long, device=input_device)

        input_embeds = []
        attention_masks = []
        labels = []
        for text_input_id, text_label, text_attention_mask, visual_embed, visual_input_id, visual_label in zip(
                text_input_ids, text_labels, text_attention_masks, visual_embeds, visual_input_ids, visual_labels
        ):
            placeholder_token_mask = torch.lt(text_input_id, 0)
            text_embed = self.get_wte()(torch.masked_fill(text_input_id, placeholder_token_mask, 0))
            for i, indicator_id in enumerate(IMAGE_INDICATOR_IDS):
                text_embed[text_input_id == indicator_id] = visual_indicator_embeds[i]
            image_atom_positions = torch.where(torch.eq(text_input_id, IMAGE_ATOM_ID))[0].tolist()
            if len(image_atom_positions) > 0:
                input_embed_parts = []
                attention_mask_parts = []
                label_parts = []
                prev_image_atom_position = -1
                for index, image_atom_position in enumerate(image_atom_positions):
                    input_embed_parts.append(
                        text_embed[prev_image_atom_position + 1:image_atom_position, :])
                    label_parts.append(
                        text_label[prev_image_atom_position + 1:image_atom_position])
                    attention_mask_parts.append(
                        text_attention_mask[prev_image_atom_position + 1:image_atom_position])
                    input_embed_parts.append(visual_embed[index])
                    attention_mask_parts.append(
                        torch.ones_like(visual_label[index], dtype=torch.bool))
                    label_parts.append(visual_label[index])
                    prev_image_atom_position = image_atom_position
                if prev_image_atom_position + 1 < text_input_id.shape[0]:
                    input_embed_parts.append(
                        text_embed[prev_image_atom_position + 1:, :])
                    attention_mask_parts.append(
                        text_attention_mask[prev_image_atom_position + 1:])
                    label_parts.append(
                        text_label[prev_image_atom_position + 1:])
                input_embed = torch.cat(input_embed_parts, dim=0)
                attention_mask = torch.cat(attention_mask_parts, dim=0)
                label = torch.cat(label_parts, dim=0)
            else:
                input_embed = text_embed
                attention_mask = text_attention_mask
                label = text_label
                if self.training:
                    # Make visual_embed & visual_indicator_embeds involved in the backward graph,
                    # to be compatible with deepspeed zero and ddp.
                    input_embed += torch.sum(visual_embed * 0.0) + torch.sum(visual_indicator_embeds * 0.0)
            input_embeds.append(input_embed)
            attention_masks.append(attention_mask)
            labels.append(label)

        batch_input_embeds = self.pad_truncate_sequence(input_embeds, batch_first=True, padding_value=0.0, left_padding=left_padding)
        batch_attention_mask = self.pad_truncate_sequence(attention_masks, batch_first=True, padding_value=False, left_padding=left_padding)
        batch_labels = self.pad_truncate_sequence(labels, batch_first=True, padding_value=IGNORE_ID, left_padding=left_padding)

        return visual_input_ids, batch_input_embeds, batch_labels, batch_attention_mask

    def pad_truncate_sequence(self, sequences: List[torch.Tensor], batch_first: bool = True, padding_value: float = 0.0, left_padding: bool = False) -> torch.Tensor:
        if not left_padding:
            pad_sequence = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
            return pad_sequence[:,:self.config.multimodal_max_length]
        else:
            pad_sequence = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in sequences],batch_first=True, padding_value=padding_value).flip(dims=[1])
            return pad_sequence[:,-self.config.multimodal_max_length:]

    def preprocess_inputs(
        self,
        text_or_conversations: Union[List[Dict], str],
        images: Optional[List[PIL.Image.Image]],
        max_partition=9,
        generation_preface='',
        return_labels=False,
        propagate_exception=True,
        frame_selector=None,
        frame_selector_kwargs=None
    ):
        # convert text to conversations
        if isinstance(text_or_conversations, str):
            conversations = [{
                "from": "human",
                "value": text_or_conversations
            }]
        elif isinstance(text_or_conversations, list):
            conversations = text_or_conversations
        else:
            raise ValueError(f'Invalid type of `text_or_conversations`, expected `List[Dict]` or `str`,'
                             f' but got {type(text_or_conversations)}')

        if frame_selector is not None:
            frame_selector_kwargs = frame_selector_kwargs or {}
            conversations, images = frame_selector(conversations=conversations, frames=images, **frame_selector_kwargs)

        # format conversations
        prompt, raw_input_ids, raw_labels = self.get_conversation_formatter().format(
            conversations, generation_preface=generation_preface)

        # place image placeholders
        input_ids = []
        labels = []
        pixel_values = []
        invalidate_label = False
        image_token_indices = [i for i, v in enumerate(raw_input_ids) if v == IMAGE_TOKEN_ID]
        last_image_token_index = -1
        for i in range(len(image_token_indices)):
            head = 0 if i == 0 else image_token_indices[i - 1] + 1
            tail = image_token_indices[i]
            last_image_token_index = tail
            input_ids.extend(raw_input_ids[head:tail])
            labels.extend(raw_labels[head:tail])
            try:
                image = images[i]
                raw_pixel_values, image_placeholders = self.visual_tokenizer.preprocess_image(
                    image, max_partition=max_partition)
            except Exception as e:
                if propagate_exception:
                    raise e
                logging.exception(e)
                invalidate_label = True
                raw_pixel_values, image_placeholders = self.visual_tokenizer.mock_input()
            input_ids.extend(image_placeholders)
            labels.extend([IGNORE_ID] * len(image_placeholders))
            pixel_values.append(raw_pixel_values)
        input_ids.extend(raw_input_ids[last_image_token_index + 1:])
        labels.extend(raw_labels[last_image_token_index + 1:])

        # return tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor([IGNORE_ID] * len(labels) if invalidate_label else labels, dtype=torch.long)
        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None

        if return_labels:
            return prompt, input_ids, pixel_values, labels
        else:
            return prompt, input_ids, pixel_values

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs
    ):
        super().save_pretrained(save_directory,
                                is_main_process=is_main_process,
                                state_dict=state_dict,
                                save_function=save_function,
                                safe_serialization=safe_serialization)
        self.get_text_tokenizer().save_pretrained(save_directory)
        self.get_visual_tokenizer().get_image_processor().save_pretrained(save_directory)

        # uncomment the following will additionally save a separate visual tokenizer
        # visual_tokenizer_directory = os.path.join(save_directory, 'visual_tokenizer')
        # self.get_visual_tokenizer().save_pretrained(visual_tokenizer_directory,
        #                                             is_main_process=is_main_process,
        #                                             state_dict=None,
        #                                             save_function=save_function,
        #                                             safe_serialization=safe_serialization)
        # self.get_visual_tokenizer().get_image_processor().save_pretrained(visual_tokenizer_directory)

    def _get_hybrid_cache_for_llm(self, batch_size: int, max_cache_len: int):
        cache_cls = HybridCache
        llm = self.get_llm()

        if version.parse(transformers.__version__) >= version.parse("4.46.0"):
            need_new_cache = (
                not hasattr(llm, "_cache")
                or (not isinstance(llm._cache, cache_cls))
                or llm._cache.batch_size != batch_size
                or llm._cache.max_cache_len < max_cache_len
            )
        else:
            need_new_cache = (
                not hasattr(llm, "_cache")
                or (not isinstance(llm._cache, cache_cls))
                or llm._cache.max_batch_size != batch_size
                or llm._cache.max_cache_len < max_cache_len
            )

        if need_new_cache:
            if hasattr(llm.config, "_pre_quantization_dtype"):
                cache_dtype = llm.config._pre_quantization_dtype
            else:
                cache_dtype = llm.dtype
            if version.parse(transformers.__version__) >= version.parse("4.46.0"):
                llm._cache = cache_cls(
                    config=llm.config,
                    batch_size=batch_size,
                    max_cache_len=max_cache_len,
                    device=llm.device,
                    dtype=cache_dtype,
                )
            else:
                llm._cache = cache_cls(
                    config=llm.config,
                    max_batch_size=batch_size,
                    max_cache_len=max_cache_len,
                    device=llm.device,
                    dtype=cache_dtype,
                )
        else:
            llm._cache.reset()
        return llm._cache

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Union[GenerateOutput, torch.LongTensor]:
        _, inputs_embeds, labels, attention_mask = self.merge_multimodal(
            text_input_ids=inputs,
            text_attention_masks=kwargs.pop('attention_mask'),
            text_labels=None,
            pixel_values=kwargs.pop('pixel_values'),
            left_padding=True
        )
        inputs_embeds = inputs_embeds.detach()
        torch.cuda.empty_cache()
        if getattr(self.generation_config, 'cache_implementation') == 'hybrid':  # mainly for Gemma2
            kwargs['past_key_values'] = self._get_hybrid_cache_for_llm(
                getattr(kwargs, "num_beams", inputs_embeds.shape[0]), kwargs['max_new_tokens'] + inputs_embeds.shape[-2])
            self.get_llm()._supports_cache_class = True
            kwargs['cache_implementation'] = None

        return self.llm.generate(inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)


AutoConfig.register("ovis", OvisConfig)
AutoModelForCausalLM.register(OvisConfig, Ovis)
