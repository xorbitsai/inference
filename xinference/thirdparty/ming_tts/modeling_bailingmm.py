#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel, Qwen2ForCausalLM, Qwen2Config
from loguru import logger
from . import prompt_contract
from .configuration_bailingmm import BailingMMConfig
from .fm.dit import Aggregator
from .fm.flowloss import FlowLoss
from .audio_tokenizer.modeling_audio_vae import AudioVAE

# The reference runtime decodes with a 512-position static KV cache, so an
# utterance can run for at most (512 - prompt length) steps.
MAX_CACHE_LEN = 512


_CONFIG_FOR_DOC = "BailingMMConfig"


class BailingMMNativeForConditionalGeneration(PreTrainedModel):
    config_class = BailingMMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _tied_weights_keys = ["model.lm_head.weight"]

    def __init__(
        self,
        config: BailingMMConfig,
    ):
        super().__init__(config)
        self.config: BailingMMConfig = config

        self.llm_dytpe = torch.bfloat16
        self.model_type = config.model_type

        # Ming-3.0-tts-0.5b builds on Qwen2.5-0.5B. The MoE sibling,
        # Ming-3.0-tts-tiny, is a separate architecture on its own tokenizer
        # and lives in ``ming_tiny/``.
        self.model = Qwen2ForCausalLM(Qwen2Config.from_dict(self.config.llm_config))

        # Audio tokenizer
        self.audio = AudioVAE(self.config.audio_tokenizer_config)

        self.latent_dim = self.config.audio_tokenizer_config.enc_kwargs['latent_dim']
        self.linear_proj_audio = Aggregator(
            in_channels=self.latent_dim,
            llm_input_dim=self.model.config.hidden_size,
            **self.config.aggregator_config,
        )

        self.patch_size = self.config.ditar_config['patch_size']
        self.history_patch_size = self.config.ditar_config.get('history_patch_size', self.patch_size)
        self.flowloss = FlowLoss(
            z_channels=self.latent_dim,
            llm_cond_dim=self.model.config.hidden_size,
            **self.config.ditar_config
        )
        self.stop_head = nn.Linear(self.model.config.hidden_size, 2, bias=True)
        self.spk_head = nn.Linear(192, self.model.config.hidden_size, bias=True)
        self._tokenizer = None
        self.post_init()

    # Weight tying runs against the outermost model, so these have to be
    # delegated or ``_tied_weights_keys`` above is inert. The inner LLM ties
    # its own head during construction, but ``from_pretrained`` then replaces
    # the embedding table with the one from the checkpoint, which breaks that
    # tie; the retie at the end of loading is what puts it back. Without these
    # methods the retie silently does nothing and ``model.lm_head.weight``
    # stays an uninitialised meta tensor, which only surfaces later as
    # "Cannot copy out of meta tensor" on the first ``.to(device)``.

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.model.set_output_embeddings(new_embeddings)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            # Not AttributeError: nn.Module.__getattr__ would swallow the
            # message and report a bare missing attribute instead.
            raise RuntimeError(
                "tokenizer is not set; assign the tokenizer shipped with the "
                "checkpoint before generating"
            )
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        # A tokenizer that splits <audio> or <|vision_start|> still produces
        # audio, just conditioned on a sequence the model never saw during
        # training. Reject it here instead of debugging the output later.
        prompt_contract.validate_tokenizer(value, self.model_type)
        self._tokenizer = value

    def get_rope_index(
        self,
        input_ids,
        image_token_id,
        video_token_id,
        image_start_token_id,
        video_start_token_id,
        image_grid_thw,
        video_grid_thw,
        attention_mask,
        spatial_merge_size=2,
        tokens_per_second=2,
        second_per_grid_ts=None,
        inputs_embeds=None
    ):
        use_abs_time_pos = second_per_grid_ts is not None

        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None:
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                if image_grid_thw is not None:
                    vision_start_indices = torch.argwhere(
                        input_ids == image_start_token_id
                    ).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum()
                if video_grid_thw is not None:
                    vision_start_indices = torch.argwhere(
                        input_ids == video_start_token_id
                    ).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    video_nums = (vision_tokens == video_token_id).sum()

                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)
                    if use_abs_time_pos:
                        time_tensor = expanded_range * second_per_grid_t * tokens_per_second
                        time_tensor_long = time_tensor.long()
                    else:
                        time_tensor_long = expanded_range.long()
                    t_index = time_tensor_long.flatten()

                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
        else:
            device = inputs_embeds.device if inputs_embeds is not None else input_ids.device
            length = inputs_embeds.size(1) if inputs_embeds is not None else input_ids.size(1)
            bsz = inputs_embeds.size(0) if inputs_embeds is not None else input_ids.size(0)
            dtype = inputs_embeds.dtype if inputs_embeds is not None else input_ids.dtype

            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(length, device=device)
                    .view(1, 1, -1)
                    .expand(3, bsz, -1)
                )
                mrope_position_deltas = torch.zeros(
                    [bsz, 1],
                    device=device,
                    dtype=dtype,
                )

        return position_ids, mrope_position_deltas

    def prepare_inputs_for_generation(self):
        # An empty function to be compatible with the PEFT library for LoRA training. This function is not used in practice.
        pass

    def prepare_input_embed(self, prompt, text, spk_emb=None, instruction=None, prompt_latent=None, prompt_text=None, use_zero_spk_emb=False):
        """Build the input sequence and splice in the continuous features.

        The token layout comes from ``prompt_contract`` so that it stays in
        one place; this method only owns projecting the speaker embeddings
        and prompt latents into the rows the contract reserved for them.
        """
        # Project speaker embeddings from the 192-dim extractor space.
        projected_spk_emb = []
        if spk_emb is not None:
            for se in spk_emb:
                se = self.spk_head(se.to(self.device))
                if use_zero_spk_emb:
                    se = torch.zeros_like(se)
                projected_spk_emb.append(se)

        # Project prompt latents patch-by-patch.
        projected_latent = None
        if prompt_latent is not None and prompt_text is not None:
            bsz = prompt_latent.size(0)
            projected_latent = prompt_latent.reshape(-1, self.patch_size, self.latent_dim)
            projected_latent = self.linear_proj_audio(projected_latent)
            projected_latent = projected_latent.reshape(bsz, -1, projected_latent.size(-1))

        input_part, spk_positions, latent_position = prompt_contract.build_input_ids(
            self.tokenizer,
            self.model_type,
            prompt,
            text,
            num_speakers=len(projected_spk_emb),
            instruction=instruction,
            prompt_text=prompt_text,
            prompt_latent_steps=0 if projected_latent is None else projected_latent.size(1),
        )

        input_ids = torch.tensor(input_part, dtype=torch.long).unsqueeze(0).to(self.device)
        inputs_embeds = self.model.get_input_embeddings()(input_ids).to(self.device)

        # NOTE: This implementation currently assumes a batch size of 1.
        for position, se in zip(spk_positions, projected_spk_emb):
            inputs_embeds[0, position] = se.to(inputs_embeds.dtype)

        if projected_latent is not None:
            end = latent_position + projected_latent.size(1)
            inputs_embeds[0, latent_position:end, :] = projected_latent[0].to(inputs_embeds.dtype)

        return input_ids, inputs_embeds

    def sample_text(
        self,
        prompt,
        text,
        max_decode_steps=200,
    ):
        assert self.model_type == 'dense', "This functionality currently is not supported for MoE model"
        input_ids, inputs_embeds = self.prepare_input_embed(
            prompt=prompt,
            text=text,
        )
        input_ids, inputs_embeds = input_ids[:,:-1], inputs_embeds[:,:-1,...]
        logger.info(self.tokenizer.decode(input_ids[0].cpu().numpy().tolist()).__repr__())
        attention_mask = torch.ones(input_ids.shape).to(input_ids.device)
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1)
        self.rope_deltas = None
        past_key_values = None

        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=200,
            eos_token_id=self.tokenizer.encode("<text_eos>")
        )

        stop_id = self.tokenizer.encode("<text_eos>")
        stop_index = [i for i, token in enumerate(generated_ids[0].tolist()) if token == stop_id[0]]
        generated_ids = generated_ids[:, 1:stop_index[0]]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        yield response, True

    def sample(
        self,
        prompt,
        text,
        spk_emb=None,
        instruction=None,
        prompt_waveform=None,
        prompt_text=None,
        max_decode_steps=1000,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
        use_zero_spk_emb=False,
        prompt_latent=None,
    ):
        # Prepare inputs
        if prompt_text is None:
            prompt_latent = None
        elif prompt_latent is None and prompt_waveform is not None:
            prompt_latent = self.encode_prompt_latent(prompt_waveform)

        input_ids, inputs_embeds = self.prepare_input_embed(
            prompt=prompt,
            text=text,
            spk_emb=spk_emb,
            instruction=instruction,
            prompt_latent=prompt_latent,
            prompt_text=prompt_text,
            use_zero_spk_emb=use_zero_spk_emb
        )
        logger.info(self.tokenizer.decode(input_ids[0].cpu().numpy().tolist()).__repr__())
        attention_mask = torch.ones(input_ids.shape).to(input_ids.device)

        # Obtain position_ids
        if self.model_type == 'dense':
            position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1)
        else:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_token_id=self.config.llm_config.image_patch_token,
                video_token_id=self.config.llm_config.image_patch_token,
                image_start_token_id=self.config.llm_config.image_start_token,
                video_start_token_id=self.config.llm_config.video_start_token,
                image_grid_thw=None,
                video_grid_thw=None,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas

        past_key_values = None
        # Each inference step combines Autoregressive (AR) decoding with Flow Matching
        decode_budget = (MAX_CACHE_LEN - inputs_embeds.shape[1]) // self.patch_size
        for step in tqdm(range(min(max_decode_steps, decode_budget))):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    audio_mask=None,
                    image_mask=None,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values
                )
            past_key_values = outputs.past_key_values
            z_diff = outputs.hidden_states[-1][:, -1:, :]

            # Initialize the latent_history for the first step
            if step == 0:
                latent_history = torch.zeros(
                    1, self.history_patch_size, self.latent_dim, device=z_diff.device, dtype=z_diff.dtype
                )
                if prompt_latent is not None:
                    prompt_latent = prompt_latent.to(z_diff.dtype)
                    start_index = self.history_patch_size - prompt_latent.size(1)
                    if start_index < 0:
                        latent_history[:] = prompt_latent[:, -start_index:,:]
                    else:
                        latent_history[:, start_index:,:] = prompt_latent

            # Predict the latent for the current timestep using the Flow Matching head, conditioned on the history and other inputs.
            sampled_token_latent = self.flowloss.sample(
                z_diff, latent_history, cfg, self.patch_size, sigma=sigma, temperature=temperature
            )

            # Check if the generation is complete.
            if self.stop_head(z_diff)[0][0].softmax(dim=-1)[1] > 0.5 and step > 3:
                yield sampled_token_latent, True
                break
            else:
                yield sampled_token_latent, False

            inputs_embeds = self.linear_proj_audio(sampled_token_latent)

            # Update position_ids, attention_mask, latent_history for next step
            if self.model_type == 'dense':
                position_ids = position_ids[:, -1:] + 1
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                if past_key_values and self.rope_deltas:
                    delta = past_key_values[0][1].shape[2] + self.rope_deltas
                elif past_key_values:
                    delta = torch.tensor(past_key_values[0][1].shape[2]).to(inputs_embeds.device)
                else:
                    delta = torch.tensor(0).to(inputs_embeds.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

            attention_mask = torch.ones(inputs_embeds.shape[0], 1).to(inputs_embeds.device)
            latent_history[:, :-self.patch_size, :] = latent_history[:, self.patch_size:, :].clone()
            latent_history[:, -self.patch_size:, :] = sampled_token_latent[:]

    @torch.inference_mode()
    def encode_prompt_latent(self, prompt_waveform):
        """Encode a ``(1, samples)`` prompt waveform into VAE latents.

        The encoder samples its posterior, so this consumes the global RNG.
        """
        waveform = prompt_waveform.to(device=self.device, dtype=torch.bfloat16)
        length = torch.tensor([waveform.size(1)], dtype=torch.long, device=self.device)
        prompt_latent, _ = self.audio.encode_latent(waveform, length)
        return prompt_latent

    @torch.inference_mode()
    def generate(
        self,
        prompt,
        text,
        spk_emb=None,
        instruction=None,
        prompt_waveform=None,
        prompt_text=None,
        max_decode_steps=1000,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
        use_zero_spk_emb=False,
        prompt_latent=None,
    ):
        sampled_tokens_list = []
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for sampled_tokens, _ in self.sample(
                    prompt=prompt,
                    text=text,
                    spk_emb=spk_emb,
                    instruction=instruction,
                    prompt_waveform=prompt_waveform,
                    prompt_text=prompt_text,
                    max_decode_steps=max_decode_steps,
                    cfg=cfg,
                    sigma=sigma,
                    temperature=temperature,
                    use_zero_spk_emb=use_zero_spk_emb,
                    prompt_latent=prompt_latent,
            ):
                sampled_tokens_list.append(sampled_tokens)

            # One pass over the whole utterance: chunked streaming decode
            # produces slightly different audio at the chunk boundaries.
            sampled_tokens = torch.cat(sampled_tokens_list, dim=1)
            speech = self.audio.decode(
                sampled_tokens, past_key_values=None, use_cache=False, last_chunk=True
            )[0]

        return speech.cpu().float()[0]

    @torch.inference_mode()
    def generate_text(
        self,
        prompt,
        text,
        max_decode_steps=200,
    ):
        sampled_texts_list = []
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for sampled_tokens, last_chunk in self.sample_text(
                    prompt=prompt,
                    text=text,
                    max_decode_steps=max_decode_steps,
            ):
                sampled_texts_list.append(sampled_tokens)

        return "".join(sampled_texts_list)
