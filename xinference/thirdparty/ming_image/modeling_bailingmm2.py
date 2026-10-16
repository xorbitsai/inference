#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from diffusers.models.normalization import RMSNorm
from transformers import PreTrainedModel
from transformers.utils import logging
from .configuration_bailingmm2 import BailingMM2Config
from .modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM
from .bailingmm_utils import process_ratio, find_first_index_of_consecutive_ones, merge_consecutive_ones
from .inference_profile import load_checkpoint_capabilities, resolve_model_directory
import os
from copy import deepcopy

# vision encoder
from .qwen2_5_vit import Qwen2_5_VisionTransformer, is_flash_attn_2_available

logger = logging.get_logger(__name__)


def _get_attn_implementation() -> str:
    return "flash_attention_2" if is_flash_attn_2_available() else "sdpa"


def _configure_attn_implementation(*configs) -> str:
    attn_implementation = _get_attn_implementation()
    for config in configs:
        if config is not None:
            config._attn_implementation = attn_implementation
    return attn_implementation


_CONFIG_FOR_DOC = "BailingMM2Config"


class BailingMM2NativeForConditionalGeneration(PreTrainedModel):
    config_class = BailingMM2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(
        self,
        config: BailingMM2Config,
        empty_load=False,
    ):
        super().__init__(config)
        self.config: BailingMM2Config = config
        self.vision = None

        self.llm_dytpe = torch.bfloat16

        if empty_load:
            self.model = None
            return

        # Both upstream sub-configs default to FA2. Select one backend for both
        # before constructing them, since the LLM config is instantiated directly.
        _configure_attn_implementation(
            self.config.vision_config, self.config.llm_config
        )
        if self.config.vision_config:
            self.vision = Qwen2_5_VisionTransformer(self.config.vision_config)

        self.model = BailingMoeV2ForCausalLM(self.config.llm_config)

        mlp_modules_img = [nn.Linear(self.vision.image_emb_dim, self.model.config.hidden_size)]
        for _ in range(1, self.config.mlp_depth):
            mlp_modules_img.append(nn.GELU())
            mlp_modules_img.append(nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size))
        self.linear_proj = nn.Sequential(*mlp_modules_img)

        self.post_init()


    def extract_image_feature(self, pixel_values, grid_thw):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            image_embeds = self.vision(pixel_values, grid_thw=grid_thw)
        image_embeds = self.linear_proj(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds


    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        audio_feats: Optional[torch.FloatTensor] = None,
        audio_feats_lengths: Optional[torch.LongTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        num_logits_to_keep: Optional[int] = 0,
        image_gen: Optional[bool] = False,
        image_gen_pixel_values_reference: Optional[torch.FloatTensor] = None,
        image_gen_negative_input_ids: Optional[torch.LongTensor] = None,
        image_gen_negative_attention_mask: Optional[torch.Tensor] = None,
        image_gen_steps: Optional[int] = None,
        image_gen_seed: Optional[int] = None,
        image_gen_cfg: Optional[float] = None,
        image_gen_image_cfg: Optional[float] = 1.0,
        image_gen_cfg_mode: Optional[int] = 1,
        image_gen_height: Optional[int] = None,
        image_gen_width: Optional[int] = None,
        image_gen_llm_hidden_states:  Optional[torch.LongTensor] = None,
        image_gen_negative_llm_hidden_states:  Optional[torch.LongTensor] = None,
        image_gen_text: Optional[list] = None,
        image_gen_highres = 512,
        image_gen_only_extract_hidden_states = False,
        image_gen_condition_embeds=None,
        image_gen_negative_condition_embeds=None,
        image_gen_condition_embeds_2=None,
        image_gen_negative_condition_embeds_2=None,
        image_gen_return_batch=False,
        image_gen_task=None,
        num_frames_per_prompt=1,
        **generate_kwargs,
    ):
        if audio_feats is not None or audio_feats_lengths is not None:
            raise ValueError("audio input is not supported by Ming Image inference")
        if image_gen_image_cfg not in (None, 1.0):
            raise ValueError(
                "image_gen_image_cfg is not supported by this inference path; "
                "guidance is controlled by image_gen_cfg only"
            )
        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None

        if image_gen:
            if not hasattr(self, "inference_profile"):
                raise RuntimeError(
                    "image modules were loaded without a checkpoint inference profile"
                )
            self.inference_profile.validate_task(
                image_gen_task,
                has_reference_image=image_gen_pixel_values_reference is not None,
                num_layers=num_frames_per_prompt,
            )
            sampling = self.inference_profile.resolve_sampling_parameters(
                steps=image_gen_steps,
                cfg=image_gen_cfg,
            )
            image_gen_steps = sampling.steps
            image_gen_cfg = sampling.cfg
            if image_gen_pixel_values_reference is not None:
                input_channels = image_gen_pixel_values_reference.shape[1]
                expected_channels = self.inference_profile.vae_input_channels
                if input_channels % expected_channels != 0:
                    raise ValueError(
                        "reference image channels do not match the checkpoint "
                        f"VAE contract: input={input_channels}, expected a "
                        f"multiple of {expected_channels}"
                    )
            condition_embeds, negative_condition_embeds = None, None
            condition_embeds_2, negative_condition_embeds_2 = None, None
            if (image_gen_condition_embeds is not None) or (image_gen_condition_embeds_2 is not None):
                if image_gen_condition_embeds is not None:
                    condition_embeds = image_gen_condition_embeds
                    negative_condition_embeds = condition_embeds * 0.0 if image_gen_negative_condition_embeds is None else image_gen_negative_condition_embeds

                if image_gen_condition_embeds_2 is not None:
                    condition_embeds_2 = image_gen_condition_embeds_2
                    negative_condition_embeds_2 = condition_embeds_2 * 0.0 if image_gen_negative_condition_embeds_2 is None else image_gen_negative_condition_embeds_2

            else:
                if image_gen_llm_hidden_states is None:
                    assert self.model is not None
                    assert self.vision is not None
                    if pixel_values is not None:
                        image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)

                assert self.loaded_image_gen_modules is True, "please add `load_image_gen=True` in from_pretrained() method"
                assert position_ids is None


                condition_embeds, condition_embeds_2 = self.get_condition_embeds_for_image_gen(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeds=image_embeds,
                    position_ids=position_ids,
                    use_cache=use_cache,
                    image_grid_thw=image_grid_thw,
                    llm_hidden_states=image_gen_llm_hidden_states,
                )
                if condition_embeds is not None:
                    negative_condition_embeds = condition_embeds * 0.0

                if condition_embeds_2 is not None:
                    negative_condition_embeds_2 = condition_embeds_2 * 0.0

                # negative prompt feature is deprecated
                # negative_condition_embeds = self.get_learnable_token_embeds_for_image_gen(
                #     input_ids=image_gen_negative_input_ids,
                #     attention_mask=image_gen_negative_attention_mask,
                #     image_embeds=image_embeds,
                #     position_ids=position_ids,
                #     use_cache=use_cache,
                #     image_grid_thw=image_grid_thw,
                #     llm_hidden_states=image_gen_negative_llm_hidden_states,
                # ) if ((image_gen_negative_input_ids is not None) or (image_gen_negative_llm_hidden_states is not None)) else condition_embeds * 0.0



                if image_gen_only_extract_hidden_states:
                    return condition_embeds, negative_condition_embeds, condition_embeds_2, negative_condition_embeds_2

            assert (condition_embeds is not None) or (condition_embeds_2 is not None)
            if (condition_embeds is not None) and (condition_embeds_2 is not None):
                assert condition_embeds.shape[0] == condition_embeds_2.shape[0]

            bsz = condition_embeds.shape[0] if condition_embeds is not None else condition_embeds_2.shape[0]

            if image_gen_height is None or image_gen_width is None:
                if isinstance(image_gen_highres, int):
                    image_gen_height, image_gen_width = [image_gen_highres] * bsz, [image_gen_highres] * bsz
                elif image_gen_highres is True:
                    image_gen_height, image_gen_width = [1024] * bsz, [1024] * bsz
                else:
                    image_gen_height, image_gen_width = [512] * bsz, [512] * bsz
            elif isinstance(image_gen_height, torch.Tensor) or isinstance(image_gen_width, torch.Tensor):
                assert isinstance(image_gen_height, torch.Tensor), image_gen_height
                assert isinstance(image_gen_width, torch.Tensor), image_gen_width
                image_gen_height = image_gen_height.cpu().tolist()
                image_gen_width = image_gen_width.cpu().tolist()
                assert len(image_gen_height) == bsz
                assert len(image_gen_width)  == bsz
            elif isinstance(image_gen_height, int) or isinstance(image_gen_width, int):
                assert isinstance(image_gen_height, int), image_gen_height
                assert isinstance(image_gen_width, int), image_gen_width
                image_gen_height = [image_gen_height] * bsz
                image_gen_width = [image_gen_width] * bsz
            else:
                assert isinstance(image_gen_height, list), image_gen_height
                assert isinstance(image_gen_width, list), image_gen_width
                assert len(image_gen_height) == bsz
                assert len(image_gen_width)  == bsz


            image_gen_height_diffusion_list = []
            image_gen_width_diffusion_list = []
            image_gen_output_resize_height = []
            image_gen_output_resize_width = []
            for height, width in zip(image_gen_height, image_gen_width):
                closest_size, resize_size = process_ratio(ori_h=height, ori_w=width, highres=image_gen_highres)
                height, width = closest_size
                image_gen_height_diffusion_list.append(height)
                image_gen_width_diffusion_list.append(width)
                height, width = resize_size
                image_gen_output_resize_height.append(height)
                image_gen_output_resize_width.append(width)

            image_gen_height = image_gen_height_diffusion_list[0]
            assert all([i == image_gen_height for i in image_gen_height_diffusion_list])
            image_gen_width = image_gen_width_diffusion_list[0]
            assert all([i == image_gen_width for i in image_gen_width_diffusion_list])

            if image_gen_pixel_values_reference is not None:
                assert (image_gen_height, image_gen_width) == (image_gen_pixel_values_reference.shape[-2], image_gen_pixel_values_reference.shape[-1])

            if image_gen_seed is None or image_gen_seed < 0:
                from datetime import datetime
                image_gen_seed = datetime.now().microsecond % 1000

            sample_kwargs = {
                "steps": image_gen_steps,
                "seed": image_gen_seed,
                "cfg": image_gen_cfg,
                "height": image_gen_height,
                "width": image_gen_width,
                "cfg_mode": image_gen_cfg_mode,
                "ref_x": image_gen_pixel_values_reference,
                "encoder_hidden_states": condition_embeds,
                "directvlm_hidden_states": condition_embeds_2,
                "num_frames_per_prompt": num_frames_per_prompt,
            }

            image = self.diffusion_loss.sample(
                **sample_kwargs,
            )
            if image_gen_task == "layer-decompose":
                output_size = (
                    image_gen_output_resize_width[0],
                    image_gen_output_resize_height[0],
                )
                image = [item.resize(output_size, Image.LANCZOS) for item in image]
            else:
                image = [
                    item.resize((width, height), Image.LANCZOS)
                    for item, width, height in zip(
                        image,
                        image_gen_output_resize_width,
                        image_gen_output_resize_height,
                    )
                ]

            if (
                image_gen_task != "layer-decompose"
                and not image_gen_return_batch
                and len(image) == 1
            ):
                image = image[0]

            return image

        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
        if pixel_values_videos is not None:
            video_embeds = self.extract_image_feature(pixel_values_videos, grid_thw=video_grid_thw)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model.generate(
                input_ids=input_ids,
                query_embeds_image=image_embeds,
                query_embeds_video=video_embeds,
                query_embeds_audio=audio_embeds,
                query_embeds_audio_lengths=audio_embeds_lengths,
                placeholder_audio_loc_lens=audio_placeholder_loc_lens,
                image_grid_thw=image_grid_thw,
                image_grid_thw_video=video_grid_thw,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                num_logits_to_keep=num_logits_to_keep,
                **generate_kwargs,
            )
        return outputs

    def load_image_gen_modules(self, inference_model_path, torch_dtype=torch.float32, load_image_gen_diffusion=True, load_image_gen_others=True, device=None):
        inference_model_path = str(resolve_model_directory(inference_model_path))
        if os.path.exists(os.path.join(inference_model_path, "byt5")):
            raise ValueError(
                "Ming Image inference does not support a byt5 component; "
                "the public checkpoint layout has no byt5/ directory."
            )
        self.inference_profile = load_checkpoint_capabilities(inference_model_path)
        if device is not None:
            device = torch.device(device)
        elif self.model is not None:
            device = self.model.device
        else:
            device = torch.device(torch.cuda.current_device())
        logger.info(f"load_image_gen_modules device={device}")
        from transformers import AutoModelForCausalLM
        from safetensors.torch import load_file
        temp_state_dict = load_file(
            os.path.join(inference_model_path, "mlp", "model.safetensors")
        )
        with open(os.path.join(inference_model_path, 'mlp', 'config.json'), 'r') as f:
            import json
            metax_config = json.load(f)
            diffusion_c_input_dim = metax_config.get("diffusion_c_input_dim", 2048)
            self.img_gen_scales = metax_config.get("img_gen_scales", [4, 8, 16])
            self.connector_norm = metax_config.get("connector_norm", True)
            self.use_vlm_directvlm_condition = metax_config.get(
                "use_vlm_directvlm_condition", False
            )
            self.use_learnable_token_condition = metax_config.get(
                "use_learnable_token_condition", True
            )
            self.selected_hidden_states_layers = metax_config.get(
                "selected_hidden_states_layers"
            )
            self.diffusion_inner_dim = metax_config.get("diffusion_inner_dim")

        if load_image_gen_others:
            self.connector = None
            self.query_tokens_dict = nn.ParameterDict()
            # cumulative token index across the scales
            self.scale_indices = []
            current_idx = 0
            for scale in self.img_gen_scales:
                num_tokens = scale * scale
                scale_name = f"{scale}x{scale}"
                #weights = temp_state_dict[f"query_tokens_dict.{scale_name}"]
                self.query_tokens_dict[scale_name] = nn.Parameter(
                    torch.nn.functional.normalize(torch.randn(num_tokens, self.config.llm_config.hidden_size), dim=-1)
                )
                current_idx += scale * scale
                self.scale_indices.append(current_idx)

            self.query_tokens_dict.to(torch_dtype).to(device)

            if self.use_learnable_token_condition:
                modified_state_dict_query_tokens = {
                    f"{scale}x{scale}": temp_state_dict[f"query_tokens_dict.{scale}x{scale}"]
                    for scale in self.img_gen_scales
                }

                self.query_tokens_dict.load_state_dict(modified_state_dict_query_tokens, strict=True)

                # self.norm_query_embeds = True
                # load connector
                self.connector = AutoModelForCausalLM.from_pretrained(inference_model_path, subfolder='connector', torch_dtype=torch_dtype)
                for layer in self.connector.model.layers:
                    layer.self_attn.is_causal = False
                self.connector.to(device)


                self.proj_in = nn.Linear(self.config.llm_config.hidden_size, self.connector.config.hidden_size)
                self.proj_out = nn.Linear(self.connector.config.hidden_size, diffusion_c_input_dim)

                modified_state_dict_in = {
                    'weight': temp_state_dict['proj_in.weight'],
                    'bias': temp_state_dict['proj_in.bias']
                }
                self.proj_in.load_state_dict(modified_state_dict_in, strict=True)
                modified_state_dict_out = {
                    'weight': temp_state_dict['proj_out.weight'],
                    'bias': temp_state_dict['proj_out.bias']
                }
                self.proj_out.load_state_dict(modified_state_dict_out, strict=True)
                self.proj_in.to(device=device, dtype=torch_dtype)
                self.proj_out.to(device=device, dtype=torch_dtype)

            self.proj_directvlm = None
            if self.use_vlm_directvlm_condition:
                directvlm_dim = self.model.config.hidden_size
                if self.selected_hidden_states_layers is not None:
                    directvlm_dim = directvlm_dim * len(self.selected_hidden_states_layers)

                self.proj_directvlm = nn.Sequential(RMSNorm(directvlm_dim, eps=1e-5), nn.Linear(directvlm_dim, self.diffusion_inner_dim, bias=True))

                modified_state_dict_directvlm = {
                    '0.weight': temp_state_dict["proj_directvlm.0.weight"],
                    '1.weight': temp_state_dict["proj_directvlm.1.weight"],
                    '1.bias': temp_state_dict["proj_directvlm.1.bias"],
                }
                self.proj_directvlm.load_state_dict(modified_state_dict_directvlm, strict=True)
                self.proj_directvlm.to(device=device, dtype=torch_dtype)

        if load_image_gen_diffusion:
            diffusion_mlp_state_dict = {
                key[len("mlp.") :] : temp_state_dict[key]
                for key in temp_state_dict if key.startswith("mlp.")
            }
            from .diffusion.generator import ImageGenerator

            self.diffusion_loss = ImageGenerator(
                model_path=inference_model_path,
                scheduler_path=inference_model_path,
                vision_dim=diffusion_c_input_dim,
                mlp_state_dict=diffusion_mlp_state_dict,
                torch_dtype=torch_dtype,
                device=device,
                use_identity_mlp=metax_config.get("use_identity_mlp", False),
                text_encoder_norm=metax_config.get("text_encoder_norm", False),
                inference_profile=self.inference_profile,
            )
            self.diffusion_loss.to(device)
        self.loaded_image_gen_modules = True
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        load_image_gen = False
        if "load_image_gen" in kwargs:
            load_image_gen = kwargs["load_image_gen"]
            del kwargs["load_image_gen"]
        load_image_gen_diffusion = True
        if "load_image_gen_diffusion" in kwargs:
            load_image_gen_diffusion = kwargs["load_image_gen_diffusion"]
            del kwargs["load_image_gen_diffusion"]

        load_image_gen_others = True
        if "load_image_gen_others" in kwargs:
            load_image_gen_others = kwargs["load_image_gen_others"]
            del kwargs["load_image_gen_others"]
        load_vlm = True
        if "load_vlm" in kwargs:
            load_vlm = kwargs["load_vlm"]
            del kwargs["load_vlm"]
        image_gen_device = kwargs.pop("image_gen_device", None)
        vlm_directory = pretrained_model_name_or_path
        if load_image_gen:
            pretrained_model_name_or_path = str(
                resolve_model_directory(
                    pretrained_model_name_or_path,
                    revision=kwargs.get("revision"),
                    cache_dir=kwargs.get("cache_dir"),
                    local_files_only=kwargs.get("local_files_only", False),
                    token=kwargs.get("token"),
                )
            )
            # The package root keeps connector/mlp/transformer/vae/scheduler;
            # the MLLM itself (config, weights, tokenizer data) lives in mllm/.
            vlm_directory = os.path.join(pretrained_model_name_or_path, "mllm")
            if not os.path.isdir(vlm_directory):
                raise FileNotFoundError(
                    "checkpoint package is missing the mllm/ component: "
                    f"{vlm_directory}. Migrate the package to the component "
                    "layout before loading."
                )
        if load_vlm:
            model = super().from_pretrained(
                vlm_directory,
                *model_args,
                **kwargs,
            )
        else:
            model = cls(
                BailingMM2Config.from_dict(BailingMM2Config.get_config_dict(vlm_directory)[0]),
                empty_load=True,
            )
        if load_image_gen:
            model.load_image_gen_modules(
                pretrained_model_name_or_path,
                torch_dtype=kwargs["torch_dtype"] if "torch_dtype" in kwargs else torch.float32,
                load_image_gen_diffusion=load_image_gen_diffusion,
                load_image_gen_others=load_image_gen_others,
                device=image_gen_device,
            )
        return model

    def append_input_ids_with_multiscale_learnable_tokens(
        self,
        text_ids,
        attention_mask,
        scales,
        start_token_id,
        end_token_id,
        patch_token_id,
    ):
        default_scaled_tokens = []
        default_scaled_attn_masks = []
        default_gen_masks = []
        for scale in scales:
            default_scaled_tokens.append(start_token_id)
            default_scaled_tokens.extend([patch_token_id for _ in range(scale * scale)])
            default_scaled_tokens.append(end_token_id)
            default_scaled_attn_masks.extend([1 for _ in range(scale * scale + 2)])
            default_gen_masks.append(0)
            default_gen_masks.extend([1 for _ in range(scale * scale)])
            default_gen_masks.append(0)

        text_ids_list = text_ids.cpu().tolist()
        attention_mask_list = attention_mask.cpu().tolist()

        new_text_ids_list = []
        new_attention_mask_list = []
        gen_mask_list = []
        new_labels_list = []
        for text_ids_one_batch, attention_mask_one_batch in zip(
            text_ids_list, attention_mask_list
        ):
            assert len(text_ids_one_batch) == len(attention_mask_one_batch)

            padding_start = 0
            for idx, value in enumerate(attention_mask_one_batch):
                if value == 0:
                    break

                padding_start += 1

            new_text_ids_list.append(text_ids_one_batch[:padding_start] + deepcopy(default_scaled_tokens) + text_ids_one_batch[padding_start:])
            new_labels_list.append([ -100 for _ in range(padding_start)] + [1 for _ in range(len(default_scaled_tokens))] + [-100 for _ in range(len(text_ids_one_batch[padding_start:]))] )

            new_attention_mask_list.append(attention_mask_one_batch[:padding_start] + deepcopy(default_scaled_attn_masks) + attention_mask_one_batch[padding_start:])
            gen_mask_list.append(
                [0 for _ in range(len(attention_mask_one_batch[:padding_start]))] + \
                deepcopy(default_gen_masks) + \
                [0 for _ in range(len(attention_mask_one_batch[padding_start:]))]
            )

        text_ids_append_lq = torch.tensor(new_text_ids_list, dtype=text_ids.dtype).to(text_ids.device)
        attention_mask_append_lq = torch.tensor(new_attention_mask_list, dtype=attention_mask.dtype).to(attention_mask.device)
        gen_mask = torch.tensor(gen_mask_list, dtype=attention_mask.dtype).to(attention_mask.device)
        labels = torch.tensor(new_labels_list, dtype=text_ids.dtype).to(text_ids.device)

        assert attention_mask_append_lq.shape == text_ids_append_lq.shape
        assert labels.shape == text_ids_append_lq.shape
        assert gen_mask.shape == text_ids_append_lq.shape
        return text_ids_append_lq, labels, attention_mask_append_lq, gen_mask

    def appand_learnable_tokens(
        self,
        text_ids,
        gen_mask,
        image_embeds,
        image_grid_thw,
        patch_token_id,
    ):
        query_tokens_embeds = torch.cat(
            [self.query_tokens_dict[f"{scale}x{scale}"] for scale in self.img_gen_scales],
            dim=0,
        )
        if image_embeds is not None:
            query_tokens_embeds = query_tokens_embeds.to(image_embeds.dtype).to(image_embeds.device)

        assert text_ids.shape == gen_mask.shape
        text_ids_aslist = text_ids.cpu().view(-1).tolist()
        gen_mask_aslist = gen_mask.cpu().view(-1).tolist()
        is_patch_list = [1 if i == patch_token_id else 0 for i in text_ids_aslist]
        idxes_start_of_patch = find_first_index_of_consecutive_ones(is_patch_list)
        isgen_indicators = merge_consecutive_ones([1 if gen_mask_aslist[i] else 0 for i in idxes_start_of_patch], len(self.img_gen_scales))
        if any([i == 0 for i in isgen_indicators]):
            assert image_grid_thw is not None
            assert image_grid_thw.ndim == 2
            assert image_embeds is not None
            assert image_embeds.ndim == 2

        new_image_grid_thw = []
        new_image_embeds = []
        cum_image_token = 0
        cnt_input_image = 0

        for is_gen in isgen_indicators:
            if is_gen:
                for scale in self.img_gen_scales:
                    new_image_grid_thw.append([1, 2, scale * scale * 2])

                new_image_embeds.append(query_tokens_embeds)
            else:
                thw = image_grid_thw[cnt_input_image].tolist()
                assert thw[0] == 1
                assert thw[1] % 2 == 0 # h
                assert thw[2] % 2 == 0 # w
                n_image_token = (thw[1] // 2) * (thw[2] // 2)
                image_embed_one = image_embeds[cum_image_token : cum_image_token + n_image_token, :]
                new_image_embeds.append(image_embed_one)
                new_image_grid_thw.append(thw)
                cnt_input_image += 1
                cum_image_token += n_image_token

        if image_grid_thw is not None:
            assert cnt_input_image == image_grid_thw.shape[0]
            assert cum_image_token == image_embeds.shape[0]
        else:
            assert cnt_input_image == 0
            assert cum_image_token == 0

        new_image_grid_thw = torch.tensor(new_image_grid_thw, dtype=text_ids.dtype).to(text_ids.device)
        new_image_embeds = torch.cat(new_image_embeds, dim=0).to(text_ids.device)

        total_patch_token = 0
        for bid in range(new_image_grid_thw.shape[0]):
            thw = new_image_grid_thw[bid].tolist()
            assert thw[0] == 1
            assert thw[1] % 2 == 0
            assert thw[2] % 2 == 0
            patch_h = thw[1] // 2
            patch_w = thw[2] // 2
            n_patch_token = patch_h * patch_w
            total_patch_token += n_patch_token

        # if torch.distributed.get_rank() == 0:
        #     embed()
        # torch.distributed.barrier()

        assert total_patch_token == new_image_embeds.shape[0], f"{total_patch_token}, vs. {new_image_embeds.shape}"

        return new_image_grid_thw, new_image_embeds

    def get_condition_embeds_for_image_gen(
        self,
        input_ids,
        attention_mask,
        image_embeds,
        position_ids,
        use_cache,
        image_grid_thw,
        llm_hidden_states,
    ):
        input_ids, labels, attention_mask, gen_mask = self.append_input_ids_with_multiscale_learnable_tokens(
            input_ids,
            attention_mask,
            self.img_gen_scales,
            self.config.llm_config.image_patch_token + 1,
            self.config.llm_config.image_patch_token + 2,
            self.config.llm_config.image_patch_token,
        )

        if llm_hidden_states is None:
            image_grid_thw, image_embeds = self.appand_learnable_tokens(
                input_ids,
                gen_mask,
                image_embeds,
                image_grid_thw,
                self.config.llm_config.image_patch_token,
            )

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if image_embeds is None or input_ids.size(1) == 1:
                    words_embeddings = self.model.get_input_embeddings()(input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1))
                    image_mask = None
                    audio_mask = None
                else:
                    words_embeddings, image_mask, audio_mask = self.model.model.prompt_wrap_navit(
                        input_ids=input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1),
                        config=self.model.model.config,
                        query_embeds_image=image_embeds,
                    )

                assert input_ids.size(1) == words_embeddings.size(1), "{} vs {}".format(
                    input_ids.size,
                    words_embeddings.size,
                )

                # if torch.distributed.get_rank() == 3:
                #     embed()
                # torch.distributed.barrier()

                outputs = self.model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=words_embeddings,
                    image_grid_thw=image_grid_thw,
                    use_cache=False,
                    image_mask=image_mask,
                    audio_mask=None,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = llm_hidden_states

        directvlm_hidden_states = None
        if self.use_vlm_directvlm_condition:
            # use hidden states
            use_input_mask = torch.lt(labels, 0).int().to(attention_mask.dtype) * attention_mask
            assert use_input_mask.ndim == 2
            directvlm_max_valid_ind = use_input_mask.cumsum(-1).argmax(-1).max().item() + 1
            #directvlm_max_valid_ind = min(directvlm_max_valid_ind, self.max_vlm_directvlm_length)
            use_input_mask = use_input_mask[:, :directvlm_max_valid_ind]

            if self.selected_hidden_states_layers is not None:
                directvlm_hidden_states = torch.cat([
                    outputs.hidden_states[layer_i].to(labels.device)[:, :directvlm_max_valid_ind, :] * use_input_mask.unsqueeze(-1)
                    for layer_i in self.selected_hidden_states_layers
                ], dim=-1)
            else:
                directvlm_hidden_states = outputs.hidden_states[-1].to(labels.device)[:, :directvlm_max_valid_ind, :] * use_input_mask.unsqueeze(-1)

            directvlm_hidden_states = directvlm_hidden_states.detach()
            directvlm_hidden_states = self.proj_directvlm(directvlm_hidden_states)

        scale_embeds = None
        if self.use_learnable_token_condition:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                gen_mask = gen_mask.unsqueeze(-1).expand(gen_mask.shape[0], gen_mask.shape[1], hidden_states.shape[-1]).to(hidden_states.device).bool()
                hidden_states_gen = torch.masked_select(hidden_states, gen_mask).view(hidden_states.shape[0], -1, hidden_states.shape[-1])
                # split hidden_states into per-scale representations
                scale_start_idxes = [0] + self.scale_indices[:-1]
                scale_end_idxes = self.scale_indices
                assert scale_end_idxes[-1] == hidden_states_gen.shape[1]

                scale, scale_start_idx, scale_end_idx = [
                    i for i in zip(self.img_gen_scales, scale_start_idxes, scale_end_idxes)
                ][-1]

                scale_hidden = hidden_states_gen[:, scale_start_idx : scale_end_idx, :]
                scale_embeds = self.proj_in(scale_hidden)
                seq_shape = scale_embeds.shape
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    scale_embeds = self.connector(
                        inputs_embeds=scale_embeds,
                        attention_mask=torch.ones(seq_shape[0],1,seq_shape[1],seq_shape[1]).to(scale_embeds.device),
                        output_hidden_states=True
                    ).hidden_states[-1]

                scale_embeds = self.proj_out(scale_embeds)
                # normalize
                if self.connector_norm:
                    scale_embeds = torch.nn.functional.normalize(scale_embeds, dim=-1)

        return scale_embeds, directvlm_hidden_states

__all__ = [
    "BailingMM2NativeForConditionalGeneration"
]
