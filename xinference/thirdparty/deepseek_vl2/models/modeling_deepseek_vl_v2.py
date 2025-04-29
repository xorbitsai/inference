from attrdict import AttrDict
from dataclasses import dataclass
import logging
import gc

from einops import rearrange, repeat
from typing import Optional, List, Tuple, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import ModelOutput
from transformers.configuration_utils import PretrainedConfig
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedModel
)
from transformers.utils import logging

from .siglip_vit import VisionTransformer
from .configuration_deepseek import DeepseekV2Config
from .modeling_deepseek import DeepseekV2ForCausalLM


logger = logging.get_logger(__name__)


class MlpProjector(nn.Module):

    def __init__(self, cfg):

        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":
            mlp_depth = cfg.depth
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == "downsample_mlp_gelu":
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio, cfg.n_embed * mlp_ratio)]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim)

        self.layers = modules

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh ** 0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            # import ipdb; ipdb.set_trace()
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == 'downsample_mlp_gelu':
            bs, hw, input_dim = x.shape
            h = w = int((hw) ** 0.5)

            """compute padding"""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), "constant", 0)

            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio,
                         padding=0)  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)


class VisionEncoderConfig(PretrainedConfig):
    model_type: str = "vision"

    model_name: str = "siglip_large_patch16_384"
    image_size: int = 384
    patch_size: int = 16
    width: int = 1024
    layers: int = 24
    heads: int = 16
    mlp_ratio: int = 4
    global_pool: str = "map"
    ignore_head: bool = True
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False
    weight_init: str = "skip"
    deterministic: bool = False
    num_recomputing_layers: int = 0

    def __init__(
            self,
            model_name: str = "siglip_large_patch16_384",
            image_size: int = 384,
            patch_size: int = 16,
            width: int = 1024,
            layers: int = 24,
            heads: int = 16,
            mlp_ratio: int = 4,
            global_pool: str = "map",
            ignore_head: bool = True,
            class_token: bool = False,
            num_classes: int = 0,
            use_checkpoint: bool = False,
            **kwargs
    ):
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.global_pool = global_pool
        self.ignore_head = ignore_head
        self.class_token = class_token
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        super().__init__(**kwargs)


class MlpProjectorConfig(PretrainedConfig):
    model_type = "mlp_projector"
    projector_type: str = "downsample_mlp_gelu"
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False

    def __init__(
            self,
            projector_type: str = "downsample_mlp_gelu",
            input_dim: int = 1152,
            n_embed: int = 2048,
            depth: int = 2,
            mlp_ratio: int = 1,
            downsample_ratio: int = 2,
            **kwargs
    ):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio

        super().__init__(**kwargs)


@dataclass
class DeepSeekVLV2CausalLMOutputWithPast(ModelOutput):
    """
    Base class for DeepSeek-VL2 causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class DeepseekVLV2Config(PretrainedConfig):
    model_type = "deepseek_vl_v2"
    vision_config: VisionEncoderConfig
    projector_config: MlpProjectorConfig
    language_config: DeepseekV2Config

    tile_tag: str = "2D"
    global_view_pos: str = "head"
    candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384),)

    def __init__(
            self,
            tile_tag: str = "tile_tag",
            global_view_pos: str = "head",
            candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384),),
            **kwargs
    ):
        super().__init__(**kwargs)

        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionEncoderConfig(**vision_config)

        projector_config = kwargs.get("projector_config", {})
        self.projector_config = MlpProjectorConfig(**projector_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, DeepseekV2Config):
            self.language_config = language_config
        else:
            self.language_config = DeepseekV2Config(**language_config)

        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions


class DeepseekVLV2PreTrainedModel(PreTrainedModel):
    config_class = DeepseekVLV2Config
    base_model_prefix = "deepseek_vl_v2"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class DeepseekVLV2ForCausalLM(DeepseekVLV2PreTrainedModel):

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__(config)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # ----------- vision encoder ------------
        vision_config = config.vision_config
        self.vision = VisionTransformer(
            img_size=vision_config.image_size,
            patch_size=vision_config.patch_size,
            embed_dim=vision_config.width,
            depth=vision_config.layers,
            num_heads=vision_config.heads,
            mlp_ratio=vision_config.mlp_ratio,
            class_token=vision_config.class_token,
            global_pool=vision_config.global_pool,
            ignore_head=vision_config.ignore_head,
            weight_init=vision_config.weight_init,
            num_classes=0,
            deterministic=vision_config.deterministic,
            num_recomputing_layers=vision_config.num_recomputing_layers
        )

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = MlpProjector(projector_config)

        # image token format 形式
        # FIXME 目前tile tag & global_view_pos的默认取值都是之前的实验策略；后续应当去掉默认取值，改为没有取值就raise error
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # 用于format image token sequence的特殊token
        embed_std = 1 / torch.sqrt(torch.tensor(projector_config.n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
            # fix the typo: view_seperater
            self.view_seperator = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
        elif self.tile_tag == "1D":
            # <|tile_x|>, <|tile_global|>
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f"len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}")
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, config.aligner.params.n_embed)) * embed_std
            )
        else:
            raise ValueError(f"tile tag should be either 1D or 2D, but got {self.tile_tag}")

        # ----------- language model ------------
        language_config = config.language_config
        self.language = DeepseekV2ForCausalLM(language_config)

    def prepare_inputs_embeds(
            self,
            input_ids: torch.LongTensor,
            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,
            **ignore_kwargs
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            images (torch.FloatTensor): [b, max_n_images, 3, height, width]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_spatial_crop (torch.LongTensor): [b, max_n_images, 2]

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        if images is None or images_spatial_crop.sum() == 0:
            return self.language.get_input_embeddings()(input_ids)

        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += (1 + num_width_tiles * num_height_tiles)

            total_tiles.append(images[idx, :batch_num_tiles[idx]])

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)
        if total_tiles.shape[0] == 0:
            return self.language.get_input_embeddings()(input_ids)

        # [batch_all_tiles, vit_seq_len, c]
        images_feature = self.vision(total_tiles)

        # [batch_all_tiles, hw, D]
        images_embeds = self.projector(images_feature)
        _, hw, n_dim = images_embeds.shape
        h = w = int(hw ** 0.5)

        # put image tokens into the input_embeds, [b, T, D]
        input_embeds = self.language.get_input_embeddings()(input_ids)

        # 根据self.tile_tag & self.global_view_pos填充image token sequence
        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):

                # extra global & local features
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embeds[tile_index]

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[tile_index + 1: tile_index + 1 + num_tiles_in_image]

                tile_index += num_tiles_in_image + 1

                # format global and local features
                if self.tile_tag == "2D":

                    # ----------------- global view add newline -----------------
                    # [hw, D] -> [h, w, D]
                    global_features = global_features.view(h, w, n_dim)
                    # [D]     -> [h, 1, D]
                    new_lines_in_global = repeat(self.image_newline, "d -> h 1 d", h=h)
                    # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                    global_features = torch.cat([global_features, new_lines_in_global], dim=1)
                    # [h, w + 1, D] -> [h * (w + 1), D]
                    global_features = global_features.view(-1, n_dim)

                    # ----------------- local view add newline -----------------
                    # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                    local_features = rearrange(
                        local_features,
                        "(th tw) (h w) d -> (th h) (tw w) d",
                        th=num_height_tiles,
                        tw=num_width_tiles,
                        h=h,
                        w=w
                    )

                    # [D] -> [num_height_tiles * h, 1, D]
                    new_lines_in_local = repeat(
                        self.image_newline,
                        "d -> (th h) 1 d",
                        th=num_height_tiles,
                        h=h
                    )

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                    local_features = local_features.view(-1, n_dim)

                    # ----------------- merge global and local tiles -----------------
                    if self.global_view_pos == "head":
                        global_local_features = torch.cat(
                            [global_features, self.view_seperator[None, :], local_features], dim=0)
                    else:
                        global_local_features = torch.cat(
                            [local_features, self.view_seperator[None, :], global_features], dim=0)

                else:
                    # abandoned，实际上不会走这个逻辑
                    global_features = torch.cat(
                        [self.tile_indicators[0:1], global_features], dim=0
                    )
                    local_features = torch.cat(
                        [self.tile_indicators[1:num_tiles_in_image + 1].unsqueeze(1), local_features], dim=1
                    )
                    local_features = rearrange(local_features, 'crop_num hw d -> (crop_num hw) d')

                    if self.global_view_pos == "head":
                        global_local_features = torch.cat([global_features, local_features], dim=0)
                    else:
                        global_local_features = torch.cat([local_features, global_features], dim=0)

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0)
                input_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1), images_in_this_batch)

        return input_embeds

    @torch.no_grad()
    def incremental_prefilling(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,

            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,
            chunk_size: int = 1024
    ):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            del images
            del images_seq_mask
            del images_spatial_crop

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

            self._clear_cuda_cache()

        bzs, seq_len, _ = inputs_embeds.shape
        past_key_values = None

        # remain the last token for the next forward
        prefilling_len = seq_len - 1
        for i in range(0, prefilling_len, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, prefilling_len)
            chunk_inputs_embeds = inputs_embeds[:, chunk_start: chunk_end]
            chunk_attention_mask = attention_mask[:, 0: chunk_end]
            # print(f"start = {chunk_start}, end = {chunk_end}, prefilling_len = {prefilling_len}, seq_len = {seq_len}")

            # compute position_ids
            if past_key_values is not None:
                position_ids = torch.arange(
                    chunk_start,
                    chunk_end,
                    dtype=torch.long,
                    device=inputs_embeds.device
                ).unsqueeze(0)
                past_key_values = self._move_past_key_values_to_gpu(past_key_values, inputs_embeds.device)
            else:
                position_ids = None

            # chunk-forward
            with torch.no_grad():
                outputs = self.forward(
                    inputs_embeds=chunk_inputs_embeds,
                    attention_mask=chunk_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                )
                # update past_key_values
                past_key_values = outputs.past_key_values
                past_key_values = self._move_past_key_values_to_cpu(past_key_values)

                del outputs, position_ids
                self._clear_cuda_cache()

        prefilling_key_values = []
        for layer_past in past_key_values:
            prefilling_key_values.append(
                (
                    layer_past[0][:, :, 0: prefilling_len, ...].to(inputs_embeds.device),
                    layer_past[1][:, :, 0: prefilling_len, ...].to(inputs_embeds.device),
                )
            )

        return inputs_embeds, prefilling_key_values

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,

            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,

            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,

            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # print(inputs_embeds.shape)
        outputs = self.language.forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )

        return outputs

    def _clear_cuda_cache(self):
        """clear CUDA memory cache"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _move_past_key_values_to_cpu(self, past_key_values):
        # print(f"past_key_values -> cpu")
        if past_key_values is None:
            return None
        return tuple(tuple(t.cpu() for t in layer) for layer in past_key_values)

    def _move_past_key_values_to_gpu(self, past_key_values, device="cuda:0"):
        # print(f"past_key_values -> gpu")
        if past_key_values is None:
            return None
        return tuple(tuple(t.to(device) for t in layer) for layer in past_key_values)

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            inputs_embeds=None,

            images: Optional[torch.FloatTensor] = None,
            images_seq_mask: Optional[torch.LongTensor] = None,
            images_spatial_crop: Optional[torch.LongTensor] = None,

            attention_mask=None,
            cache_position=None,

            pixel_values=None,
            image_sizes=None,
            num_logits_to_keep=None,
            **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model
        model_inputs = self.language.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        cache_position = model_inputs["cache_position"]
        if cache_position[0] == 0:
            model_inputs["images"] = images
            model_inputs["images_seq_mask"] = images_seq_mask
            model_inputs["images_spatial_crop"] = images_spatial_crop

        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


AutoConfig.register("vision", VisionEncoderConfig)
AutoConfig.register("mlp_projector", MlpProjectorConfig)
AutoConfig.register("deepseek_vl_v2", DeepseekVLV2Config)
AutoModelForCausalLM.register(DeepseekVLV2Config, DeepseekVLV2ForCausalLM)
