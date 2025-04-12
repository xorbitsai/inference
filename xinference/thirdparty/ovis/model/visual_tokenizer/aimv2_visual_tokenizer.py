from transformers import AutoConfig, AutoModel
from transformers import CLIPImageProcessor
from .modeling_aimv2 import AIMv2Model
from .base_visual_tokenizer import BaseVisualTokenizerConfig, BaseVisualTokenizer

MODEL_TYPE = "aimv2_visual_tokenizer"


class Aimv2VisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = MODEL_TYPE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.drop_cls_token:
            self.drop_cls_token = False
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs['num_hidden_layers'] = self.depths[0]


class Aimv2VisualTokenizer(BaseVisualTokenizer):
    config_class = Aimv2VisualTokenizerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["AIMv2ViTPreprocessor", "AIMv2Block"]
    _image_processor_class = CLIPImageProcessor
    _image_processor_kwargs = dict(do_center_crop=False)
    _backbone_class = AIMv2Model

    def get_monitor_tensors(self):
        return dict(
            backbone_bottom=self.backbone.trunk.blocks[0].attn.qkv.weight,
            backbone_top=self.backbone.trunk.blocks[-1].attn.qkv.weight,
            head=self.head[0].weight
        )

    def get_image_size(self):
        height = self.image_processor.crop_size["height"]
        width = self.image_processor.crop_size["width"]
        return height, width


AutoConfig.register(MODEL_TYPE, Aimv2VisualTokenizerConfig)
AutoModel.register(Aimv2VisualTokenizerConfig, Aimv2VisualTokenizer)
