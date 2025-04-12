from transformers import AutoConfig, AutoModel
from transformers import SiglipVisionModel, SiglipImageProcessor
from .base_visual_tokenizer import BaseVisualTokenizerConfig, BaseVisualTokenizer

MODEL_TYPE = "siglip_visual_tokenizer"


class SiglipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = MODEL_TYPE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.drop_cls_token:
            self.drop_cls_token = False
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs['num_hidden_layers'] = self.depths[0]


class SiglipVisualTokenizer(BaseVisualTokenizer):
    config_class = SiglipVisualTokenizerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["SiglipVisionTransformer"]
    _image_processor_class = SiglipImageProcessor
    _image_processor_kwargs = {}
    _backbone_class = SiglipVisionModel

    def get_monitor_tensors(self):
        return dict(
            backbone_bottom=self.backbone.vision_model.encoder.layers[0].self_attn.k_proj.weight,
            backbone_top=self.backbone.vision_model.encoder.layers[-1].self_attn.out_proj.weight,
            head=self.head[0].weight
        )

    def get_image_size(self):
        height = self.image_processor.size["height"]
        width = self.image_processor.size["width"]
        return height, width


AutoConfig.register(MODEL_TYPE, SiglipVisualTokenizerConfig)
AutoModel.register(SiglipVisualTokenizerConfig, SiglipVisualTokenizer)
