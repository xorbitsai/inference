from transformers import AutoConfig, AutoModel
from transformers import CLIPVisionModel, CLIPImageProcessor
from .base_visual_tokenizer import BaseVisualTokenizerConfig, BaseVisualTokenizer

MODEL_TYPE = "clip_visual_tokenizer"


class ClipVisualTokenizerConfig(BaseVisualTokenizerConfig):
    model_type = MODEL_TYPE

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.depths:
            assert len(self.depths) == 1
            self.backbone_kwargs['num_hidden_layers'] = self.depths[0]


class ClipVisualTokenizer(BaseVisualTokenizer):
    config_class = ClipVisualTokenizerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["CLIPEncoderLayer"]
    _image_processor_class = CLIPImageProcessor
    _image_processor_kwargs = dict(do_center_crop=False)
    _backbone_class = CLIPVisionModel

    def get_monitor_tensors(self):
        return dict(
            backbone_bottom=self.backbone.vision_model.encoder.layers[0].self_attn.k_proj.weight,
            backbone_top=self.backbone.vision_model.encoder.layers[-1].self_attn.out_proj.weight,
            head=self.head[0].weight
        )

    def get_image_size(self):
        height = self.image_processor.crop_size["height"]
        width = self.image_processor.crop_size["width"]
        return height, width


AutoConfig.register(MODEL_TYPE, ClipVisualTokenizerConfig)
AutoModel.register(ClipVisualTokenizerConfig, ClipVisualTokenizer)
