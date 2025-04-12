from transformers import AutoConfig, AutoModel
from .visual_tokenizer.configuration_aimv2 import AIMv2Config
from .visual_tokenizer.modeling_aimv2 import AIMv2Model
from .visual_tokenizer.clip_visual_tokenizer import ClipVisualTokenizerConfig, ClipVisualTokenizer
from .visual_tokenizer.siglip_visual_tokenizer import SiglipVisualTokenizerConfig, SiglipVisualTokenizer
from .visual_tokenizer.aimv2_visual_tokenizer import Aimv2VisualTokenizerConfig, Aimv2VisualTokenizer

AutoConfig.register('aimv2', AIMv2Config)
AutoModel.register(AIMv2Config, AIMv2Model)
