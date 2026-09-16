# Copyright 2022-2026 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os

from packaging import version

try:
    import vllm  # noqa: F401

    if not getattr(vllm, "__version__", None):
        raise ImportError(
            "vllm not installed properly, or wrongly be found in sys.path"
        )

    VLLM_INSTALLED = True
    VLLM_VERSION = version.parse(vllm.__version__)
except ImportError:
    VLLM_INSTALLED = False
    VLLM_VERSION = None

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..core import ImageModelFamilyV2

logger = logging.getLogger(__name__)


class Mineru2_5Model:

    def __init__(
        self,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = "cuda",
        model_spec: Optional["ImageModelFamilyV2"] = None,
        **kwargs,
    ):
        self.model_family = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._model = None
        self._tokenizer = None
        self._model_spec = model_spec
        self._device = device
        self._abilities = model_spec.model_ability or []  # type: ignore
        self._kwargs = kwargs

    @property
    def model_ability(self):
        return self._abilities

    def load(self):
        from ....thirdparty.mineru.backend.vlm.custom_logits_processors import (
            enable_custom_logits_processors,
        )
        from ....thirdparty.mineru_vl_utils import MinerUClient, MinerULogitsProcessor

        backend = self._kwargs.pop("backend", "vllm-async-engine")
        if backend == "vllm-engine":
            raise Exception(
                "vlm-vllm-engine backend is not supported in async mode, please use vlm-vllm-async-engine backend"
            )

        if (
            backend == "vllm-async-engine"
            and VLLM_INSTALLED
            and VLLM_VERSION < version.parse("0.10.1")
        ):
            raise Exception(
                f"vllm version: {VLLM_VERSION} < 0.10.1, disable vlm-async-engine backend, please upgrade vllm to >=0.10.1"
            )

        self._kwargs.pop("cpu_offload", None)

        model = None
        processor = None
        vllm_llm = None
        vllm_async_llm = None
        if backend in ["transformers", "vllm-async-engine"]:
            if backend == "transformers":
                try:
                    from transformers import (
                        AutoProcessor,
                        Qwen2VLForConditionalGeneration,
                    )
                    from transformers import __version__ as transformers_version
                except ImportError:
                    raise ImportError(
                        "Please install transformers to use the transformers backend."
                    )

                if version.parse(transformers_version) >= version.parse("4.56.0"):
                    dtype_key = "dtype"
                else:
                    dtype_key = "torch_dtype"
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self._model_path,
                    device_map={"": self._device},
                    **{dtype_key: "auto"},  # type: ignore
                )
                processor = AutoProcessor.from_pretrained(
                    self._model_path,
                    use_fast=True,
                )

            elif backend == "vllm-async-engine":
                try:

                    from vllm.engine.arg_utils import AsyncEngineArgs
                    from vllm.v1.engine.async_llm import AsyncLLM
                except ImportError:
                    raise ImportError(
                        "Please install vllm to use the vllm-async-engine backend."
                    )
                if "gpu_memory_utilization" not in self._kwargs:
                    self._kwargs["gpu_memory_utilization"] = 0.5
                if "model" not in self._kwargs:
                    self._kwargs["model"] = self._model_path
                if enable_custom_logits_processors() and (
                    "logits_processors" not in self._kwargs
                ):

                    self._kwargs["logits_processors"] = [MinerULogitsProcessor]
                os.environ["VLLM_USE_V1"] = "1"
                # 使用kwargs为 vllm初始化参数
                vllm_async_llm = AsyncLLM.from_engine_args(
                    AsyncEngineArgs(**self._kwargs)
                )

        self._model = MinerUClient(
            backend=backend,
            model=model,
            processor=processor,
            vllm_llm=vllm_llm,
            vllm_async_llm=vllm_async_llm,
            server_url=None,
            batch_size=8,
        )

    async def docanalyze(
        self,
        file_bytes: bytes,
        file_name: str,
        **kwargs,
    ) -> Dict[str, Any]:
        # 处理逻辑来自parse_pdf函数 具体参考mineru源码
        from ....thirdparty.mineru.cli.common import (
            convert_pdf_bytes_to_bytes_by_pypdfium2,
        )
        from ....thirdparty.mineru.utils.enum_class import ImageType, MakeMode
        from ....thirdparty.mineru.utils.pdf_image_tools import load_images_from_pdf

        if self._model is None:
            raise RuntimeError("Model not loaded")
        file_bytes = read_fn(file_bytes, file_name)
        # 预处理PDF字节数据
        pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(file_bytes)

        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        images_base64_list = [image_dict["img_pil"] for image_dict in images_list]

        results = await self._model.aio_batch_two_step_extract(
            images=images_base64_list
        )
        middle_json = result_to_middle_json(results, images_list, pdf_doc)
        pdf_info = middle_json["pdf_info"]
        content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST)

        check_json_structure(content_list)
        return content_list


def read_fn(file_bytes: bytes, file_name: str):
    from ....thirdparty.mineru.cli.common import image_suffixes, pdf_suffixes
    from ....thirdparty.mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes

    file_suffix = f"{file_name.split('.')[-1].lower()}"
    if file_suffix in pdf_suffixes:
        pass
    elif file_suffix in image_suffixes:
        file_bytes = images_bytes_to_pdf_bytes(file_bytes)
    else:
        raise Exception(f"Unknown file suffix: {file_suffix}")
    return file_bytes


def result_to_middle_json(token_list, images_list, pdf_doc):
    from ....thirdparty.mineru.utils.config_reader import get_table_enable
    from ....thirdparty.mineru.utils.table_merge import merge_table
    from ....thirdparty.mineru.version import __version__

    middle_json = {"pdf_info": [], "_backend": "vlm", "_version_name": __version__}
    for index, token in enumerate(token_list):
        page = pdf_doc[index]
        image_dict = images_list[index]
        page_info = blocks_to_page_info(token, image_dict, page, index)
        middle_json["pdf_info"].append(page_info)

    """表格跨页合并"""
    table_enable = get_table_enable(
        os.getenv("MINERU_VLM_TABLE_ENABLE", "True").lower() == "true"
    )
    if table_enable:
        merge_table(middle_json["pdf_info"])

    # 关闭pdf文档
    pdf_doc.close()
    return middle_json


def blocks_to_page_info(token, image_dict, page, page_index):
    """将token转换为页面信息"""
    # 解析token，提取坐标和类型
    # 假设token格式为：asp x0 y0 x1 y1 type content
    # 这里需要根据实际的token格式进行解析
    # 提取所有完整块，每个块从asp开始到</content>或</span>结束
    from ....thirdparty.mineru.backend.vlm.vlm_magic_model import MagicModel
    from ....thirdparty.mineru.utils.enum_class import ContentType

    scale = image_dict["scale"]
    # page_pil_img = base64_to_pil_image(image_dict["img_base64"])
    page_pil_img = image_dict["img_pil"]
    width, height = map(int, page.get_size())

    magic_model = MagicModel(token, width, height)
    image_blocks = magic_model.get_image_blocks()
    table_blocks = magic_model.get_table_blocks()
    title_blocks = magic_model.get_title_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    code_blocks = magic_model.get_code_blocks()
    ref_text_blocks = magic_model.get_ref_text_blocks()
    phonetic_blocks = magic_model.get_phonetic_blocks()
    list_blocks = magic_model.get_list_blocks()

    text_blocks = magic_model.get_text_blocks()
    interline_equation_blocks = magic_model.get_interline_equation_blocks()

    all_spans = magic_model.get_all_spans()
    # 对image/table/interline_equation的span截图
    for span in all_spans:
        if span["type"] in [
            ContentType.IMAGE,
            ContentType.TABLE,
            ContentType.INTERLINE_EQUATION,
        ]:
            span["image_base64"] = cut_image(span["bbox"], page_pil_img, scale=scale)

    page_blocks = []
    page_blocks.extend(
        [
            *image_blocks,
            *table_blocks,
            *code_blocks,
            *ref_text_blocks,
            *phonetic_blocks,
            *title_blocks,
            *text_blocks,
            *interline_equation_blocks,
            *list_blocks,
        ]
    )
    # 对page_blocks根据index的值进行排序
    page_blocks.sort(key=lambda x: x["index"])

    page_info = {
        "para_blocks": page_blocks,
        "discarded_blocks": discarded_blocks,
        "page_size": [width, height],
        "page_idx": page_index,
    }
    return page_info


def cut_image(bbox: tuple, page_pil_img, scale=2):
    """从第page_num页的page中，根据bbox进行裁剪出一张jpg图片，返回图片路径 save_path：需要同时支持s3和本地,
    图片存放在save_path下，文件名是:
    {page_num}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg , bbox内数字取整。"""
    from ....thirdparty.mineru.utils.pdf_image_tools import get_crop_img
    from ....thirdparty.mineru.utils.pdf_reader import image_to_b64str

    crop_img = get_crop_img(bbox, page_pil_img, scale=scale)

    img_b64str = image_to_b64str(crop_img, image_format="JPEG")

    return img_b64str


def vlm_union_make(
    pdf_info_dict: list,
    make_mode: str,
):
    from ....thirdparty.mineru.utils.enum_class import MakeMode

    output_content = []
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get("para_blocks")
        paras_of_discarded = page_info.get("discarded_blocks")
        page_idx = page_info.get("page_idx")
        page_size = page_info.get("page_size")
        if not paras_of_layout:
            continue
        if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
            # from mineru.utils.config_reader import get_formula_enable, get_table_enable
            # formula_enable = get_formula_enable(os.getenv('MINERU_VLM_FORMULA_ENABLE', 'True').lower() == 'true')
            # table_enable = get_table_enable(os.getenv('MINERU_VLM_TABLE_ENABLE', 'True').lower() == 'true')

            # page_markdown = mk_blocks_to_markdown(paras_of_layout, make_mode, formula_enable, table_enable, img_buket_path)
            # output_content.extend(page_markdown)
            pass
        elif make_mode == MakeMode.CONTENT_LIST:
            for para_block in paras_of_layout + paras_of_discarded:
                para_content = make_blocks_to_content_list(
                    para_block, page_idx, page_size
                )
                output_content.append(para_content)

    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return "\n\n".join(output_content)
    elif make_mode == MakeMode.CONTENT_LIST:
        return output_content
    return None


def make_blocks_to_content_list(para_block, page_idx, page_size):
    from ....thirdparty.mineru.utils.enum_class import BlockType, ContentType

    para_type = para_block["type"]
    para_content = {}
    if para_type in [
        BlockType.TEXT,
        BlockType.REF_TEXT,
        BlockType.PHONETIC,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_FOOTNOTE,
    ]:
        para_content = {
            "type": para_type,
            "text": merge_para_with_text(para_block),
        }
    elif para_type == BlockType.LIST:
        para_content = {
            "type": para_type,
            "sub_type": para_block.get("sub_type", ""),
            "list_items": [],
        }
        for block in para_block["blocks"]:
            item_text = merge_para_with_text(block)
            if item_text.strip():
                para_content["list_items"].append(item_text)
    elif para_type == BlockType.TITLE:
        title_level = get_title_level(para_block)
        para_content = {
            "type": ContentType.TEXT,
            "text": merge_para_with_text(para_block),
        }
        if title_level != 0:
            para_content["text_level"] = title_level
    elif para_type == BlockType.INTERLINE_EQUATION:
        para_content = {
            "type": ContentType.EQUATION,
            "text": merge_para_with_text(para_block),
            "text_format": "latex",
        }
    elif para_type == BlockType.IMAGE:
        para_content = {
            "type": ContentType.IMAGE,
            "image_base64": "",
            BlockType.IMAGE_CAPTION: [],
            BlockType.IMAGE_FOOTNOTE: [],
        }
        for block in para_block["blocks"]:
            if block["type"] == BlockType.IMAGE_BODY:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["type"] == ContentType.IMAGE:
                            if span.get("image_base64", ""):
                                para_content["image_base64"] = span["image_base64"]
            if block["type"] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(
                    merge_para_with_text(block)
                )
            if block["type"] == BlockType.IMAGE_FOOTNOTE:
                para_content[BlockType.IMAGE_FOOTNOTE].append(
                    merge_para_with_text(block)
                )
    elif para_type == BlockType.TABLE:
        para_content = {
            "type": ContentType.TABLE,
            "image_base64": "",
            BlockType.TABLE_CAPTION: [],
            BlockType.TABLE_FOOTNOTE: [],
        }
        for block in para_block["blocks"]:
            if block["type"] == BlockType.TABLE_BODY:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["type"] == ContentType.TABLE:

                            if span.get("html", ""):
                                para_content[BlockType.TABLE_BODY] = f"{span['html']}"

                            if span.get("image_base64", ""):
                                para_content["image_base64"] = span["image_base64"]

            if block["type"] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(
                    merge_para_with_text(block)
                )
            if block["type"] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(
                    merge_para_with_text(block)
                )
    elif para_type == BlockType.CODE:
        para_content = {
            "type": BlockType.CODE,
            "sub_type": para_block["sub_type"],
            BlockType.CODE_CAPTION: [],
        }
        for block in para_block["blocks"]:
            if block["type"] == BlockType.CODE_BODY:
                para_content[BlockType.CODE_BODY] = merge_para_with_text(block)
                if para_block["sub_type"] == BlockType.CODE:
                    para_content["guess_lang"] = para_block["guess_lang"]
            if block["type"] == BlockType.CODE_CAPTION:
                para_content[BlockType.CODE_CAPTION].append(merge_para_with_text(block))

    page_weight, page_height = page_size
    para_bbox = para_block.get("bbox")
    if para_bbox:
        x0, y0, x1, y1 = para_bbox
        para_content["bbox"] = [
            int(x0 * 1000 / page_weight),
            int(y0 * 1000 / page_height),
            int(x1 * 1000 / page_weight),
            int(y1 * 1000 / page_height),
        ]

    para_content["page_idx"] = page_idx

    return para_content


def merge_para_with_text(para_block, formula_enable=True):
    from ....thirdparty.mineru.utils.enum_class import ContentType

    default_delimiters = {
        "display": {"left": "$$", "right": "$$"},
        "inline": {"left": "$", "right": "$"},
    }

    delimiters = default_delimiters

    display_left_delimiter = delimiters["display"]["left"]
    display_right_delimiter = delimiters["display"]["right"]
    inline_left_delimiter = delimiters["inline"]["left"]
    inline_right_delimiter = delimiters["inline"]["right"]

    para_text = ""
    for line in para_block["lines"]:
        for j, span in enumerate(line["spans"]):
            span_type = span["type"]
            content = ""
            if span_type == ContentType.TEXT:
                content = span["content"]
            elif span_type == ContentType.INLINE_EQUATION:
                content = (
                    f"{inline_left_delimiter}{span['content']}{inline_right_delimiter}"
                )
            elif span_type == ContentType.INTERLINE_EQUATION:
                if formula_enable:
                    content = f"\n{display_left_delimiter}\n{span['content']}\n{display_right_delimiter}\n"
                else:
                    if span.get("image_base64", ""):
                        content = span["image_base64"]
            # content = content.strip()
            if content:
                if span_type in [ContentType.TEXT, ContentType.INLINE_EQUATION]:
                    if j == len(line["spans"]) - 1:
                        para_text += content
                    else:
                        para_text += f"{content} "
                elif span_type == ContentType.INTERLINE_EQUATION:
                    para_text += content
    return para_text


def get_title_level(block):
    title_level = block.get("level", 1)
    if title_level > 4:
        title_level = 4
    elif title_level < 1:
        title_level = 0
    return title_level


def check_json_structure(json_list):

    from ....thirdparty.mineru.utils.enum_class import BlockType, ContentType
    from .schemas import (
        DocAnalyzeCodeResponse,
        DocAnalyzeEquationResponse,
        DocAnalyzeImageResponse,
        DocAnalyzeListResponse,
        DocAnalyzeTableResponse,
        DocAnalyzeTextResponse,
        DocAnalyzeTitleResponse,
    )

    for item in json_list:
        if item["type"] in [
            BlockType.TEXT,
            BlockType.REF_TEXT,
            BlockType.PHONETIC,
            BlockType.HEADER,
            BlockType.FOOTER,
            BlockType.PAGE_NUMBER,
            BlockType.ASIDE_TEXT,
            BlockType.PAGE_FOOTNOTE,
            BlockType.TITLE,
        ]:
            if "text_level" in item:
                DocAnalyzeTitleResponse(**item)
            else:
                DocAnalyzeTextResponse(**item)
        elif item["type"] == BlockType.LIST:
            DocAnalyzeListResponse(**item)
        elif item["type"] == ContentType.EQUATION:
            DocAnalyzeEquationResponse(**item)
        elif item["type"] == ContentType.IMAGE:
            DocAnalyzeImageResponse(**item)
        elif item["type"] == ContentType.TABLE:
            DocAnalyzeTableResponse(**item)
        elif item["type"] == BlockType.CODE:
            DocAnalyzeCodeResponse(**item)
        else:
            raise ValueError(f"Unknown type: {item['type']}")
