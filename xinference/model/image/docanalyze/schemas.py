from typing import List, Optional

from pydantic import BaseModel


class DocAnalyzeBaseResponse(BaseModel):
    type: str
    bbox: List[float]
    page_idx: int


class DocAnalyzeTextResponse(DocAnalyzeBaseResponse):
    text: str


class DocAnalyzeTitleResponse(DocAnalyzeTextResponse):
    text_level: int


class DocAnalyzeListResponse(DocAnalyzeBaseResponse):
    sub_type: str
    list_items: List[str]


class DocAnalyzeEquationResponse(DocAnalyzeBaseResponse):
    text: str
    text_format: str = "latex"


class DocAnalyzeImageResponse(DocAnalyzeBaseResponse):
    image_base64: Optional[str] = ""
    image_caption: List[str] = []
    image_footnote: List[str] = []


class DocAnalyzeTableResponse(DocAnalyzeBaseResponse):
    table_body: Optional[str] = ""
    image_base64: Optional[str] = ""
    table_caption: List[str] = []
    table_footnote: List[str] = []


class DocAnalyzeCodeResponse(DocAnalyzeBaseResponse):
    sub_type: str
    guess_lang: Optional[str] = ""
    table_caption: List[str] = []
    table_footnote: List[str] = []
