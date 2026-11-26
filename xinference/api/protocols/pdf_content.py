# Copyright 2022-2026 Xinference Holdings Pte. Ltd
"""Render PDF pages for the chat protocol adapters."""

from __future__ import annotations

import base64
import io
import math
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Dict, List

from ...utils import dependency_import_message

PDF_DATA_URL_PREFIX = "data:application/pdf;base64,"


@dataclass
class PDFRenderBudget:
    """Resource use across all PDFs in one request."""

    pages: int = 0
    pixels: int = 0
    decoded_bytes: int = 0
    rendered_bytes: int = 0

    def add_bytes(self, size: int, *, rendered: bool = False) -> None:
        from ...model.llm.media import _max_bytes

        name = "rendered_bytes" if rendered else "decoded_bytes"
        total = getattr(self, name) + size
        if total > _max_bytes():
            raise ValueError("PDF input exceeds the request byte limit")
        setattr(self, name, total)


def decode_pdf_base64(data: str, budget: PDFRenderBudget) -> bytes:
    from ...model.llm.media import _max_bytes, decode_data_uri

    if len(data) > ((_max_bytes() - budget.decoded_bytes + 2) // 3) * 4:
        raise ValueError("PDF input exceeds the request byte limit")
    return decode_data_uri(PDF_DATA_URL_PREFIX + data)


def render_pdf_pages(raw: bytes, budget: PDFRenderBudget) -> List[Dict[str, Any]]:
    """Render each page as a PNG image content part in document order."""
    budget.add_bytes(len(raw))
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ValueError(
            dependency_import_message(
                "pypdfium2",
                exc,
                install_hint='Install with `pip install "xinference[image]"`.',
            )
        ) from exc
    parts: List[Dict[str, Any]] = []
    from ..pdf_ocr import _pdfium_lock

    # PDFium operations on different documents must also share the OCR lock.
    try:
        with _pdfium_lock, closing(pdfium.PdfDocument(raw)) as pdf:
            pdf.init_forms()
            budget.pages += len(pdf)
            if budget.pages > 20:
                raise ValueError("PDF input supports at most 20 pages per request")
            for page in pdf:
                with closing(page):
                    width, height = page.get_size()
                    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
                        raise ValueError(
                            "PDF page dimensions must be positive and finite"
                        )
                    scale = min(1.5, 2048 / max(width, height))
                    budget.pixels += math.ceil(width * scale) * math.ceil(
                        height * scale
                    )
                    if budget.pixels > 20_000_000:
                        raise ValueError(
                            "PDF input exceeds 20000000 rendered pixels per request"
                        )
                    with closing(page.render(scale=scale)) as bitmap:
                        output = io.BytesIO()
                        bitmap.to_pil().save(output, format="PNG")
                        budget.add_bytes(output.tell(), rendered=True)
                        encoded = base64.b64encode(output.getvalue()).decode("ascii")
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}"
                                },
                            }
                        )
    except pdfium.PdfiumError as exc:
        raise ValueError("Invalid PDF document") from exc
    return parts
