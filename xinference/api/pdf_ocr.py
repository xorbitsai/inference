# Copyright 2022-2026 Xinference Holdings Pte. Ltd
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

"""
Helpers for PDF input on the ``/v1/images/ocr`` endpoint.

OCR models consume a single PIL image, so PDF uploads are rasterized
page by page and the per-page OCR results are merged back into one
JSON-serializable response body.
"""

import json
from typing import Any, List, Optional, Tuple, Union

PDF_MAGIC = b"%PDF"
DEFAULT_PDF_OCR_DPI = 200
# Rasterizing above this resolution rarely helps OCR quality but can
# exhaust memory on large pages.
MAX_PDF_OCR_DPI = 600


def is_pdf_upload(content_type: Optional[str], head: bytes) -> bool:
    """Detect a PDF upload by content type or ``%PDF`` magic bytes."""
    if content_type and content_type.split(";")[0].strip() == "application/pdf":
        return True
    return head.startswith(PDF_MAGIC)


def normalize_pages(
    pages: Optional[Union[int, List[int]]], page_count: int
) -> List[int]:
    """Validate the ``pages`` kwarg and return 1-based page numbers.

    ``None`` selects all pages.
    """
    if pages is None:
        return list(range(1, page_count + 1))
    if isinstance(pages, int) and not isinstance(pages, bool):
        pages = [pages]
    if (
        not isinstance(pages, (list, tuple))
        or not pages
        or not all(isinstance(p, int) and not isinstance(p, bool) for p in pages)
    ):
        raise ValueError(
            "`pages` must be a 1-based page number or a non-empty list of "
            f"1-based page numbers, got: {pages!r}"
        )
    for p in pages:
        if p < 1 or p > page_count:
            raise ValueError(
                f"Page {p} is out of range, the PDF has {page_count} page(s)"
            )
    return list(pages)


def rasterize_pdf(
    data: bytes,
    pages: Optional[Union[int, List[int]]] = None,
    dpi: Union[int, float] = DEFAULT_PDF_OCR_DPI,
) -> List[Tuple[int, Any]]:
    """Rasterize a PDF into PIL images.

    Returns a list of ``(page_number, PIL.Image)`` tuples, page numbers
    being 1-based.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        error_message = "Failed to import module 'pypdfium2' required for PDF OCR"
        installation_guide = [
            "Please make sure 'pypdfium2' is installed, e.g. with `pip install pypdfium2`\n",
        ]
        raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

    if not isinstance(dpi, (int, float)) or isinstance(dpi, bool) or dpi <= 0:
        raise ValueError(f"`dpi` must be a positive number, got: {dpi!r}")
    dpi = min(float(dpi), float(MAX_PDF_OCR_DPI))

    pdf = pdfium.PdfDocument(data)
    try:
        page_numbers = normalize_pages(pages, len(pdf))
        images = []
        for page_number in page_numbers:
            page = pdf[page_number - 1]
            try:
                bitmap = page.render(scale=dpi / 72.0)
                try:
                    images.append((page_number, bitmap.to_pil()))
                finally:
                    bitmap.close()
            finally:
                page.close()
        return images
    finally:
        pdf.close()


def merge_ocr_page_results(page_results: List[Tuple[int, Any]]) -> str:
    """Merge per-page OCR results into one JSON response body.

    When every page yields plain text, the texts are joined so the
    response stays a JSON string like the single-image contract.
    Otherwise (e.g. ``return_dict=True`` style models) a per-page
    structure is returned. Either way the body parses with
    ``response.json()``.
    """
    if all(isinstance(result, str) for _, result in page_results):
        joined = "\n\n".join(result for _, result in page_results)
        return json.dumps(joined, ensure_ascii=False)
    merged = {
        "pages": [
            {"page": page_number, "result": result}
            for page_number, result in page_results
        ]
    }
    return json.dumps(merged, ensure_ascii=False)
