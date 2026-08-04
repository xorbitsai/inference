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
from typing import Any, Generator, List, Optional, Tuple, Union

PDF_MAGIC = b"%PDF"
DEFAULT_PDF_OCR_DPI = 200
# Rasterizing above this resolution rarely helps OCR quality but can
# exhaust memory on large pages.
MAX_PDF_OCR_DPI = 600
# One request OCRs at most this many pages.
MAX_PDF_OCR_PAGES = 200
# A page whose raster would exceed this many pixels (~240 MB of RGB
# pixels; an A3 page at 600 DPI is ~70 MP) is rejected so a single
# oversized page cannot exhaust the API process.
MAX_PDF_OCR_PAGE_PIXELS = 80_000_000


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
) -> Generator[Tuple[int, Any], None, None]:
    """Rasterize a PDF into PIL images, one page at a time.

    Returns an iterator of ``(page_number, PIL.Image)`` tuples, page
    numbers being 1-based. Page selection and page sizes are validated
    eagerly (raising ``ValueError``) so invalid requests fail before any
    OCR runs; rendering itself is lazy so only one page's pixels are
    alive at a time. The caller must exhaust or ``close()`` the iterator
    to release the underlying document.
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
    scale = dpi / 72.0

    pdf = pdfium.PdfDocument(data)
    try:
        page_numbers = normalize_pages(pages, len(pdf))
        if len(page_numbers) > MAX_PDF_OCR_PAGES:
            raise ValueError(
                f"{len(page_numbers)} pages selected, at most "
                f"{MAX_PDF_OCR_PAGES} pages can be OCRed per request; "
                "use `pages` to select a subset"
            )
        for page_number in page_numbers:
            page = pdf[page_number - 1]
            try:
                width, height = page.get_size()
            finally:
                page.close()
            pixels = int(width * scale) * int(height * scale)
            if pixels > MAX_PDF_OCR_PAGE_PIXELS:
                raise ValueError(
                    f"Page {page_number} would rasterize to {pixels} pixels "
                    f"at {dpi:g} DPI, exceeding the limit of "
                    f"{MAX_PDF_OCR_PAGE_PIXELS}; lower `dpi`"
                )
    except Exception:
        pdf.close()
        raise

    def _render() -> Generator[Tuple[int, Any], None, None]:
        try:
            for page_number in page_numbers:
                page = pdf[page_number - 1]
                try:
                    bitmap = page.render(scale=scale)
                    try:
                        image = bitmap.to_pil()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
                yield page_number, image
        finally:
            pdf.close()

    return _render()


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
